/* 3D 地圖(CesiumJS)— 由 ui/map3d.py 的 Map3DWidget 載入。
 *
 * Python → JS:app.setScene(scene, reason) / app.setTime(t) / app.setOptions(o) / app.viewAll(sec)
 * JS → Python(QWebChannel 物件 bridge):ready(ok, reason) / seek(t) / status(msg) / error(msg)
 *
 * 場景內時間皆為「相對 t_base 的秒」;setTime 與 seek 用 log 絕對秒。
 * 高度 h = 原點地面高(S.ground)+ u(ENU 天向)。
 */
"use strict";

// Cesium 來源:預設 CDN;頁面網址可帶 ?cesium=<Build/Cesium/ 目錄網址>(離線時指向本機副本)
const CESIUM_BASE = (() => {
  try {
    const q = new URLSearchParams(location.search).get("cesium");
    if (q) return q.endsWith("/") ? q : q + "/";
  } catch (e) { /* 預設 */ }
  return "https://cdn.jsdelivr.net/npm/cesium@1.146.0/Build/Cesium/";
})();
const ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const LOAD_TIMEOUT_MS = 30000;
const TRAIL_S = 10;          // 播放頭前的高亮尾巴長度(秒)
const PICK_PX = 12;          // 點擊軌跡的容許距離(像素)
const OUT_COLOR = "#6b6b68"; // 誤差著色時,分析區間外的軌跡

/** 輔助線(估測細線、尾巴、誤差連線、垂直線)的墨色:衛星圖上用白、淺色街道圖上用深灰。 */
function inkHex() {
  return S.applied.base && S.applied.base.startsWith("osm") ? "#262624" : "#ffffff";
}
function lightBase() {
  return !!(S.applied.base && S.applied.base.startsWith("osm"));
}

const S = {
  bridge: null, viewer: null, ready: false, available: false, reason: "",
  scene: null, sceneKey: "",
  ref: null, est: null, att: null, err: null, modes: null,
  t: null,
  opts: { base: "esri", color: "mode", showEst: true, follow: false, token: "" },
  applied: { base: null, token: null },
  ion: { img: "", ter: "" },   // Cesium ion 影像 / 地形狀態:'' / loading / ok / fail
  ionMsg: null,                // 上次送出的 ion 狀態文字
  ground: 0,
  fitKey: null,                // 新 log 的全覽要等 ion 地形的地面高取樣完再做一次(sceneKey)
  pendingViewAll: null,
  prim: { ref: null, est: null },
  dyn: null,
  model: null, modelKey: null, modelReady: false,
  following: false,
  linePts: null,
  hudHtml: "",
  errors: [],
};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- 回報 / 訊息

function report(fn, ...args) {
  try {
    if (S.bridge && typeof S.bridge[fn] === "function") S.bridge[fn](...args);
  } catch (e) {
    console.error("bridge." + fn + " 失敗", e);
  }
}

function reportError(where, e) {
  const msg = where + ":" + (e && e.message ? e.message : String(e));
  console.error(msg, e && e.stack ? e.stack : "");
  S.errors.push(msg);
  if (S.errors.length > 50) S.errors.shift();
  report("error", msg);
}

function overlay(kind, title, msg, retry) {
  const ov = $("overlay");
  if (!kind) { ov.classList.add("hidden"); return; }
  ov.classList.remove("hidden", "warn", "empty");
  if (kind === "warn") ov.classList.add("warn");
  if (kind === "empty") ov.classList.add("empty");
  $("ov-spin").classList.toggle("hidden", kind !== "loading");
  $("ov-title").textContent = title;
  $("ov-msg").textContent = msg || "";
  let btn = $("ov-retry");
  if (retry && !btn) {
    btn = document.createElement("button");
    btn.id = "ov-retry";
    btn.textContent = "重新載入";
    btn.style.cssText = "margin-top:14px;padding:5px 16px;border-radius:6px;border:1px solid #3a3a37;" +
      "background:#2f2f2d;color:#fff;font:inherit;cursor:pointer;pointer-events:auto";
    btn.onclick = () => location.reload();
    ov.querySelector(".box").appendChild(btn);
  } else if (!retry && btn) {
    btn.remove();
  }
}

window.addEventListener("error", (ev) => reportError("JS 例外", ev.error || ev.message));
window.addEventListener("unhandledrejection", (ev) => reportError("非同步錯誤", ev.reason));

// ---------------------------------------------------------------- 啟動

function hasWebGL() {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch (e) {
    return false;
  }
}

function initChannel() {
  return new Promise((resolve) => {
    try {
      if (typeof QWebChannel === "undefined" || typeof qt === "undefined" || !qt.webChannelTransport) {
        resolve(null);
        return;
      }
      new QWebChannel(qt.webChannelTransport, (ch) => resolve(ch.objects.bridge || null));
    } catch (e) {
      console.error("QWebChannel 初始化失敗", e);
      resolve(null);
    }
  });
}

function loadCesium() {
  return new Promise((resolve) => {
    window.CESIUM_BASE_URL = CESIUM_BASE;
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = CESIUM_BASE + "Widgets/widgets.css";
    document.head.insertBefore(css, document.head.firstChild);
    const s = document.createElement("script");
    s.src = CESIUM_BASE + "Cesium.js";
    s.async = true;
    let done = false;
    const fin = (ok) => {
      if (done) return;
      done = true;
      resolve(ok && typeof window.Cesium !== "undefined");
    };
    s.onload = () => fin(true);
    s.onerror = () => fin(false);
    setTimeout(() => fin(false), LOAD_TIMEOUT_MS);
    document.head.appendChild(s);
  });
}

async function boot() {
  const webgl = hasWebGL();
  const [bridge, cesium] = await Promise.all([initChannel(), webgl ? loadCesium() : Promise.resolve(false)]);
  S.bridge = bridge;
  if (!webgl) return fail("nowebgl");
  if (!cesium) return fail("offline");
  try {
    createViewer();
  } catch (e) {
    console.error("Cesium Viewer 建立失敗:" + (e && e.message ? e.message : String(e)));
    return fail(/webgl/i.test(String(e && e.message)) ? "nowebgl" : "error", e);
  }
  S.ready = true;
  S.available = true;
  overlay("empty", "尚未有軌跡", "載入 .bin 或模擬資料後,參考軌跡會畫在這裡");
  report("ready", true, "");
}

function fail(reason, e) {
  S.ready = true;
  S.available = false;
  S.reason = reason;
  if (reason === "nowebgl") {
    overlay("warn", "此環境不支援 WebGL",
      "3D 地圖需要 WebGL(GPU 加速繪圖)。\n遠端桌面、虛擬機或停用硬體加速時常見;其他分頁不受影響。");
  } else if (reason === "offline") {
    overlay("warn", "無法載入 3D 地圖",
      "需要網路下載 Cesium 與衛星圖。\n請確認網路連線(或允許 cdn.jsdelivr.net、server.arcgisonline.com)後重新載入。", true);
  } else {
    overlay("warn", "3D 地圖初始化失敗", (e && e.message) ? e.message : String(e || ""), true);
  }
  report("ready", false, reason);
}

// ---------------------------------------------------------------- Viewer

function createViewer() {
  const C = Cesium;
  // ion 的請求一律明確帶使用者的 token;清掉 Cesium 內建的示範 token,任何漏網的 ion 請求
  // 都會明確失敗,而不是偷偷用示範 token(並在地圖下方掛「default ion access token」橫幅)
  C.Ion.defaultAccessToken = "";
  const viewer = new C.Viewer("viewer", {
    baseLayer: false,
    terrainProvider: new C.EllipsoidTerrainProvider(),
    animation: false, timeline: false, baseLayerPicker: false, geocoder: false,
    homeButton: false, sceneModePicker: false, navigationHelpButton: false,
    fullscreenButton: false, infoBox: false, selectionIndicator: false,
    vrButton: false, projectionPicker: false,
    scene3DOnly: true, shouldAnimate: false,
    requestRenderMode: true, maximumRenderTimeChange: Infinity,
    showRenderLoopErrors: false, msaaSamples: 4,
  });
  S.viewer = viewer;
  const sc = viewer.scene;
  if (window.devicePixelRatio > 1 && window.devicePixelRatio <= 2) {
    viewer.useBrowserRecommendedResolution = false;      // 高 DPI 螢幕上線條清楚
  }
  sc.backgroundColor = C.Color.fromCssColorString("#0d0d0d");
  sc.globe.baseColor = C.Color.fromCssColorString("#1f1f1e");
  sc.globe.enableLighting = false;
  sc.globe.depthTestAgainstTerrain = false;               // 軌跡永遠畫在地表之上
  sc.globe.maximumScreenSpaceError = 1.5;                 // 影像清楚一點
  sc.highDynamicRange = false;
  sc.postProcessStages.fxaa.enabled = true;
  sc.screenSpaceCameraController.minimumZoomDistance = 2.0;
  sc.screenSpaceCameraController.maximumZoomDistance = 2.5e7;
  viewer.cesiumWidget.screenSpaceEventHandler.removeInputAction(C.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

  // 光源:跟著相機,從相機左後上方往下打(不受當地晝夜影響;模型永遠看得清楚)
  const light = new C.DirectionalLight({ direction: new C.Cartesian3(0, 0, -1), intensity: 4.5 });
  sc.light = light;
  const d = new C.Cartesian3(), tmp = new C.Cartesian3(), up = new C.Cartesian3();
  sc.preRender.addEventListener(() => {
    const cam = sc.camera;
    C.Ellipsoid.WGS84.geodeticSurfaceNormal(cam.positionWC, up);
    C.Cartesian3.multiplyByScalar(cam.directionWC, 0.6, d);
    C.Cartesian3.multiplyByScalar(up, -0.85, tmp);
    C.Cartesian3.add(d, tmp, d);
    C.Cartesian3.multiplyByScalar(cam.rightWC, 0.3, tmp);
    C.Cartesian3.add(d, tmp, d);
    C.Cartesian3.normalize(d, light.direction);
  });

  // 尾巴與軌跡重疊會深度打架(閃爍成虛線):每個影格把尾巴沿視線往相機挪一點點,
  // 螢幕上位置不變,但永遠畫在軌跡前面
  const toCam = new C.Cartesian3();
  sc.preRender.addEventListener(() => {
    if (!S.dyn || !S.dyn.trail.show || !S.trailRaw) return;
    const cam = sc.camera.positionWC;
    const out = S.trailRaw.map((p) => {
      C.Cartesian3.subtract(cam, p, toCam);
      return C.Cartesian3.add(p, C.Cartesian3.multiplyByScalar(toCam, 0.004, toCam), new C.Cartesian3());
    });
    S.dyn.trail.positions = out;
  });

  sc.renderError.addEventListener((_scene, e) => {
    reportError("繪圖錯誤", e);
    overlay("warn", "3D 繪圖發生錯誤", (e && e.message) ? e.message : String(e), true);
  });

  const handler = new C.ScreenSpaceEventHandler(sc.canvas);
  handler.setInputAction((ev) => {
    try { pickSeek(ev.position); } catch (e) { reportError("點擊軌跡", e); }
  }, C.ScreenSpaceEventType.LEFT_CLICK);
  // 使用者動了相機:取消「地面高到了再全覽」,不蓋掉使用者的視角
  const userCam = () => { S.fitKey = null; };
  const T = C.ScreenSpaceEventType, K = C.KeyboardEventModifier;
  for (const type of [T.LEFT_DOWN, T.RIGHT_DOWN, T.MIDDLE_DOWN, T.WHEEL, T.PINCH_START]) {
    for (const mod of [undefined, K.SHIFT, K.CTRL, K.ALT]) handler.setInputAction(userCam, type, mod);
  }

  sc.camera.setView({ destination: C.Cartesian3.fromDegrees(121.0, 23.7, 9.0e6) });
  applyBase();
  applyTerrain();

  // 畫布尺寸改變(分頁第一次顯示、視窗縮放):補做延後的全覽並重畫
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      try { flushPendingViewAll(); requestRender(); } catch (e) { reportError("resize", e); }
    }).observe(viewer.container);
  }
}

function requestRender() {
  if (S.viewer) S.viewer.scene.requestRender();
}

// ---------------------------------------------------------------- 底圖 / 地形

function makeBaseLayer(kind) {
  const C = Cesium;
  if (kind === "osm") {
    return new C.ImageryLayer(new C.OpenStreetMapImageryProvider({
      url: "https://tile.openstreetmap.org/", maximumLevel: 19,    // OSM 圖磚只到 z19
      credit: new C.Credit("© OpenStreetMap contributors", true),
    }));
  }
  if (kind === "ion") {
    // 明確帶使用者的 token:fromWorldImagery 會用 Ion.defaultAccessToken(Cesium 內建示範 token),
    // 而且端點依 token 快取 —— 不帶的話使用者的 token 根本沒用到、換 token 也不會生效
    const token = S.opts.token;
    const layer = C.ImageryLayer.fromProviderAsync(
      C.IonImageryProvider.fromAssetId(C.IonWorldImageryStyle.AERIAL_WITH_LABELS, { accessToken: token }));
    S.ion.img = "loading";
    pushIonStatus();
    const current = () => !!(S.viewer && S.viewer.imageryLayers.contains(layer));
    layer.readyEvent.addEventListener(() => {
      if (!current()) return;
      S.ion.img = "ok";
      pushIonStatus();
    });
    layer.errorEvent.addEventListener((e) => {
      // 失敗的端點請求也被快取:清掉,之後(網路恢復、重選 ion)才會真的重試
      try {
        const cache = C.IonImageryProvider._endpointCache || {};
        for (const k of Object.keys(cache)) if (k.indexOf(String(token)) >= 0) delete cache[k];
      } catch (_) { /* 內部結構改了就算了 */ }
      if (!current()) return;                     // 使用者已換別的底圖
      reportError("Cesium ion 影像載入失敗(token 無效或無網路)", e);
      S.ion.img = "fail";
      S.applied.base = null;
      S.opts.base = "esri";
      applyBase();
      pushIonStatus();
    });
    return layer;
  }
  // 影像沒有涵蓋的層級會回傳「Map data not yet available」灰圖:比對已知缺圖的像素丟掉,
  // Cesium 會改用上一層影像放大(與 ArcGisMapServerImageryProvider 的做法相同)
  const missing = ESRI_URL.replace("{z}", "19").replace("{y}", "0").replace("{x}", "0");
  return new C.ImageryLayer(new C.UrlTemplateImageryProvider({
    url: ESRI_URL, maximumLevel: 20,
    credit: new C.Credit("Esri, Maxar, Earthstar Geographics", true),
    tileDiscardPolicy: new C.DiscardMissingTileImagePolicy({
      missingImageUrl: missing,
      pixelsToCheck: [new C.Cartesian2(0, 0), new C.Cartesian2(200, 0), new C.Cartesian2(0, 200),
                      new C.Cartesian2(200, 200), new C.Cartesian2(100, 100)],
      disableCheckIfAllPixelsAreTransparent: true,
    }),
  }));
}

function applyBase() {
  if (!S.viewer) return;
  const kind = (S.opts.base === "ion" && !S.opts.token) ? "esri" : (S.opts.base || "esri");
  const key = kind + (kind === "ion" ? "|" + S.opts.token : "");
  if (S.applied.base === key) return;
  S.applied.base = key;
  if (kind !== "ion" && S.ion.img !== "fail") S.ion.img = "";
  try {
    const layers = S.viewer.imageryLayers;
    layers.removeAll(true);
    layers.add(makeBaseLayer(kind));
  } catch (e) {
    reportError("底圖切換失敗", e);
  }
  styleInk();
  if (S.ref) { buildTracks(); renderLegend(); }
  requestRender();
}

/** 依底圖深淺更新輔助線顏色。 */
function styleInk() {
  if (!S.dyn) return;
  const C = Cesium, ink = C.Color.fromCssColorString(inkHex()), light = lightBase();
  S.dyn.trail.material.uniforms.color = ink.withAlpha(light ? 0.8 : 0.95);
  S.dyn.drop.material.uniforms.color = ink.withAlpha(0.55);
  S.dyn.link.material.uniforms.color = ink.withAlpha(1.0);
}

/** Cesium ion 影像與地形的狀態合成一句(分開回報會互相蓋掉,看不出哪裡失敗、退回什麼)。 */
function ionStatus() {
  const { img, ter } = S.ion;
  const load = [];
  if (img === "loading") load.push("影像");
  if (ter === "loading") load.push("地形");
  const out = [];
  if (img === "fail" && ter === "fail") out.push("Cesium ion 無法使用(token 無效或無網路):改用 Esri 衛星圖與橢球地形");
  else if (img === "fail") out.push("Cesium ion 影像載入失敗(token 無效或無網路):改用 Esri 衛星圖");
  else if (ter === "fail") out.push("Cesium ion 地形載入失敗(token 無效或無網路):改用橢球地形");
  if (load.length) out.push(`Cesium ion ${load.join("與")}載入中…`);
  if (ter === "ok") out.push(S.scene ? `Cesium ion 地形:原點地面高 ${S.ground.toFixed(1)} m` : "Cesium ion 地形已啟用");
  return out.join(" · ");
}

function pushIonStatus() {
  const msg = ionStatus();
  if (msg === S.ionMsg) return;
  S.ionMsg = msg;
  report("status", msg);
}

async function applyTerrain() {
  if (!S.viewer) return;
  const C = Cesium;
  const token = S.opts.token || "";
  if (S.applied.token === token) return;
  S.applied.token = token;
  if (!token) {
    S.ion.ter = "";
    S.fitKey = null;
    S.viewer.terrainProvider = new C.EllipsoidTerrainProvider();
    setGround(0);
    pushIonStatus();
    return;
  }
  S.ion.ter = "loading";
  pushIonStatus();
  try {
    // 同影像:明確帶使用者的 token(createWorldTerrainAsync 只會用 Ion.defaultAccessToken)
    const tp = await C.CesiumTerrainProvider.fromUrl(
      C.IonResource.fromAssetId(1, { accessToken: token }),
      { requestVertexNormals: false, requestWaterMask: false, ellipsoid: C.Ellipsoid.WGS84 });
    if (S.applied.token !== token) return;
    S.viewer.terrainProvider = tp;
    S.ion.ter = "ok";
    await sampleGround();
  } catch (e) {
    if (S.applied.token !== token) return;        // 期間又換了 token:結果作廢
    reportError("Cesium ion 地形載入失敗(token 無效或無網路)", e);
    S.ion.ter = "fail";
    S.fitKey = null;
    S.viewer.terrainProvider = new C.EllipsoidTerrainProvider();
    setGround(0);
  }
  pushIonStatus();
}

/** 原點處的地形高(公尺,取不到為 NaN)。獨立成全域函式,測試可換成假的取樣。 */
async function groundAt(tp, lat, lng) {
  const [c] = await Cesium.sampleTerrainMostDetailed(tp, [Cesium.Cartographic.fromDegrees(lng, lat)]);
  return c && Number.isFinite(c.height) ? c.height : NaN;
}

async function sampleGround() {
  const C = Cesium;
  const tp = S.viewer && S.viewer.terrainProvider;
  if (!S.scene || !tp || tp instanceof C.EllipsoidTerrainProvider) { setGround(0); return; }
  const o = S.scene.origin;
  const key = S.sceneKey;
  try {
    const h = await groundAt(tp, o[0], o[1]);
    if (key !== S.sceneKey) return;
    setGround(Number.isFinite(h) ? h : 0);
    // 新 log 的第一次全覽是用舊的地面高算的(相機可能在地形底下):地面高到了再全覽一次;
    // 使用者已經動過相機、按過全覽或開了跟隨就不動
    if (S.fitKey === key) {
      S.fitKey = null;
      if (!S.opts.follow) viewAll(0);
    }
    pushIonStatus();
  } catch (e) {
    if (S.fitKey === key) S.fitKey = null;
    reportError("取樣原點地形高度失敗", e);
  }
}

function setGround(h) {
  if (Math.abs(h - S.ground) < 1e-3) return;
  S.ground = h;
  buildTracks();
  updateTime();
}

// ---------------------------------------------------------------- 場景資料

function nums(a) {
  return Float64Array.from(a || [], (v) => (v === null ? NaN : v));
}

function prepTrack(tr) {
  if (!tr || !tr.t || tr.t.length < 2) return null;
  return {
    name: tr.name || "", gap: tr.gap || 1.0,
    t: nums(tr.t), lat: nums(tr.lat), lng: nums(tr.lng), u: nums(tr.u),
    spd: tr.spd ? nums(tr.spd) : null, e3: tr.e3 ? nums(tr.e3) : null,
    pieces: tr.pieces || [], note: tr.note || "", aligned: !!tr.aligned,
  };
}

/** 第一個 > x 的索引 */
function upper(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] > x) hi = m; else lo = m + 1;
  }
  return lo;
}

/** 軌跡在相對時刻 x 的內插值;超出範圍或落在資料空洞內回傳 null(與 core.timeline.Track 相同規則)。 */
function sample(tr, x) {
  const t = tr.t, n = t.length;
  if (!n || x < t[0] - 1e-6 || x > t[n - 1] + 1e-6) return null;
  const j = upper(t, x);
  if (j <= 0) return pick(tr, 0, 0);
  if (j >= n) return pick(tr, n - 1, 0);
  const ta = t[j - 1], tb = t[j];
  if (tb - ta > tr.gap) {
    if (x - ta <= tr.gap / 2) return pick(tr, j - 1, 0);
    if (tb - x <= tr.gap / 2) return pick(tr, j, 0);
    return null;
  }
  return pick(tr, j - 1, tb > ta ? (x - ta) / (tb - ta) : 0);
}

function pick(tr, i, w) {
  const k = w > 0 ? i + 1 : i;
  const L = (a) => (w > 0 ? a[i] + (a[k] - a[i]) * w : a[i]);
  return {
    i: w > 0.5 ? k : i,
    lat: L(tr.lat), lng: L(tr.lng), u: L(tr.u),
    spd: tr.spd ? L(tr.spd) : NaN,
  };
}

function attAt(x) {
  const a = S.att;
  if (!a || !a.t.length) return null;
  const t = a.t, n = t.length;
  let j = upper(t, x);
  if (j <= 0) j = 1;
  if (j >= n) j = n - 1;
  if (n === 1) return { roll: a.roll[0], pitch: a.pitch[0], yaw: a.yaw[0] };
  const ta = t[j - 1], tb = t[j];
  let w = tb > ta ? (x - ta) / (tb - ta) : 0;
  if (tb - ta > a.gap) w = w < 0.5 ? 0 : 1;       // 空洞內不內插,取較近一端
  w = Math.min(1, Math.max(0, w));
  const L = (v) => v[j - 1] + (v[j] - v[j - 1]) * w;
  return { roll: L(a.roll), pitch: L(a.pitch), yaw: L(a.yaw) };
}

function errAt(x) {
  const e = S.err;
  if (!e) return { state: "none" };
  if (x < e.t0 - 1e-6 || x > e.t1 + 1e-6) return { state: "out" };
  const t = e.t, n = t.length;
  const j = Math.min(Math.max(upper(t, x), 1), n - 1);
  const ta = t[j - 1], tb = t[j];
  if (tb - ta > e.gap && x - ta > e.gap / 2 && tb - x > e.gap / 2) return { state: "gap" };
  const w = tb > ta ? Math.min(1, Math.max(0, (x - ta) / (tb - ta))) : 0;
  return { state: "ok", v: e.e3[j - 1] + (e.e3[j] - e.e3[j - 1]) * w };
}

function modeAt(x) {
  const m = S.modes;
  if (!m || !m.t.length) return null;
  const i = upper(m.t, x) - 1;
  return i >= 0 ? { name: m.name[i], color: m.color[i] } : null;
}

function armAt(x) {
  const segs = (S.scene && S.scene.arm) || [];
  for (const s of segs) if (x >= s[0] && x <= s[1]) return s[2];
  return null;
}

// ---------------------------------------------------------------- 著色

function hexColor(hex, alpha) {
  const c = Cesium.Color.fromCssColorString(hex);
  if (alpha !== undefined) c.alpha = alpha;
  return c;
}

function rampColor(f) {
  const ramp = S.scene.colors.ramp;
  const x = Math.min(1, Math.max(0, f)) * (ramp.length - 1);
  const i = Math.min(Math.floor(x), ramp.length - 2);
  const a = Cesium.Color.fromCssColorString(ramp[i]), b = Cesium.Color.fromCssColorString(ramp[i + 1]);
  return Cesium.Color.lerp(a, b, x - i, new Cesium.Color());
}

function colorMode() {
  const m = S.opts.color || "mode";
  if (m === "error" && !S.err) return "mode";
  return m;
}

function refColorFn(mode) {
  const c = S.scene.colors;
  if (mode === "error" && S.ref.e3) {
    const lo = S.err.vmin, hi = S.err.vmax, span = Math.max(hi - lo, 1e-9);
    const out = hexColor(OUT_COLOR, 0.9);
    return (i) => (Number.isFinite(S.ref.e3[i]) ? rampColor((S.ref.e3[i] - lo) / span) : out);
  }
  if (mode === "mode" && S.modes && S.modes.t.length) {
    const before = hexColor(c.muted);
    const cols = S.modes.color.map((h) => hexColor(h));
    return (i) => {
      const k = upper(S.modes.t, S.ref.t[i] + 1e-9) - 1;
      return k >= 0 ? cols[k] : before;
    };
  }
  const ref = hexColor(c.ref);
  return () => ref;
}

function estColorFn(mode) {
  const col = mode === "role" ? hexColor(S.scene.colors.est) : hexColor(inkHex(), lightBase() ? 0.7 : 0.6);
  return () => col;
}

// ---------------------------------------------------------------- 折線

function makeLine(tr, colorAt, width, perVertex, translucent, collect) {
  const C = Cesium;
  const inst = [];
  for (const piece of tr.pieces) {
    if (piece.length < 2) continue;
    const pos = new Array(piece.length), col = new Array(piece.length);
    for (let k = 0; k < piece.length; k++) {
      const i = piece[k];
      pos[k] = C.Cartesian3.fromDegrees(tr.lng[i], tr.lat[i], S.ground + tr.u[i]);
      col[k] = colorAt(i);
      if (collect) collect.push({ p: pos[k], i });
    }
    inst.push(new C.GeometryInstance({
      geometry: new C.PolylineGeometry({
        positions: pos, colors: col, colorsPerVertex: perVertex, width, arcType: C.ArcType.NONE,
      }),
    }));
  }
  if (!inst.length) return null;
  return S.viewer.scene.primitives.add(new C.Primitive({
    geometryInstances: inst,
    appearance: new C.PolylineColorAppearance({ translucent }),
    asynchronous: false, allowPicking: false,
  }));
}

function buildTracks() {
  if (!S.viewer) return;
  const prims = S.viewer.scene.primitives;
  for (const k of ["ref", "est"]) {
    if (S.prim[k]) { prims.remove(S.prim[k]); S.prim[k] = null; }
  }
  S.linePts = null;
  if (!S.ref) { requestRender(); return; }
  const mode = colorMode();
  try {
    if (S.est) {
      S.prim.est = makeLine(S.est, estColorFn(mode), mode === "role" ? 2.2 : 1.8, false, mode !== "role");
      if (S.prim.est) S.prim.est.show = !!S.opts.showEst;
    }
    const pts = [];
    S.prim.ref = makeLine(S.ref, refColorFn(mode), 3.5, mode === "error", false, pts);
    S.linePts = pts;
  } catch (e) {
    reportError("建立軌跡失敗", e);
  }
  requestRender();
}

function buildDynamic() {
  if (S.dyn || !S.viewer) return;
  const C = Cesium;
  const prims = S.viewer.scene.primitives;
  const points = prims.add(new C.PointPrimitiveCollection());
  const lines = prims.add(new C.PolylineCollection());
  const z = [new C.Cartesian3(0, 0, 0), new C.Cartesian3(1, 0, 0)];
  S.dyn = {
    points, lines,
    trail: lines.add({
      show: false, positions: z, width: 9,
      material: C.Material.fromType("PolylineGlow", { color: C.Color.WHITE.withAlpha(0.95), glowPower: 0.22, taperPower: 1.0 }),
    }),
    drop: lines.add({
      show: false, positions: z, width: 1.5,
      material: C.Material.fromType("PolylineDash", { color: C.Color.WHITE.withAlpha(0.55), dashLength: 10.0 }),
    }),
    link: lines.add({
      show: false, positions: z, width: 2.5,
      material: C.Material.fromType("Color", { color: C.Color.WHITE }),
    }),
    est: points.add({
      show: false, position: z[0], pixelSize: 11, color: C.Color.WHITE,
      outlineColor: C.Color.WHITE, outlineWidth: 2, disableDepthTestDistance: Number.POSITIVE_INFINITY,
    }),
  };
  styleInk();
}

async function ensureModel(m) {
  if (!m || !m.uri || !S.viewer) return;
  if (S.modelKey === m.kind && S.model) return;
  const prims = S.viewer.scene.primitives;
  if (S.model) { prims.remove(S.model); S.model = null; }
  S.modelKey = m.kind;
  S.modelReady = false;
  try {
    const model = await Cesium.Model.fromGltfAsync({
      url: m.uri, minimumPixelSize: 110, maximumScale: 20000, allowPicking: false, show: false,
      // 不用動態環境貼圖(會把天空 / 影像反射成斑駁色塊);只留漫射環境光,顏色乾淨好辨識
      environmentMapOptions: { enabled: false },
      imageBasedLighting: new Cesium.ImageBasedLighting({
        imageBasedLightingFactor: new Cesium.Cartesian2(1.0, 0.0),
        sphericalHarmonicCoefficients: [new Cesium.Cartesian3(1.25, 1.25, 1.3)].concat(
          Array.from({ length: 8 }, () => new Cesium.Cartesian3(0, 0, 0))),
      }),
    });
    if (S.modelKey !== m.kind || !S.viewer) { model.destroy(); return; }
    S.model = prims.add(model);
    model.readyEvent.addEventListener(() => { S.modelReady = true; requestRender(); });
    // 模型資源要經過幾個影格才載完;requestRenderMode 下主動多要幾次重繪
    let n = 0;
    const tick = setInterval(() => {
      requestRender();
      if (S.modelReady || ++n > 60) clearInterval(tick);
    }, 100);
    updateTime();
  } catch (e) {
    reportError("載具模型載入失敗", e);
  }
}

// ---------------------------------------------------------------- 播放頭

const scratch = {
  r: null, e: null, g: null, m: null, off: null, T: null, w: null,
};

function initScratch() {
  const C = Cesium;
  scratch.r = new C.Cartesian3(); scratch.e = new C.Cartesian3(); scratch.g = new C.Cartesian3();
  scratch.mp = new C.Cartesian3();
  scratch.off = new C.Cartesian3(); scratch.T = new C.Matrix4(); scratch.m = new C.Matrix4();
  scratch.w = new C.Cartesian2(); scratch.hpr = new C.HeadingPitchRoll();
}

function updateTime() {
  if (!S.viewer) return;
  const C = Cesium;
  if (!scratch.r) initScratch();
  const rel = (S.t === null || !S.scene) ? null : S.t - S.scene.t_base;
  const r = (rel === null || !S.ref) ? null : sample(S.ref, rel);
  // 估測點與白色誤差連線只在分析區間內(區間外的估測是用別段的對齊放的,連線長度不是 ATE 誤差)
  const inWin = rel !== null && errAt(rel).state === "ok";
  const showEstPt = !!(S.err && S.est && S.opts.showEst && inWin);
  const e = (rel === null || !showEstPt) ? null : sample(S.est, rel);
  const a = (rel === null || !r) ? null : attAt(rel);
  let rp = null, ep = null;
  if (r) rp = C.Cartesian3.fromDegrees(r.lng, r.lat, S.ground + r.u, C.Ellipsoid.WGS84, scratch.r);
  if (e) ep = C.Cartesian3.fromDegrees(e.lng, e.lat, S.ground + e.u, C.Ellipsoid.WGS84, scratch.e);

  if (S.model) {
    S.model.show = !!rp;
    if (rp) {
      const lift = (S.scene.model && S.scene.model.lift) || 0;
      const mp = lift ? C.Cartesian3.fromDegrees(r.lng, r.lat, S.ground + r.u + lift, C.Ellipsoid.WGS84, scratch.mp) : rp;
      const h = scratch.hpr;
      h.heading = C.Math.toRadians((a ? a.yaw : 0) - 90.0);
      h.pitch = C.Math.toRadians(a ? a.pitch : 0);
      h.roll = C.Math.toRadians(a ? a.roll : 0);
      C.Transforms.headingPitchRollToFixedFrame(mp, h, C.Ellipsoid.WGS84, undefined, scratch.m);
      C.Matrix4.clone(scratch.m, S.model.modelMatrix);
    }
  }

  const d = S.dyn;
  if (d) {
    d.est.show = !!ep;
    if (ep) {
      d.est.position = ep;
      d.est.color = hexColor(S.scene.colors.est);
    }
    d.link.show = !!(rp && ep);
    if (rp && ep) d.link.positions = [C.Cartesian3.clone(rp), C.Cartesian3.clone(ep)];
    d.drop.show = !!(rp && r.u > 0.8);
    if (d.drop.show) {
      const g = C.Cartesian3.fromDegrees(r.lng, r.lat, S.ground, C.Ellipsoid.WGS84, scratch.g);
      d.drop.positions = [C.Cartesian3.clone(rp), C.Cartesian3.clone(g)];
    }
    S.trailRaw = rp ? trailPositions(rel, rp) : null;
    d.trail.show = !!(S.trailRaw && S.trailRaw.length >= 2);
  }

  renderHud(rel, r, a);
  if (S.opts.follow && rp) follow(rp);
  requestRender();
}

function trailPositions(rel, rp) {
  const C = Cesium, tr = S.ref, t = tr.t;
  const b = upper(t, rel);
  let a = upper(t, rel - TRAIL_S);
  for (let k = b - 1; k > a; k--) {                // 只取最後一個資料空洞之後
    if (t[k] - t[k - 1] > tr.gap) { a = k; break; }
  }
  if (b > 0 && rel - t[b - 1] > tr.gap) return null;
  const out = [];
  const step = Math.max(1, Math.ceil((b - a) / 400));
  for (let i = a; i < b; i += step) {
    out.push(C.Cartesian3.fromDegrees(tr.lng[i], tr.lat[i], S.ground + tr.u[i]));
  }
  out.push(C.Cartesian3.clone(rp));
  return out;
}

// ---------------------------------------------------------------- 跟隨 / 全覽

function follow(pos) {
  const C = Cesium, cam = S.viewer.camera;
  const T = C.Transforms.eastNorthUpToFixedFrame(pos, C.Ellipsoid.WGS84, scratch.T);
  if (!S.following) {
    // 從遠處開始跟隨時先拉近到約 45 m;使用者本來就更近則保留
    const range = Math.min(Math.max(C.Cartesian3.distance(cam.positionWC, pos), 8), 45);
    const pitch = Math.min(Math.max(cam.pitch, C.Math.toRadians(-75)), C.Math.toRadians(-15));
    cam.lookAtTransform(T, new C.HeadingPitchRange(cam.heading, pitch, range));
    S.following = true;
  } else {
    // 相機在目前局部框(以飛機為原點的 ENU)內的位置 = 使用者的環繞角度與距離,原樣帶到新位置
    const off = C.Cartesian3.clone(cam.position, scratch.off);
    if (C.Cartesian3.magnitude(off) < 1e-6) off.z = 50;
    cam.lookAtTransform(T, off);
  }
}

function stopFollow() {
  if (!S.following || !S.viewer) return;
  S.viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
  S.following = false;
}

function viewAll(duration) {
  const C = Cesium;
  stopFollow();
  if (!S.viewer || !S.linePts || !S.linePts.length) return false;
  const cv = S.viewer.scene.canvas;
  if (!cv.clientWidth || !cv.clientHeight) {
    // 畫布還沒排版(分頁剛顯示、視窗剛開):視角算不出來,等有尺寸再全覽
    S.pendingViewAll = duration === undefined ? 0.8 : duration;
    return true;
  }
  S.pendingViewAll = null;
  const bs = C.BoundingSphere.fromPoints(S.linePts.map((o) => o.p));
  const r = Math.max(bs.radius, 25);
  // 依實際視角算距離:寬螢幕的垂直視角只有 30 幾度,固定倍數會把航跡切出畫面
  const fr = S.viewer.camera.frustum;
  const aspect = cv.clientWidth / cv.clientHeight;
  const fovx = fr.fov !== undefined ? (aspect >= 1 ? fr.fov : 2 * Math.atan(Math.tan(fr.fov / 2) * aspect))
                                    : C.Math.toRadians(60);
  const fovy = 2 * Math.atan(Math.tan(fovx / 2) / aspect);
  const half = Math.min(fovx, fovy) / 2;
  const range = (r / Math.sin(half)) * 1.08;
  S.viewer.camera.flyToBoundingSphere(new C.BoundingSphere(bs.center, r), {
    offset: new C.HeadingPitchRange(0.0, C.Math.toRadians(-45), range),
    duration: duration === undefined ? 0.8 : duration,
  });
  requestRender();
  return true;
}

// 等畫布排版完成再補做全覽(setScene 可能在分頁顯示前就收到)
function flushPendingViewAll() {
  if (S.pendingViewAll === null || S.pendingViewAll === undefined || !S.viewer) return;
  const cv = S.viewer.scene.canvas;
  if (cv.clientWidth && cv.clientHeight) viewAll(S.pendingViewAll);
}

// ---------------------------------------------------------------- 點擊跳轉

function pickSeek(pos) {
  if (!S.ref || !S.linePts || !S.linePts.length || !S.scene) return;
  const C = Cesium, sc = S.viewer.scene, w = scratch.w || new C.Cartesian2();
  const cands = [];
  let dmin = Infinity;
  for (const o of S.linePts) {
    const p = C.SceneTransforms.worldToWindowCoordinates(sc, o.p, w);
    if (!p) continue;
    const dd = Math.hypot(p.x - pos.x, p.y - pos.y);
    if (dd <= PICK_PX) {
      cands.push([dd, o.i]);
      if (dd < dmin) dmin = dd;
    }
  }
  if (!cands.length) return;
  // 軌跡重疊處(來回飛同一條路)取時間上最接近目前播放頭的那一次
  const cur = S.t === null ? null : S.t - S.scene.t_base;
  let best = null;
  for (const [dd, i] of cands) {
    if (dd > dmin + 4) continue;
    const score = cur === null ? dd : Math.abs(S.ref.t[i] - cur);
    if (!best || score < best[0]) best = [score, i];
  }
  // 在相鄰的原始樣本中再找螢幕上最近的一點
  const tr = S.ref, n = tr.t.length;
  const span = Math.max(2, Math.ceil(n / Math.max(1, S.linePts.length)) + 1);
  let bi = best[1], bd = Infinity;
  for (let i = Math.max(0, best[1] - span); i <= Math.min(n - 1, best[1] + span); i++) {
    const p = C.SceneTransforms.worldToWindowCoordinates(
      sc, C.Cartesian3.fromDegrees(tr.lng[i], tr.lat[i], S.ground + tr.u[i], C.Ellipsoid.WGS84, scratch.g), w);
    if (!p) continue;
    const dd = Math.hypot(p.x - pos.x, p.y - pos.y);
    if (dd < bd) { bd = dd; bi = i; }
  }
  report("seek", tr.t[bi] + S.scene.t_base);
}

// ---------------------------------------------------------------- HUD / 圖例

function fmtClock(sec) {
  if (!Number.isFinite(sec)) return "—";
  const sign = sec < 0 ? "-" : "";
  sec = Math.abs(sec);
  if (sec >= 3600) {
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = Math.floor(sec % 60);
    return `${sign}${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  let m = Math.floor(sec / 60);
  let s = (sec - m * 60).toFixed(1);
  if (s.startsWith("60")) { m += 1; s = "0.0"; }
  return `${sign}${m}:${s.padStart(4, "0")}`;
}

function fmtDeg(v) {
  return (v < 0 ? "-" : "") + Math.abs(v).toFixed(1) + "°";
}

function renderHud(rel, r, a) {
  const el = $("hud");
  const txt = $("hud-text");
  if (!S.scene || rel === null) {
    if (S.hudHtml) { el.classList.add("hidden"); S.hudHtml = ""; }
    return;
  }
  const arm = armAt(rel);
  const mode = modeAt(rel);
  let h = `<div class="time">${fmtClock(rel)}<span class="rel">(${rel.toFixed(1)} s)</span></div>`;
  h += `<div class="state">`;
  h += arm ? `<span class="tag armed">${esc(arm)}</span>` : `<span class="tag">地面(未 ARM)</span>`;
  if (mode) h += `<span class="chip" style="background:${mode.color}"></span><span class="mode-name">${esc(mode.name)}</span>`;
  h += `</div><div class="grid">`;
  if (r) {
    h += `<span>高度</span><b>${r.u.toFixed(1)} m</b>`;
    h += `<span>水平速度</span><b>${Number.isFinite(r.spd) ? r.spd.toFixed(1) + " m/s" : "—"}</b>`;
  } else {
    h += `<span>參考 ${esc(S.ref ? S.ref.name : "")}</span><b class="dim">此刻無資料</b>`;
  }
  const er = errAt(rel);
  if (er.state === "ok") h += `<span>ATE 3D 誤差</span><b>${er.v.toFixed(2)} m</b>`;
  else h += `<span>ATE 3D 誤差</span><b class="dim">${{ none: "—(尚未計算)", out: "—(區間外)", gap: "—(無資料)" }[er.state]}</b>`;
  if (a && r) {
    const yaw = ((Math.round(a.yaw) % 360) + 360) % 360;      // 先取整再折回:不會顯示 360°
    if (S.scene.attitude === "derived") {
      h += `<span>航向</span><b>${yaw.toFixed(0)}°</b>`;
    } else {
      h += `<span>姿態</span><b>R ${fmtDeg(a.roll)} · P ${fmtDeg(a.pitch)} · Y ${yaw.toFixed(0)}°</b>`;
    }
  }
  h += `</div>`;
  if (S.scene.attitude === "derived") h += `<div class="foot">航向由航跡推估(log 無 ATT 姿態)</div>`;
  if (h !== S.hudHtml) {
    txt.innerHTML = h;
    S.hudHtml = h;
  }
  // 姿態儀:只有 log 真的有 ATT 才畫(推估姿態的 roll / pitch 恆為 0,畫了反而誤導)
  const cv = $("hud-att");
  const showAtt = !!(a && r && S.scene.attitude !== "derived");
  cv.classList.toggle("hidden", !showAtt);
  if (showAtt) drawAttitude(cv, a.roll, a.pitch);
  el.classList.remove("hidden");
}

// 人工地平儀:roll 右傾為正、pitch 抬頭為正(地平線隨抬頭下移、隨右傾逆時針轉)
function drawAttitude(cv, roll, pitch) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = 140;
  if (cv.width !== Math.round(cssW * dpr)) { cv.width = cv.height = Math.round(cssW * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cssW, cssW);
  const c = cssW / 2, R = c - 4, k = R / 25;           // 圓內可見 ±25° 俯仰
  const rad = Math.PI / 180;
  g.save();
  g.beginPath(); g.arc(c, c, R, 0, 2 * Math.PI); g.clip();
  g.translate(c, c); g.rotate(-roll * rad);
  const off = pitch * k;
  g.fillStyle = "#2f6fb5"; g.fillRect(-3 * R, -3 * R + off, 6 * R, 3 * R);
  g.fillStyle = "#8a5a2b"; g.fillRect(-3 * R, off, 6 * R, 3 * R);
  g.strokeStyle = "#fff"; g.fillStyle = "#fff"; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(-3 * R, off); g.lineTo(3 * R, off); g.stroke();
  g.font = "9px sans-serif"; g.textBaseline = "middle";
  for (let d = -20; d <= 20; d += 5) {
    if (d === 0) continue;
    const y = off - d * k, w = R * (d % 10 === 0 ? 0.34 : 0.18);
    g.beginPath(); g.moveTo(-w, y); g.lineTo(w, y); g.stroke();
    if (Math.abs(d) === 10) { g.textAlign = "left"; g.fillText(String(Math.abs(d)), w + 3, y); }
  }
  g.restore();
  g.save();
  g.translate(c, c);
  g.lineWidth = 2.5; g.strokeStyle = "#3a3a37"; g.beginPath(); g.arc(0, 0, R, 0, 2 * Math.PI); g.stroke();
  g.lineWidth = 1.5; g.strokeStyle = "#fff";
  for (const d of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) {      // 滾轉刻度(隨機身固定)
    const len = R * (d % 30 === 0 ? 0.10 : 0.06), s = Math.sin(d * rad), co = Math.cos(d * rad);
    g.beginPath(); g.moveTo(s * (R - len), -co * (R - len)); g.lineTo(s * R, -co * R); g.stroke();
  }
  g.rotate(-roll * rad);                                                    // 滾轉指標:隨地平線轉
  g.fillStyle = "#fab219";
  g.beginPath(); g.moveTo(0, -R + 1); g.lineTo(-R * 0.06, -R + R * 0.14); g.lineTo(R * 0.06, -R + R * 0.14); g.fill();
  g.rotate(roll * rad);
  g.strokeStyle = "#fab219"; g.lineWidth = 3;                              // 機身符號(固定)
  g.beginPath();
  g.moveTo(-R * 0.45, 0); g.lineTo(-R * 0.15, 0); g.lineTo(-R * 0.15, R * 0.08);
  g.moveTo(R * 0.45, 0); g.lineTo(R * 0.15, 0); g.lineTo(R * 0.15, R * 0.08);
  g.stroke();
  g.beginPath(); g.arc(0, 0, 2.5, 0, 2 * Math.PI); g.fill();
  g.restore();
}

function renderLegend() {
  const el = $("legend");
  if (!S.scene || !S.ref) { el.classList.add("hidden"); return; }
  const c = S.scene.colors, mode = colorMode();
  let h = "";
  const ml = S.scene.mode_legend || [];
  if (mode === "mode") {
    h += `<h4>飛行模式</h4>`;
    if (ml.length) {
      for (const [name, col] of ml) h += `<div class="row"><span class="chip" style="background:${col}"></span>${esc(name)}</div>`;
    } else {
      h += `<div class="note">log 沒有飛行模式紀錄,參考軌跡以藍色顯示</div>`;
    }
  } else if (mode === "error") {
    const grad = c.ramp.join(",");
    h += `<h4>ATE 3D 誤差</h4><div class="ramp" style="background:linear-gradient(90deg,${grad})"></div>`;
    h += `<div class="ramp-lab"><span>${S.err.vmin.toFixed(2)} m</span><span>${S.err.vmax.toFixed(2)} m</span></div>`;
    h += `<div class="row"><span class="line" style="border-color:${OUT_COLOR}"></span>分析區間外</div>`;
  }
  if (h) h += `<div class="sep"></div>`;
  const refTxt = mode === "mode" && ml.length ? "(依模式著色)" : mode === "error" ? "(依誤差著色)" : "";
  let sample = `<span class="line" style="border-color:${c.ref}"></span>`;
  if (mode === "mode" && ml.length) {
    const cols = ml.map((m) => m[1]);
    const stops = cols.map((col, k) => `${col} ${(k * 100) / cols.length}% ${((k + 1) * 100) / cols.length}%`);
    sample = `<span class="line grad" style="background:linear-gradient(90deg,${stops.join(",")})"></span>`;
  } else if (mode === "error") {
    sample = `<span class="line grad" style="background:linear-gradient(90deg,${c.ramp.join(",")})"></span>`;
  }
  h += `<div class="row">${sample}參考 ${esc(S.ref.name)}${refTxt}</div>`;
  if (S.est) {
    const estCol = mode === "role" ? c.est : (lightBase() ? "rgba(38,38,36,0.7)" : "rgba(255,255,255,0.6)");
    const hid = S.opts.showEst ? "" : "(已隱藏)";
    const estTxt = mode === "role" ? "" : (lightBase() ? "(深灰細線)" : "(白色細線)");
    const dark = mode !== "role" && lightBase() ? " dark" : "";
    h += `<div class="row"><span class="line thin${dark}" style="border-color:${estCol}"></span>估測 ${esc(S.est.name)}${estTxt}${hid}</div>`;
    if (S.err && S.opts.showEst) {
      h += `<div class="row"><span class="dot" style="background:${c.est}"></span>估測此刻位置 · ${lightBase() ? "黑線" : "白線"} = 誤差(分析區間內)</div>`;
    }
    if (S.est.note) h += `<div class="note" style="margin-top:4px">${esc(S.est.note)}</div>`;
  }
  el.innerHTML = h;
  el.classList.remove("hidden");
}

// ---------------------------------------------------------------- 對外介面

function clearScene() {
  if (!S.viewer) return;
  const prims = S.viewer.scene.primitives;
  for (const k of ["ref", "est"]) {
    if (S.prim[k]) { prims.remove(S.prim[k]); S.prim[k] = null; }
  }
  S.scene = null; S.ref = null; S.est = null; S.att = null; S.err = null; S.modes = null;
  S.linePts = null;
  if (S.model) S.model.show = false;
  if (S.dyn) for (const k of ["trail", "drop", "link", "est"]) S.dyn[k].show = false;
  renderHud(null);
  renderLegend();
}

const NO_ORIGIN_MSG = "這份 log 沒有地理原點(ORGN)或經緯度來源(GPS 等),軌跡放不到地球上。\n" +
  "局部 NED 軌跡(VIO、EKF)請看「平面軌跡」或「3D 軌跡」分頁。";

const app = {
  /** reason(scene 為 null 時):'noorigin' = log 已載入但沒有地理原點。 */
  setScene(scene, reason) {
    if (!S.viewer) return 0;
    try {
      const key = scene ? [scene.origin.join(","), scene.t_base, scene.ref && scene.ref.name].join("|") : "";
      const sameLog = key && key === S.sceneKey;
      clearScene();
      S.sceneKey = key;
      if (!scene) {
        S.fitKey = null;
        stopFollow();
        $("hint").classList.add("hidden");
        if (reason === "noorigin") {
          overlay("empty", "無法放上地圖", NO_ORIGIN_MSG);
          // 別停在上一份 log 的場地(看起來像這份 log 在那裡):退回地球全景
          S.pendingViewAll = null;
          S.viewer.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(121.0, 23.7, 9.0e6) });
        } else overlay("empty", "尚未有軌跡", "載入 .bin 或模擬資料後,參考軌跡會畫在這裡");
        pushIonStatus();
        requestRender();
        return 1;
      }
      S.scene = scene;
      S.ref = prepTrack(scene.ref);
      S.est = prepTrack(scene.est);
      if (scene.att) {
        S.att = { t: nums(scene.att.t), roll: nums(scene.att.roll), pitch: nums(scene.att.pitch),
                  yaw: nums(scene.att.yaw), gap: scene.att.gap || 1.0 };
      }
      if (scene.err) {
        S.err = { t: nums(scene.err.t), e3: nums(scene.err.e3), t0: scene.err.t0, t1: scene.err.t1,
                  gap: scene.err.gap || 1.0, vmin: scene.err.vmin, vmax: scene.err.vmax };
      }
      S.modes = { t: nums(scene.modes.map((m) => m[0])), name: scene.modes.map((m) => m[1]),
                  color: scene.modes.map((m) => m[2]) };
      overlay(null);
      $("hint").classList.remove("hidden");
      if (!S.opts.token) S.ground = 0;
      buildDynamic();
      buildTracks();
      renderLegend();
      ensureModel(scene.model);
      if (!sameLog) {
        viewAll(0);
        // 用 ion 地形時地面高是非同步取樣的:取到後再全覽一次(見 sampleGround)
        S.fitKey = S.opts.token && !S.opts.follow ? key : null;
      }
      if (S.opts.token) sampleGround();
      updateTime();
      return 1;
    } catch (e) {
      reportError("setScene", e);
      return 0;
    }
  },

  setTime(t) {
    S.t = (t === null || t === undefined || !Number.isFinite(+t)) ? null : +t;
    if (!S.viewer) return 0;
    try {
      updateTime();
    } catch (e) {
      reportError("setTime", e);
    }
    return 1;
  },

  setOptions(o) {
    const prev = S.opts;
    S.opts = Object.assign({}, S.opts, o || {});
    if (prev.base !== S.opts.base || prev.token !== S.opts.token) S.ion.img = "";   // 使用者重選:清掉上次的失敗
    if (S.opts.follow) S.fitKey = null;
    if (!S.viewer) return 0;
    try {
      applyBase();
      applyTerrain();
      if (prev.color !== S.opts.color) {
        buildTracks();
        renderLegend();
      }
      if (prev.showEst !== S.opts.showEst) {
        if (S.prim.est) S.prim.est.show = !!S.opts.showEst;
        renderLegend();
      }
      if (!S.opts.follow) stopFollow();
      updateTime();
      pushIonStatus();
    } catch (e) {
      reportError("setOptions", e);
    }
    return 1;
  },

  viewAll(duration) {
    S.fitKey = null;                 // 使用者按了全覽:之後的地面高不再自動改視角
    try {
      return viewAll(duration) ? 1 : 0;
    } catch (e) {
      reportError("viewAll", e);
      return 0;
    }
  },

  /** 狀態快照(測試輪詢用)。 */
  status() {
    const v = S.viewer;
    return {
      ready: S.ready, available: S.available, reason: S.reason, hasScene: !!S.scene,
      tilesLoaded: v ? v.scene.globe.tilesLoaded : false,
      modelReady: !!(S.model && S.modelReady), following: S.following,
      ground: S.ground, base: S.applied.base, t: S.t, ion: Object.assign({}, S.ion), fitKey: S.fitKey,
      linePts: S.linePts ? S.linePts.length : 0, errors: S.errors.slice(-5),
      overlay: $("overlay").classList.contains("hidden") ? "" : $("ov-title").textContent,
    };
  },

  /** 把世界座標點投影到視窗像素(測試用:模擬點擊軌跡)。 */
  project(lat, lng, u) {
    const p = Cesium.SceneTransforms.worldToWindowCoordinates(
      S.viewer.scene, Cesium.Cartesian3.fromDegrees(lng, lat, S.ground + u));
    return p ? [p.x, p.y] : null;
  },

  /** 測試用:模擬在視窗座標 (x, y) 點擊。 */
  click(x, y) {
    pickSeek({ x, y });
    return 1;
  },
};
window.app = app;

boot();
