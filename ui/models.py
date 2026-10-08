"""3D 地圖用的載具模型:以 numpy / struct 程序化產生 glTF 2.0 二進位(.glb)。

幾何先在機體座標(x 前、y 左、z 上,公尺)建好,寫檔時換成 glTF 慣例
(+Y 上、+Z 前、+X 左)。Cesium 載入 glTF 2.0 時會把 +Z 前 / +Y 上轉成模型座標
+X 前 / +Z 上,所以 heading = yaw − 90°(yaw 為羅盤航向)時機頭就指向航向。

辨識方向的記號:
- 四旋翼:前方兩支臂與螺旋槳盤為紅色、機頭有紅色鼻錐;後方為淺灰。
- 定翼機:機頭紅色;左翼尖紅燈、右翼尖綠燈(航空慣例)。
- 車輛:車頭紅色擋板。
"""
from __future__ import annotations

import base64
import json
import struct
from functools import lru_cache

import numpy as np

# (R, G, B, A) 0..1
_DARK = (0.16, 0.17, 0.19, 1.0)
_BODY = (0.62, 0.65, 0.69, 1.0)
_CANOPY = (0.90, 0.92, 0.94, 1.0)
_LIGHT = (0.86, 0.88, 0.90, 1.0)
_WHITE = (0.95, 0.96, 0.97, 1.0)
_RED = (0.91, 0.24, 0.20, 1.0)
_GREEN = (0.15, 0.80, 0.35, 1.0)
_MOTOR = (0.10, 0.10, 0.11, 1.0)
_DISC_FRONT = (0.95, 0.30, 0.26, 0.22)
_DISC_REAR = (0.92, 0.93, 0.95, 0.16)


# ---------------- 網格積木(機體座標 x 前 / y 左 / z 上) ----------------

class _Mesh:
    """依材質分組累積三角形;法向量一律由頂點繞序算出(雙面材質下光照仍正確)。"""

    def __init__(self):
        self.groups: dict[tuple, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    def add(self, color, verts, tris, smooth: bool = False):
        v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if smooth:
            n = np.zeros_like(v)
            fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            for k in range(3):
                np.add.at(n, f[:, k], fn)
            pos, idx = v, f
        else:
            pos = v[f.reshape(-1)]
            fn = np.cross(pos[1::3] - pos[0::3], pos[2::3] - pos[0::3])
            n = np.repeat(fn, 3, axis=0)
            idx = np.arange(pos.shape[0]).reshape(-1, 3)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), np.array([0.0, 0.0, 1.0]))
        self.groups.setdefault(tuple(color), []).append((pos, n, idx))

    # --- 基本形狀 ---

    def box(self, color, center, size, yaw_deg: float = 0.0, pitch_deg: float = 0.0):
        """長方體;先繞 y 轉 pitch 再繞 z 轉 yaw(度)。"""
        hx, hy, hz = (np.asarray(size, dtype=float) / 2.0)
        c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        c = c @ _rot(yaw_deg, pitch_deg).T + np.asarray(center, dtype=float)
        # 角點索引:i = 4*ix + 2*iy + iz
        quads = [(0, 1, 3, 2), (4, 6, 7, 5),      # -x / +x
                 (0, 4, 5, 1), (2, 3, 7, 6),      # -y / +y
                 (0, 2, 6, 4), (1, 5, 7, 3)]      # -z / +z
        tris = [t for a, b, cc, d in quads for t in ((a, b, cc), (a, cc, d))]
        self.add(color, c, tris)

    def cylinder(self, color, center, radius: float, height: float, n: int = 24,
                 axis: str = "z"):
        """圓柱(側面平滑、上下蓋平面);axis 為圓柱軸方向 'x' / 'y' / 'z'。"""
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        ring = np.column_stack([radius * np.cos(a), radius * np.sin(a)])
        h2 = height / 2.0
        bot = np.column_stack([ring, np.full(n, -h2)])
        top = np.column_stack([ring, np.full(n, h2)])
        side_v = np.vstack([bot, top])
        side_t = []
        for i in range(n):
            j = (i + 1) % n
            side_t += [(i, j, n + j), (i, n + j, n + i)]
        cap_v = np.vstack([bot, top, [[0, 0, -h2], [0, 0, h2]]])
        cb, ct = 2 * n, 2 * n + 1
        cap_t = []
        for i in range(n):
            j = (i + 1) % n
            cap_t += [(cb, j, i), (ct, n + i, n + j)]
        m = {"z": np.eye(3),                                   # 循環置換(det = +1,繞序不變)
             "x": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
             "y": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)}[axis]
        off = np.asarray(center, dtype=float)
        self.add(color, side_v @ m.T + off, side_t, smooth=True)
        self.add(color, cap_v @ m.T + off, cap_t)

    def lathe(self, color, profile, n: int = 24, center=(0.0, 0.0, 0.0), z_scale: float = 1.0):
        """沿 x 軸的迴轉體;profile = [(x, r), ...](x 遞增),r = 0 的端點收成尖頂。"""
        prof = [(float(x), float(r)) for x, r in profile]
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        verts, rings = [], []
        for x, r in prof:
            if r <= 1e-9:
                rings.append([len(verts)] * n)
                verts.append((x, 0.0, 0.0))
            else:
                base = len(verts)
                rings.append(list(range(base, base + n)))
                verts += [(x, r * np.cos(t), r * np.sin(t) * z_scale) for t in a]
        tris = []
        for k in range(len(rings) - 1):
            r0, r1 = rings[k], rings[k + 1]
            for i in range(n):
                j = (i + 1) % n
                a, b, c, d = r0[i], r0[j], r1[j], r1[i]   # 四邊形(外側看為逆時針)
                if a != b:
                    tris.append((a, b, c))
                if c != d:
                    tris.append((a, c, d))
        # 開口端補平面蓋(起點朝 -x、終點朝 +x)
        for k in (0, len(rings) - 1):
            ring = rings[k]
            if ring[0] == ring[1]:
                continue
            c = len(verts)
            verts.append((prof[k][0], 0.0, 0.0))
            for i in range(n):
                j = (i + 1) % n
                tris.append((c, ring[j], ring[i]) if k == 0 else (c, ring[i], ring[j]))
        v = np.asarray(verts, dtype=float) + np.asarray(center, dtype=float)
        self.add(color, v, tris, smooth=True)

    def panel(self, color, outline, thickness: float, z: float = 0.0, dihedral_deg: float = 0.0,
              vertical: bool = False):
        """薄板(機翼 / 尾翼):outline 為凸多邊形 [(x, s), ...];
        水平板 s = y(左),垂直板 s = z(上)。dihedral 讓板沿 x 軸上反(左右翼各給正負號)。"""
        pts = np.asarray(outline, dtype=float)
        k = len(pts)
        h = thickness / 2.0
        if vertical:
            lo = np.column_stack([pts[:, 0], np.full(k, -h), pts[:, 1]])
            hi = np.column_stack([pts[:, 0], np.full(k, h), pts[:, 1]])
        else:
            lo = np.column_stack([pts[:, 0], pts[:, 1], np.full(k, -h)])
            hi = np.column_stack([pts[:, 0], pts[:, 1], np.full(k, h)])
            if dihedral_deg:
                s = np.tan(np.radians(dihedral_deg))
                lo[:, 2] += np.abs(pts[:, 1]) * s
                hi[:, 2] += np.abs(pts[:, 1]) * s
            lo[:, 2] += z
            hi[:, 2] += z
        v = np.vstack([lo, hi])
        tris = []
        for i in range(1, k - 1):
            tris += [(0, i + 1, i), (k, k + i, k + i + 1)]
        for i in range(k):
            j = (i + 1) % k
            tris += [(i, j, k + j), (i, k + j, k + i)]
        self.add(color, v, tris)


def _rot(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    cy, sy = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rz @ ry


# ---------------- 載具 ----------------

def _quad() -> _Mesh:
    """X 型四旋翼,軸距約 0.64 m(對角馬達間距)。"""
    m = _Mesh()
    m.box(_BODY, (0.0, 0.0, 0.0), (0.26, 0.16, 0.07))
    m.box(_CANOPY, (-0.01, 0.0, 0.05), (0.18, 0.12, 0.035))
    m.box(_DARK, (-0.11, 0.0, 0.02), (0.05, 0.10, 0.05))          # 電池 / 尾端
    m.lathe(_RED, [(0.131, 0.04), (0.19, 0.03), (0.26, 0.0)], n=16, z_scale=0.7)  # 鼻錐(前)
    r_motor = 0.32
    for ang, front in ((45, True), (-45, True), (135, False), (-135, False)):
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        mid = 0.5 * (0.07 + r_motor)
        m.box(_RED if front else _LIGHT, (mid * c, mid * s, 0.005),
              (r_motor - 0.07, 0.034, 0.03), yaw_deg=ang)
        mx, my = r_motor * c, r_motor * s
        m.cylinder(_MOTOR, (mx, my, 0.035), 0.032, 0.05, n=16)
        # 螺旋槳:兩葉實心槳(前紅後白)+ 淡色旋轉盤
        m.box(_RED if front else _WHITE, (mx, my, 0.066), (0.30, 0.028, 0.008), yaw_deg=ang + 60)
        m.cylinder(_DISC_FRONT if front else _DISC_REAR, (mx, my, 0.068), 0.16, 0.004, n=32)
    # 起落架:兩根腳 + 滑橇(看得出姿態的底部參考)
    for sy in (1, -1):
        y = 0.085 * sy
        for x in (0.07, -0.07):
            m.box(_DARK, (x, y, -0.075), (0.015, 0.015, 0.09))
        m.box(_DARK, (0.0, y, -0.12), (0.26, 0.02, 0.015))
    return m


def _plane() -> _Mesh:
    """定翼機,翼展約 2 m、機身長約 1.5 m。"""
    m = _Mesh()
    m.lathe(_LIGHT, [(-0.80, 0.0), (-0.78, 0.025), (-0.40, 0.055), (0.0, 0.085),
                     (0.35, 0.085), (0.55, 0.065), (0.66, 0.04)], n=24)
    m.lathe(_RED, [(0.66, 0.04), (0.70, 0.032), (0.76, 0.0)], n=24)          # 機頭
    m.cylinder(_DISC_REAR, (0.765, 0.0, 0.0), 0.16, 0.005, n=32, axis="x")   # 螺旋槳盤
    m.box(_CANOPY, (0.25, 0.0, 0.075), (0.30, 0.09, 0.05))                     # 座艙
    # 主翼(上反 3°)
    for side in (1, -1):
        m.panel(_WHITE, [(0.16, 0.0), (0.16, 1.0 * side), (-0.03, 1.0 * side), (-0.10, 0.0)][::side],
                0.025, z=0.04, dihedral_deg=3.0)
        tip_z = 0.04 + np.tan(np.radians(3.0)) * 1.0
        m.box(_RED if side > 0 else _GREEN, (0.07, 1.0 * side, tip_z), (0.20, 0.03, 0.04))
    # 水平尾翼
    m.panel(_WHITE, [(-0.58, 0.0), (-0.62, 0.32), (-0.74, 0.32), (-0.78, 0.0),
                     (-0.74, -0.32), (-0.62, -0.32)][::-1], 0.016, z=0.02)
    # 垂直尾翼
    m.panel(_RED, [(-0.56, 0.03), (-0.70, 0.30), (-0.80, 0.30), (-0.80, 0.03)][::-1],
            0.016, vertical=True)
    return m


def _rover() -> _Mesh:
    """地面車輛(車長約 0.9 m)。"""
    m = _Mesh()
    m.box(_BODY, (0.0, 0.0, 0.14), (0.80, 0.42, 0.12))
    m.box(_CANOPY, (-0.05, 0.0, 0.24), (0.40, 0.32, 0.08))
    m.box(_RED, (0.41, 0.0, 0.14), (0.04, 0.44, 0.10))                         # 車頭
    for x in (0.27, -0.27):
        for y in (0.25, -0.25):
            m.cylinder(_MOTOR, (x, y, 0.09), 0.09, 0.08, n=20, axis="y")
    return m


# ---------------- glTF 2.0 二進位 ----------------

def _to_gltf_axes(p: np.ndarray) -> np.ndarray:
    """機體 (x 前, y 左, z 上) → glTF (+X 左, +Y 上, +Z 前)。循環置換,繞序不變。"""
    return np.column_stack([p[:, 1], p[:, 2], p[:, 0]])


def _pad4(b: bytes, fill: bytes) -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def mesh_to_glb(mesh: _Mesh, name: str = "vehicle") -> bytes:
    """把 _Mesh 寫成單一 node / 單一 mesh(每種材質一個 primitive)的 .glb。"""
    blobs: list[bytes] = []
    views, accessors, prims, materials = [], [], [], []
    offset = 0

    def push(data: bytes, target: int) -> int:
        nonlocal offset
        data = _pad4(data, b"\x00")
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        blobs.append(data)
        offset += len(data)
        return len(views) - 1

    for color, parts in mesh.groups.items():
        pos = np.vstack([_to_gltf_axes(p) for p, _, _ in parts]).astype(np.float32)
        nrm = np.vstack([_to_gltf_axes(n) for _, n, _ in parts]).astype(np.float32)
        base, idx_list = 0, []
        for p, _, idx in parts:
            idx_list.append(idx.reshape(-1) + base)
            base += p.shape[0]
        idx = np.concatenate(idx_list)
        itype, ictype = (np.uint16, 5123) if pos.shape[0] < 65535 else (np.uint32, 5125)
        vp = push(pos.tobytes(), 34962)
        vn = push(nrm.tobytes(), 34962)
        vi = push(idx.astype(itype).tobytes(), 34963)
        accessors.append({"bufferView": vp, "componentType": 5126, "count": int(pos.shape[0]),
                          "type": "VEC3", "min": pos.min(axis=0).tolist(),
                          "max": pos.max(axis=0).tolist()})
        accessors.append({"bufferView": vn, "componentType": 5126, "count": int(nrm.shape[0]),
                          "type": "VEC3"})
        accessors.append({"bufferView": vi, "componentType": ictype, "count": int(idx.size),
                          "type": "SCALAR"})
        mat = {"pbrMetallicRoughness": {"baseColorFactor": [float(c) for c in color],
                                        "metallicFactor": 0.0, "roughnessFactor": 0.65},
               "doubleSided": True}
        if color[3] < 1.0:
            mat["alphaMode"] = "BLEND"
        materials.append(mat)
        a = len(accessors)
        prims.append({"attributes": {"POSITION": a - 3, "NORMAL": a - 2}, "indices": a - 1,
                      "material": len(materials) - 1, "mode": 4})

    bin_chunk = b"".join(blobs)
    gltf = {
        "asset": {"version": "2.0", "generator": "vio_ate_studio ui/models.py"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": prims}],
        "materials": materials,
        "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": len(bin_chunk)}],
    }
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(js) + 8 + len(bin_chunk)
    return b"".join([
        struct.pack("<4sII", b"glTF", 2, total),
        struct.pack("<I4s", len(js), b"JSON"), js,
        struct.pack("<I4s", len(bin_chunk), b"BIN\x00"), bin_chunk,
    ])


# ---------------- 對外介面 ----------------

MODEL_KINDS = ("quad", "plane", "rover")
# 模型原點(機體中心)到最低點的距離:貼地時把模型抬高這麼多,起落架才不會陷進地面
MODEL_LIFT = {"quad": 0.13, "plane": 0.16, "rover": 0.0}


def model_kind(vehicle: str | None) -> str:
    """LogIndex.vehicle → 模型種類:plane → 定翼機;rover / 船 → 車輛;其餘(含直升機)→ 四旋翼。"""
    v = (vehicle or "").lower()
    if v == "plane":
        return "plane"
    if v in ("rover", "boat"):
        return "rover"
    return "quad"


@lru_cache(maxsize=None)
def model_glb(kind: str) -> bytes:
    builders = {"quad": _quad, "plane": _plane, "rover": _rover}
    if kind not in builders:
        raise ValueError(f"未知模型:{kind}")
    return mesh_to_glb(builders[kind](), name=kind)


@lru_cache(maxsize=None)
def model_data_uri(kind: str) -> str:
    """data: URI(Cesium.Model.fromGltfAsync 可直接載入)。"""
    return "data:model/gltf-binary;base64," + base64.b64encode(model_glb(kind)).decode("ascii")


def parse_glb(data: bytes) -> tuple[dict, bytes]:
    """解析 .glb 為 (JSON, BIN);供測試驗證結構。"""
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or total != len(data):
        raise ValueError("不是有效的 glTF 2.0 二進位檔")
    jlen, jtype = struct.unpack_from("<I4s", data, 12)
    if jtype != b"JSON":
        raise ValueError("第一個 chunk 應為 JSON")
    gltf = json.loads(data[20:20 + jlen].decode("utf-8"))
    blen, btype = struct.unpack_from("<I4s", data, 20 + jlen)
    if btype != b"BIN\x00":
        raise ValueError("第二個 chunk 應為 BIN")
    return gltf, data[28 + jlen:28 + jlen + blen]
