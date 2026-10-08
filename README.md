# VIO / GPS ATE Studio — 飛行 log 檢視與絕對軌跡誤差分析

讀 **ArduPilot `.bin` 記錄檔**的桌面程式(PyQt6),操作方式仿
[plot.ardupilot.org(UAV Log Viewer)](https://plot.ardupilot.org/):
**3D 地球 + 依飛行模式著色的航跡 + 帶姿態的 3D 機體**、**任意欄位繪圖**、
參數 / 訊息面板,全部跟著同一條**可播放的時間軸**。

在這之上做 **ATE(Absolute Trajectory Error,絕對軌跡誤差)**:兩條位置軌跡時間同步、
空間對齊後計算誤差。主要用途是評估從 **ROS2 以 `VISION_POSITION_ESTIMATE` 餵進飛控的
VIO 定位**(log 中的 `VISP`)相對 GPS 的偏差 —— 偏多少、往哪偏、何時開始漂、在場地哪裡漂。

---

## 安裝與啟動

```powershell
cd C:\Users\joy46\Documents\Projects\vio_ate_studio
pip install -r requirements.txt
python main.py
```

需求:Python ≥ 3.10、PyQt6、**PyQt6-WebEngine**(3D 地圖)、numpy、matplotlib、pymavlink。

**3D 地圖需要網路**:CesiumJS 從 `cdn.jsdelivr.net` 載入,衛星圖來自 Esri
(`server.arcgisonline.com`),街道圖來自 OpenStreetMap。看過的圖磚會快取在本機。
沒網路時地圖分頁會顯示說明,其他功能照常可用;要完全離線可下載 Cesium 的
`Build/Cesium/` 資料夾,並把環境變數 `VIO_ATE_CESIUM_BASE` 指到它。

同時開兩個視窗時,第二個會改用另一份地圖快取(第一份被鎖住)。

不需要 Cesium ion 帳號。若有免費的 [Cesium ion](https://cesium.com/ion/) token,
在地圖上方按「Cesium ion token…」貼上,就能用 Cesium World Terrain 地形與 ion 衛星圖
(航跡高度改為「原點地面高度 + 相對高度」)。

沒有 log 也能先試跑 —— 按 **「模擬資料」** 產生一段 30 m 方形任務
(GPS 5 Hz / VISP 30 Hz / POS 10 Hz,另含 ATT、BAT、VIBE、RCIN/RCOU、PARM、MSG… 等欄位),
VIO 端注入了已知誤差:偏航 1.5°、尺度 1.004、隨機漫步漂移約 0.9 m。

「檔案」選單另有 **「載入模擬資料(開機到關機一份 log、兩次 ARM)」**:
地面待機 → 第一次飛行 → 上鎖、把飛機搬到旁邊 → 第二次 ARM(**VIO 同時重置**)→
第二次飛行 → 上鎖。用來試時間軸選段 —— 整份 log 一起算 ATE 會因為 VIO 換了座標框而
大幅失真(約 7.7 m),逐段算才準(約 0.9 m)。

---

## 版面

```
┌ 左側分頁 ───────────┐┌ 欄位圖(畫了欄位才展開;背景依飛行模式上色)──────────┐
│ ATE 分析 │ 欄位 │    ││                                                       │
│ 參數 │ 訊息          │├ 3D 地圖 │ 資料總覽 │ 軌跡播放 │ 平面軌跡 │ … │ 報告 ────┤
│                     ││                                                       │
│                     │├ 時間軸:飛行段 / 模式 / 高度 / 速度 / 誤差 + 播放控制 ────┤
└─────────────────────┘└───────────────────────────────────────────────────────┘
```

所有東西共用一個播放頭:3D 地圖的機體、欄位圖與各分析圖的游標、訊息表的目前列。
點地圖航跡、點欄位圖、雙擊訊息,都會把播放頭跳到那一刻。

---

## 使用流程

1. **開啟 `.bin`**(或按「模擬資料」)。解析在背景進行,大檔也只要零點幾秒
   (120 分鐘、88 MB 的 log 約 0.3 s)。載入後先看 **3D 地圖** 與底部 **時間軸**:
   哪裡有 ARM、飛了哪些模式、VIO 有沒有重置。
2. 想看任何欄位(姿態、振動、電池、EKF 創新值…)→ 左側 **「欄位」**:勾欄位或點預設組合。
3. 做 ATE → 左側 **「ATE 分析」**:參考(基準)通常選 `GPS`,估測(受測)選 `VISP`,
   選空間對齊方式(見下節)。
4. 一份 log 有好幾次 ARM 時,在時間軸 **「飛行段」** 選要分析的那一次(見下節)。
5. 兩邊時間戳若有系統性延遲,按 **「自動估」**(只在分析區間內用速度剖面互相關)。
6. 看各分頁,或匯出 CSV / 報告 / PNG。

---

## 3D 地圖

- 衛星底圖上的**參考軌跡依飛行模式著色**(右上角圖例列出這份 log 用到的模式);
  也可改成 **依 ATE 誤差**(只在分析區間內上色)或 **參考藍 / 估測橘**。
- **3D 機體**(四軸 / 定翼 / 車,依 log 的機型自動選)依 `ATT` 的 roll / pitch / yaw 轉動,
  紅色是機頭;沒有 `ATT` 時改用 `AHR2`,再沒有就用航向推算偏航。
- 估測軌跡為細線(已套用 ATE 對齊與時間偏移),橘點 = 估測此刻位置,白線 = 此刻誤差;
  虛線是機體到地面的垂線。
- 左上數值框:時間、ARM 段 / 地面、模式、高度、水平速度、ATE 誤差、姿態。
- **點航跡 = 跳到那一刻**;「跟隨飛機」保留你的縮放與環繞角度追著機體;「全覽」看整條航跡。
- 滑鼠:左鍵拖曳旋轉、右鍵拖曳或滾輪縮放、中鍵拖曳傾斜。

---

## 欄位繪圖(UAV Log Viewer 式)

左側 **「欄位」**:

- **搜尋**訊息或欄位(`ATT`、`Roll`、`gps[1].sp` 都可以)。
- **預設組合**:姿態、高度、速度、**VIO / 外部導航**(VISP vs XKF1、VIO 偏航 vs AHRS、
  PErr/AErr、重置 / 忽略、VISV、XKF3 創新值、EKF 來源切換…)、GPS、EKF、振動、電力、
  RC 輸入、馬達輸出、控制器、羅盤、板子。log 裡沒有的欄位會自動隱藏。
- **訊息欄位樹**:訊息 → instance → 欄位,勾選即畫;右鍵可選畫到哪條軸。
- **已繪製**清單:改顏色、改軸(最多 4 條 y 軸:左、右、外右、外左)、隱藏、移除。
- **新增運算式…**:可寫
  `degrees(atan2(VISP.PY, VISP.PX))`、`GPS[1].Spd*3.6`、`mag(VIBE.VibeX, VIBE.VibeY, VIBE.VibeZ)`、
  `wrap_180(VISP.Yaw - ATT.Yaw)`、`ddt(CTUN.Alt)`、`lowpass(BAT.Curr, 2)`。
  - 欄位寫法 `訊息.欄位` 或 `訊息[instance].欄位`;運算子 `+ - * / ** % //` 與比較(結果 0/1)。
  - 函式:abs sqrt sin cos tan asin acos atan atan2 degrees radians hypot mag min max log log10
    exp floor ceil sign clip wrap_180 wrap_360 rel(減去第一個值)ddt(對時間微分)lowpass(一階低通)。
  - 常數 pi、e、g。時間軸以第一個欄位為準,其他欄位內插過來,在各自範圍外或資料空洞內為 NaN;
    航向類欄位(Yaw、GCrs…)跨過 0/360° 時走短弧內插,不會內插出假的 180°。
  - 只允許上述語法(白名單解析,不會執行任意 Python)。

欄位圖:背景依飛行模式上色、上緣寫模式名稱、下緣綠條為 ARM 段;
**滾輪縮放時間、拖曳平移、雙擊全覽、點一下跳到該時刻**;
縮放會和底部時間軸**雙向同步**。游標旁的數值框列出每條序列此刻的值
(整數 / 狀態類欄位取當下的實際樣本、航向走短弧,不會出現 log 裡不存在的數值)。
「檢視 → 欄位圖加上 ATE 3D 誤差」可把 ATE 誤差和任意欄位放在一起對照。

左側 **「參數」**:全部 `PARM` 的數值與預設值,標出改過的參數、飛行中變更的次數與歷史
(滑鼠停留看),可只看已修改、複製、匯出 Mission Planner 相容的 `.param`。

左側 **「訊息」**:`MSG` 文字、事件(`EV`,如 EKF 來源切換、降落完成)、錯誤(`ERR`)、
模式、ARM / DISARM;可依類型與文字篩選,**雙擊跳到該時刻**,勾「跟隨播放頭」會自動捲動。

---

## PID 調參建議(「PID 調參」分頁)

比較每一軸(Roll / Pitch / Yaw)的**期望 vs 實際**,找出「追不上」與「震盪」,
對應到 ArduCopter 參數的調整方向,例如 `ATC_RAT_RLL_P 0.405 → 0.324(−20 %)`。

- **分析範圍** = 目前的分析區間(在時間軸選 ARM 段最準);只看離地飛行的部分(裁掉起降前後各 2 s)。
- **追蹤**:RMS / 最大誤差、延遲(交叉相關)、懸停穩態偏差;**頻率響應**(共振峰 = 會過衝、
  −3 dB 頻寬 = 反應快慢);**步階響應**(反卷積估計:過衝 %、上升時間);**頻譜**找持續震盪。
- **症狀 → 方向**(多旋翼,每次 10–25 %、一次一項):
  | 症狀 | 優先調整 |
  |---|---|
  | 中頻持續震盪(約 4–15 Hz) | 降 `ATC_RAT_<軸>_P`(Roll / Pitch 的 I 跟著 P) |
  | 高頻震盪 / D 項主導 | 降 `ATC_RAT_<軸>_D`,再降 P;檢查濾波 / 陷波 |
  | 角速度環欠阻尼(過衝大、無持續震盪) | 降 P 或加 D(依 D/P 比例決定) |
  | 反應慢、延遲大、無過衝 | 加 `ATC_RAT_<軸>_P`(D 等比例) |
  | 角度環過衝 / 慢速擺動 | 降 `ATC_ANG_<軸>_P` |
  | 懸停長期偏一邊 | 加 `ATC_RAT_<軸>_I`;檢查重心 / 馬達 / 槳 |
  | Yaw 追不上 | 加 `ATC_RAT_YAW_P`(I ≈ P/10) |
- **先處理再調 PID**:振動過大(VIBE > 30 m/s²、clipping)、馬達飽和(上限 = 推力不足;
  下限 = 多半是噪音)、陀螺儀馬達噪音(→ 陷波濾波)。這些情況不會建議加大增益。
- **資料夠不夠**:ATT / RATE 只記 10 Hz 時(Nyquist 5 Hz),一般 PID 震盪(5–20 Hz)看不到、
  還會混疊成假的低頻 —— 分頁會明講哪些無法判斷,並給出該開的記錄位元
  (`LOG_BITMASK` 加 ATTITUDE_FAST bit 0 + PID bit 12;`INS_LOG_BAT_MASK = 1` 量馬達噪音)。
  有 `PIDR / PIDP / PIDY` 時還會看 P / I / D 各自的貢獻、Dmod / SRate(ArduPilot 的震盪抑制)。
- 建議表可**複製**、**匯出 .param**(只含有建議值的參數)、**複製報告文字**;
  點診斷項目的「跳到」會把播放頭移到最能看出問題的時刻。

> ⚠ **建議僅供參考。** 一次只改一個參數、每次 10–25 %,先在空曠處以 STABILIZE / ALT_HOLD 低空試飛;
> 不確定時優先使用 ArduPilot 的 AutoTune(調參流程:初始參數 → 振動 / 陷波 → AutoTune)。

沒有 log 也能試:「檔案 → 載入 PID 調參模擬情境」有 10 種情境(good、p_high、d_high、p_low、
i_low_bias、ang_p_high、yaw_p_low、saturation、vibration、noisy_notch),來自 ArduCopter
串級控制器的閉迴路模擬;勾「模擬一般 log 的記錄方式」可看 10 Hz log 能判斷到哪裡。
命令列:`python main.py --cli tune <log.bin> --segment 1 [--param out.param]`、
`python main.py --cli tune --scenario p_high [--slow]`。

---

## 時間軸與播放 —— 一份 log 不等於一次飛行

ArduPilot 常見是「每次 ARM 一份 log」,但只要開了 `LOG_DISARMED`(或沒依 ARM 切檔),
一份 `.bin` 就會涵蓋**整次開機**:前後一長段地面待機、中間可能起降好幾次,
VIO 也可能在兩次 ARM 之間重置。整份一起算 ATE 會被地面段稀釋,VIO 重置後
還會變成兩個座標框硬湊一個對齊 —— 數字沒有意義。視窗下方的時間軸就是為此設計的:

| 時間軸上的東西 | 意義 |
|---|---|
| 綠色帶「ARM n」 | 一次 ARM → DISARM 的飛行段(`ARM` 訊息;舊韌體用 `EV` 10/11) |
| 「模式」列 | 依飛行模式上色的色帶(和地圖、欄位圖同一套顏色) |
| 黃色 ▼ | VIO 重置(`VISP.Rst` 計數改變) |
| ◆ | 事件(`EV`:EKF 來源切換、降落完成、EKF 偏航重置…) |
| 紅色 ▲ | 錯誤 / 失效保護(`ERR`) |
| 高度 / 速度列 | 參考軌跡的高度與水平速度 —— 一眼看出哪裡在飛、哪裡停在地上 |
| 誤差列 | 計算 ATE 後出現,只涵蓋分析區間 |
| 亮區 + 刻度列上的把手 | 目前的分析區間(區間外變暗) |
| 白色直線 | 播放頭 |

操作:

- **選段**:「飛行段」選單選某次 ARM,或雙擊綠色帶 → 分析區間設成那一段並重算。
  每段各自對齊,VIO 重置前後才不會互相污染。
- **微調區間**:拖曳刻度列上的區間把手;或把播放頭移到想要的位置按「[ 設起點」「設終點 ]」。
- **播放 / 拖曳**:點擊或拖曳時間軸跳到該時刻;▶ 播放,速度 0.25×~50×。
  勾「只播分析區間」時只在區間內播,播到終點自動停。
- **縮放**:滾輪以游標為中心縮放、Shift+滾輪平移、雙擊刻度列或右鍵「全覽」還原。
- 滑鼠停在時間軸上會列出該時刻附近的事件、模式與各列數值。

快捷鍵(焦點在右側時;點一下右側任何地方即可,也列在「播放」選單):

| 鍵 | 動作 |
|---|---|
| 空白鍵 | 播放 / 暫停 |
| ← / → | 後退 / 前進 1 秒(加 Shift 為 10 秒) |
| PgUp / PgDn | 上一個 / 下一個事件(ARM、DISARM、VIO 重置、EV、ERR、區間邊界) |
| Home / End | 播放範圍起點 / 終點 |
| `[` / `]` | 把播放頭位置設為分析起點 / 終點 |

時間軸涵蓋到 log 的**最後一筆訊息**:`LOG_DISARMED=0` 時位置資料在 DISARM 後就停,
但 `MSG` / `ERR` 還會繼續記,這些事件照樣畫在時間軸上、點得到。

若 log 的第一個狀態事件是 DISARM(ARM 事件沒記到,例如 4.1+ 在 `LOG_DISARMED=0`
時解鎖失敗後又成功 ARM),該段起點會推定為「DISARM 之前最後一段資料空洞之後」。

---

## 空間對齊為什麼必要

VIO 的座標原點與朝向通常跟 GPS 不同框(VIO 從開機位置起算、朝向由初始化決定),
不對齊就算出一堆「其實只是座標系差異」的誤差。

| 方式 | 自由度 | 什麼時候用 |
|---|---|---|
| **SE(3)**(預設) | 平移 + 旋轉 | VIO 與 GPS 不同框。最常用 |
| **Sim(3)** | 再加尺度 | 想量 VIO 的**尺度誤差** —— 看報告的「尺度」欄位 |
| **4DOF** | 偏航 + 平移 | 重力方向已對齊(只差水平旋轉) |
| **僅平移** | 平移 | 只想去掉原點差 |
| **不對齊** | — | 兩者本來就同框(例如都已是 EKF 原點座標) |

對齊採 Umeyama 最小平方解。注意:**任何含平移的對齊都會讓各軸平均誤差歸零**
(質心被對齊),所以「平均偏差」在對齊後沒有資訊量 —— 軸向誤差圖畫的是
**移動平均趨勢線**,那才看得出低頻漂移。

---

## 各分頁在回答什麼問題

| 分頁 | 回答 |
|---|---|
| **3D 地圖** | 飛機在哪、飛了哪些模式、姿態如何;VIO 軌跡和 GPS 在實地上差多少 |
| **PID 調參** | 各軸追得上嗎、有沒有震盪、該調哪個參數往哪個方向(見上節) |
| 資料總覽 | log 裡有哪些位置來源、各自長什麼樣、時間涵蓋是否重疊、ARM 段在哪 |
| 軌跡播放 | 2D 俯視的播放版本(含尾巴與即時誤差,可跟隨) |
| 平面軌跡 | 兩條軌跡的整體對照,細線是每時刻的誤差連線 |
| **誤差熱點** | **在場地的哪個位置誤差變大**(誤差映成顏色畫在軌跡上) |
| 3D 軌跡 | 含高度的立體對照 |
| 誤差時序 | 誤差**何時**變大;VIO 漂移會看到單調上升 |
| 軸向誤差 | 是**哪個方向**在偏(東/北/天三軸小倍數 + 移動平均趨勢) |
| 誤差分佈 | 直方圖與累積分佈;P95 = 95% 的樣本落在多少誤差之內 |
| 統計表 | 全部數值的表格版 |
| 報告 | 可直接貼進文件的文字報告(含分析區間涵蓋哪些 ARM 段) |

左側數值欄位(起訖、偏移、間隙)打完或停手約 0.3 秒才重算;滑鼠沒點進欄位時滾輪不會改到數值。
沒顯示的分頁要切過去才畫,大 log 改設定也不會卡住。

設定改了但算不出 ATE(區間太短、兩者沒有重疊、參考 = 估測)時,
結果磚、統計表、報告與結果圖會一起清空並顯示原因,不會留著舊區間的數字。

---

## 統計定義

同步後成對位置差 `Δ = 估測 − 參考`(ENU,公尺):

- **3D 誤差** = ‖Δ‖;**水平誤差** = ‖Δ_EN‖(只取東、北)
- **RMSE / 平均 / 中位數 / 標準差 / P90 / P95 / 最大**
- 軸向(東 E / 北 N / 天 U)的**平均與中位數保留正負號**(代表系統性偏差方向);
  最大 / P90 / P95 取絕對值

時間同步:把估測軌跡線性內插到參考軌跡的時間戳上,只取兩者重疊的區間。
估測軌跡若有資料空洞,超過「內插間隙上限」的區段會被剔除,不會跨洞內插出假資料。

> ⚠ **GPS 本身有 0.3~2 m 誤差。** 拿 GPS 當參考時,算出來的是「VIO 與 GPS 的差異」,
> 不是 VIO 的絕對精度。要真值請用 RTK 或動作捕捉(OptiTrack)。

---

## 命令列(headless)

```powershell
python main.py --cli info <log.bin>                      # 列出位置來源、ARM 段、模式、VIO 重置
python main.py --cli ate <log.bin> --ref GPS --est VISP  # 算 ATE
python main.py --cli ate <log.bin> --ref GPS --est VISP --align sim3 --auto-offset
python main.py --cli ate <log.bin> --csv out.csv --report out.txt --t0 20 --t1 180
python main.py --cli ate <log.bin> --segment 2           # 只算第 2 次 ARM 的飛行段
python main.py --cli ate --sim --align se3               # 用模擬資料
python main.py --cli info --sim-multi                    # 多航段模擬資料
```

`ate --segment N` 把分析區間設成第 N 段(不能與 `--t0` / `--t1` 併用);
`--auto-offset` 只在分析區間內搜尋。輸入有誤或算不出 ATE 時印出原因並回傳 2。

---

## 支援的 log 訊息

欄位繪圖、參數與訊息面板可讀 **log 裡的任何訊息**(含多 instance,如 `GPS[0]` / `GPS[1]`、
`XKF1[0..2]`、`BAT[0]`)。下表是會被當成**位置來源**或**時間軸事件**的訊息:

| 訊息 | 座標框 | 說明 |
|---|---|---|
| `GPS` / `GPS2` / `GPSB` | 經緯高 | 衛星定位。VIO 若以 `GPS_INPUT` 注入會出現在 instance 1(顯示為 `GPS[1]`) |
| `POS` | 經緯高 | EKF 融合後的位置 |
| `VISP` | 局部 NED | `VISION_POSITION_ESTIMATE` —— ROS2 餵進來的 vision position |
| `XKF1` / `NKF1` | 局部 NED | EKF3 / EKF2 內部位置估計(每個 core 一條) |
| `AHR2` | 經緯高 | 次要 AHRS 解 |
| `ORGN` | — | EKF 原點,用來把經緯度換算成局部 ENU |
| `ATT`(`AHR2`) | — | 3D 機體姿態 |
| `ARM` / `EV` | — | 解鎖 / 上鎖 → ARM 飛行段(有 `ARM` 就用它;舊韌體只有 `EV` 10/11);其他 `EV` 依名稱標成事件 |
| `ERR` | — | 錯誤 / 失效保護(子系統 + 代碼,例如 `FAILSAFE_BATT:觸發 (1)`) |
| `MODE` | — | 飛行模式(名稱依 `MSG` / `VER` 判斷出的機型解出;認不出機型時只顯示編號) |
| `VISP.Rst` | — | 計數改變 = VIO 重置(ArduPilot 4.1+;4.0 以前的 VISP 沒有重置計數) |
| `MSG` / `PARM` | — | 訊息面板 / 參數面板 |

`GPS` 會濾掉非 3D-fix 的樣本。若 log 沒有 `ORGN`,程式改用參考軌跡的第一點當原點
(只影響絕對座標,不影響 ATE 數值)。找不到位置訊息時,錯誤訊息會列出這份 log
**實際包含的所有訊息型別**。要加新的位置來源:改 `core/logio.py` 的 `CANDIDATE_TYPES`。

解析方式:沿用 pymavlink 的位移索引,每種訊息用 numpy 一次解碼(結果與 pymavlink 逐位元相同),
比逐筆解析快 10~30 倍。

---

## 專案結構

```
core/
  frames.py    WGS84 ↔ 局部 ENU、NED ↔ ENU 轉換
  types.py     TrajSource / LogData / LogEvent / TimeSegment 資料模型
  dflog.py     LogIndex:任意訊息 / 欄位的向量化讀取、參數、訊息、EV / ERR 名稱
  logio.py     .bin → LogData(位置來源、原點、事件、索引)
  expr.py      欄位運算式(白名單解析、內插、ddt / lowpass)
  presets.py   ArduPilot 欄位繪圖預設組合
  sync.py      時間同步、時間偏移自動搜尋
  align.py     SE(3) / Sim(3) / 4DOF / 平移 對齊(Umeyama)
  metrics.py   ATE 統計
  pipeline.py  完整計算管線 + 報告 / CSV 匯出
  timeline.py  ARM 飛行段、模式查詢、逐時內插、播放時鐘(不依賴 Qt)
  tuning.py    PID 調參分析(頻譜、頻率響應、步階響應、症狀 → 參數建議)
  tuning_model.py  調參報告的資料模型
  sim_tuning.py    ArduCopter 串級 PID 閉迴路模擬(10 種調參情境,驗證建議方向)
  sim.py       模擬資料(含已知注入誤差;另有開機到關機、兩次 ARM 的多航段版本)
  fakebin.py   把模擬資料寫成 .bin(供測試解析路徑)
ui/
  theme.py     深色主題、色彩角色、飛行模式配色、matplotlib rcParams
  main_window.py
  map3d.py     3D 地圖(QtWebEngine + CesiumJS)與場景資料準備
  models.py    程式產生的 3D 機體模型(glTF:四軸 / 定翼 / 車)
  web/         地圖頁面(map3d.html / .js / .css)
  fieldplot.py 欄位圖(多 y 軸、模式色帶、降採樣、游標數值)
  sidebar.py   欄位 / 參數 / 訊息 面板
  tuning.py    「PID 調參」分頁(各軸摘要、診斷、建議表、追蹤 / 頻譜 / 頻率響應 / 步階響應圖)
  plots.py     ATE 分析圖表(含播放頭游標與「軌跡播放」)
  timeline.py  時間軸(TimelineWidget)與播放列(PlaybackPanel)
  widgets.py   卡片、統計磚、統計表
cli.py         命令列
tests/         見下節
output/
  sample_sim.bin           模擬資料寫成的 .bin,可用來試「開啟 .bin」流程
  sample_multi_flight.bin  開機到關機一份 log、兩次 ARM + VIO 重置,試時間軸選段
  shots/                   tests\shot.py 產生的圖表範例
```

## 測試

```powershell
python tests\test_core.py         # 核心驗證
python tests\test_dflog.py        # log 索引:與 pymavlink 逐欄位比對、載入速度
python tests\test_expr.py         # 欄位運算式 / 預設組合
python tests\test_timeline_ui.py  # 時間軸 / 播放列元件
python tests\test_plots_ui.py     # 分析圖表:延後繪製、播放游標、跟隨模式
python tests\test_fieldplot_ui.py # 欄位圖與側欄面板
python tests\test_map3d.py        # 3D 地圖場景與元件(加 --visual 在實體視窗驗證,需網路)
python tests\test_tuning.py       # PID 調參分析(10 情境 × 3 種記錄方式的矩陣 + 真實 log)
python tests\test_sim_tuning.py   # 調參閉迴路模擬本身的正確性
python tests\test_tuning_ui.py    # 「PID 調參」分頁
python tests\test_gui.py          # 主視窗煙霧測試(會短暫開關視窗)
python tests\shot.py              # 把每張圖存成 PNG 到 output/shots
```

- `test_core.py`:座標轉換往返精度、對齊還原已知變換、ATE 統計、時間同步與空洞保護、
  時間偏移搜尋、`.bin` 往返;ARM 飛行段配對與起點推定、`ARM` / `EV` 去重、模式名稱、
  VIO 重置、播放時鐘、多航段模擬「逐段 ATE 遠比整份 log 準」。
- `test_dflog.py`:兩份範例 log 每種訊息每個欄位都與 pymavlink 相同、instance 拆分、
  字串與倍率、參數 / 訊息 / EV / ERR、新舊載入結果逐位元相同、各種壞 log;
  `DFLOG_EXTRA_LOGS="a.bin;b.bin"` 可加測自己的 log。
- `test_gui.py`:主視窗、背景載入、每張圖、對齊 / 裁切 / 偏移 / 換來源、匯出、
  時間軸播放與選段、各圖游標、算不出結果時清空舊結果、欄位圖展開 / 收起與同步、
  參數 / 訊息面板、地圖點擊跳轉、焦點與快捷鍵、連續載入以最後一個為準、壞檔不影響原 log。

---

## 色彩

介面與圖表用固定的**角色配色**(顏色跟著實體走,不跟著排序):
參考=藍、估測=橘、誤差=洋紅、誤差(水平)=紫;誤差大小用單一藍色相的連續色階。
**飛行模式**另有一套類別色(避開上述角色色),地圖航跡、欄位圖背景、時間軸「模式」列共用;
模式色永遠和模式名稱一起出現(圖例、色帶文字)。
所有配色組合都通過 OKLab 色覺缺陷分離度(protan / deutan ΔE)與表面對比度檢查;
序列身分一律同時有圖例與文字標籤,不單靠顏色傳達。
#   l o g _ v i e w  
 