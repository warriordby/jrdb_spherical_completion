# JRDB 球面视角补全与 MOT 可用性技术方案

状态：设计冻结，待实现与小样验收  
版本：2026-08-14  
设备：RTX PRO 6000D 84 GB  
输入：JRDB stitched `3760×480`，27 个序列、27,661 帧  
输出：`3760×720` 图像序列及 MOTChallenge 视图

## 1. 目标与不可违反条件

1. 输出中央 `y=[120,600)` 必须与输入逐像素一致，`max_abs_error=0`。
2. 只生成顶部和底部各 120 行；不得把生成区域伪装为真实观测或真实 GT。
3. 输出保留原帧号、文件名、序列长度和 `15 FPS`，支持断点续传与确定性复现。
4. 生成内容必须跨帧稳定；不得新增行人、复制人物、改变身份或制造虚假轨迹。
5. 每帧输出 `synthetic_mask`，评测时生成区默认作为 ignore region。
6. 未通过代表序列的 MOT 质量门禁前，不运行 27,661 帧全量任务。

## 2. 现有方案结论

SDXL 单帧 tile、OpenAI Chat Image2 单帧 tile 均不再作为正式主线。二者虽能保持中央区域不变，但存在跨 tile 结构变化、跨帧闪烁、人物幻觉和第三方上游不稳定等问题。RAFT 对 RGB 结果做后处理只能传播已有像素，不能从根本上约束扩散模型的时序 latent 和身份。

`3760×720` 也不是标准 2:1 ERP：水平方向约 `10.44 px/degree`，垂直方向为 `4 px/degree`。直接在目标画布生成会让模型面对非球面等距采样，增加极区拉伸与接缝错误。

## 3. 最终技术路线

```text
JRDB 3760×480 观测带
        │ 球面坐标重采样
        ▼
标准 2:1 ERP 代理画布 1440×720
        │ 81 帧 masked video-to-video
        ▼
Wan2.1 VACE-14B + latent/flow 时序传播
        │ 球面逆映射，仅提取生成带
        ▼
覆盖原始中央带，得到 3760×720
        │ 几何、时序、检测、ReID、MOT 门禁
        ▼
images + mot/img1 + synthetic_mask + metadata
```

### 3.1 球面代理

主代理使用 `1440×720` 标准 ERP，恰好为 `4 px/degree`：

- 输入的 `120°` 垂直观测带映射到代理 `y=[120,600)`。
- 缺失纬度 `[-90°,-60°)` 与 `(60°,90°]` 分别对应 120 行。
- 经度按球面坐标从 3760 列重采样到 1440 列。
- 生成后仅将代理上下带按经度逆映射到 `3760×120`。
- 最终中央 480 行直接复制输入，禁止经过编码、缩放或色彩转换。

高质量试验可使用 `2048×1024` 代理；观测纬度仍为 `[-60°,60°]`，不能按固定 120 行处理。所有映射通过经纬度计算，横向索引必须取模。

VACE 分支采用循环经度 padding，并对画布滚动 `W/2` 做接缝修复；只融合合成区。极点行增加球面一致性检查，不修改中央观测带。

### 3.2 主生成模型：Wan2.1 VACE-14B

采用官方 masked video-to-video / Expand-Anything 路径：

- BF16 autocast，不量化权重；84 GB 单卡固定启用同精度 CPU offload，在阶段边界卸载 T5/VAE/输入，并将已计算的 VACE hints 暂存到 CPU。
- RoPE 保持 `FP64 → FP32 → BF16` 舍入链，按 token 分块计算，避免整段 FP64/complex128 临时张量的多 GiB 峰值。
- 单窗口 81 帧，窗口重叠初始值 17 帧。
- 重叠区在 latent/噪声层保持一致，禁止只做逐帧 RGB 淡入淡出。
- 固定序列级 seed；相邻窗口复用重叠帧、参考 latent 和条件视频。
- mask 内生成，mask 外强制恢复代理观测像素。
- prompt 只允许延续天空、天花板、墙面、地面、光照和反射，禁止新增实体、文字与标志。

主线目标是先实现稳定的 `1440×720` 代理结果，再评估 `2048×1024`；显存未拉满时优先增加代理分辨率、窗口长度或常驻模块，不通过增加独立 tile 数提高占用。

### 3.3 时序增强：Seen-to-Scene 思路

将当前“RAFT warp 后混合 RGB”升级为：

1. 在观测带估计双向光流，并对缺失带完成 flow。
2. 从窗口内按相似度选择参考帧，而非只依赖上一帧。
3. 在 VAE latent 中传播可见结构，再用轻量 refinement 修复传播误差。
4. 将传播 latent 作为 VACE 条件，生成无法可靠传播的区域。
5. 前后向不一致、遮挡和低置信度区域交回生成模型，不强行 warp。

第一阶段可独立接入官方 Seen-to-Scene 推理；若其分辨率或底模不兼容，则移植“参考帧选择 + flow completion + latent propagation”到 VACE wrapper。

### 3.4 360° A/B 分支：CubeComposer

CubeComposer 与任务的球面输出最匹配，采用 cubemap face × 时间窗口生成，可原生输出 2K/3K/4K 360°视频。作为第二分支验证：

- 将 JRDB 观测带投影为部分 cubemap 条件。
- 增加 arbitrary known-region mask，确保已有水平观测带不被重绘。
- 生成后转回 ERP，再执行中央带逐像素覆盖。
- 若其透视输入假设无法稳定适配“完整经度、有限纬度”的观测带，则保留为研究分支，不阻塞 VACE 主线。

### 3.5 API 的定位

`gpt-image-2` / OpenAI Chat Image2 仅用于生成少量关键帧语义参考或做画质上界对照，不直接逐帧生产 MOT 序列。第三方转接 API 可继续使用，但必须具备缓存、重试、响应指纹和失败恢复。

Seedance、Hunyuan 等闭源模型只有在接口同时支持“视频输入、区域 mask、参考帧、确定性参数、原分辨率返回”时才进入 A/B；若只能 text/image-to-video，不作为正式补全后端。

## 4. 无 GPU 下载与准备阶段

所有资源固定在 `/root/autodl-tmp/model_cache_v2`，生成 `resources_manifest_v2.json`，记录来源、版本/commit、文件大小、SHA-256 和 `ready=true`。

主线必须准备：

- `ali-vilab/VACE` 官方代码，固定 commit。
- `Wan-AI/Wan2.1-VACE-14B` 官方权重。
- VACE 所需 Wan VAE、文本编码器及预处理资源。
- RAFT-Large 与 flow-completion/Seen-to-Scene 权重。
- 固定 person detector、ReID 模型和 TrackEval。

A/B 可选准备：

- `TencentARC/CubeComposer` 代码、`cubecomposer-3k` 权重及配置。
- Follow-Your-Canvas 代码和权重，仅作为高分辨率 outpainting 基线。
- PanoWorld 代码/权重，仅用于深度、轨迹与球面一致性研究。

下载顺序：镜像/HF Mirror → ModelScope → 官方源；分块下载支持续传。任何缺失、`.incomplete`、大小不符或校验失败都不得写入 ready 标志。下载阶段不初始化 CUDA。

## 5. 实施阶段

### P0：代码与资源准备，不开 GPU

- 新增 VACE 独立环境、下载 profile、锁定依赖和离线启动脚本。
- 实现 `3760×480 ↔ 1440×720 ↔ 3760×720` 球面映射。
- 添加经度循环、极点、mask、逆映射和中央零误差单测。
- 实现窗口清单、缓存指纹、断点状态和输出元数据。
- 准备检测、ReID、TrackEval 和 synthetic ignore mask 工具。

### P1：GPU 冒烟测试

每类先取连续 81 帧，不能只取离散单帧：

- 室内近人：`cubberly-auditorium-2019-04-22_1`
- 室外行走：`discovery-walk-2019-02-28_0`
- 反光/复杂地面：`indoor-coupa-cafe-2019-02-06_0`

先跑 `1440×720 / 81 frames`，记录峰值显存、速度、窗口接缝和失败原因。通过后再跑每类 243 帧及 `2048×1024` A/B。

2026-08-14 RTX 6000D 冒烟结果：同一 81 帧窗口连续完成 3/50 个采样 step，采样期占用约 83,230 MiB、剩余约 1,804 MiB，每 step 约 157 秒；为释放测试 GPU 主动中断，未将该窗口标记为完成。完整 primary + rolled 预计约 4.4 小时。

### P2：时序与球面增强

- 加入重叠窗口 latent 复用。
- 加入 reference-guided latent/flow propagation。
- 对 VACE 与 CubeComposer 做相同输入、seed、帧段的盲测。
- 仅在已有方案通过结构门禁后尝试 API 关键帧参考。

### P3：小规模 MOT 验收

三类代表序列分别运行完整短片段。冻结检测器、ReID 与 tracker，对原始序列和补全序列做成对评测。任何门禁失败都回到 P2，不开始全量。

### P4：全量生产

- 先检查至少 150 GiB 可用空间。
- 按序列运行，原子写入，完成一帧即登记状态。
- 每个序列完成后立即执行 verify 和 MOT 门禁。
- 失败序列隔离，不覆盖已通过结果；最终生成全量 manifest。

## 6. MOT 质量门禁

### 6.1 硬门禁

- 输出尺寸 `3760×720`，帧数和文件名完全匹配输入。
- 中央区域 `max_abs_error=0`。
- ERP 左右接缝、上下连接行无孤立跳变。
- 生成区新增 person 检测数为 0；无法排除的检测全部进入 ignore，绝不写入 GT。
- 无文字水印、重复人物、漂浮肢体、跨帧物体突然出现/消失。

### 6.2 成对 MOT 门禁

以下为首轮阈值，需用原始序列基线校准：

- 原观测区 HOTA、IDF1、MOTA 相对下降均不超过 1.0 个百分点。
- 原观测区 person recall 相对变化不超过 1%。
- 已匹配轨迹 ReID cosine distance 的 P95 增量不超过 0.05。
- 生成区 temporal warp error 的 P95 不超过相邻观测边界带基线的 `1.25×`。
- 窗口重叠位置不得出现指标尖峰；连续 3 帧异常即判失败。

输出数据只能声明为“带合成 ignore 区域的 MOT 输入序列”。除非额外完成人工标注和审核，不得声明顶部/底部具有 MOT ground truth。

## 7. 输出结构

```text
JRDB-Spherical-VFOV180-VACE/
├── images/image_stitched/<sequence>/<frame>.png
├── mot/<sequence>/img1/<frame>.png
├── mot/<sequence>/seqinfo.ini
├── synthetic_masks/<sequence>/<frame>.png
├── metadata/<sequence>.json
├── quality/<sequence>.json
└── manifest.json
```

`metadata` 至少记录：输入哈希、模型与 commit、权重哈希、代理分辨率、seed、窗口及 overlap、prompt、mask 哈希、软件版本和生成时间。

## 8. 模型优先级

| 优先级 | 模型 | 用途 | 结论 |
|---|---|---|---|
| P0 | Wan2.1 VACE-14B | masked video-to-video 主生成 | 正式主线 |
| P1 | Seen-to-Scene | latent/flow 时序传播 | 主线增强 |
| P1 | CubeComposer | 原生高分辨率 360°视频 | A/B 分支 |
| P2 | PanoWorld | 深度/轨迹/球面一致性参考 | 研究与损失设计 |
| P2 | Follow-Your-Canvas | 高分辨率滑窗基线 | 对照组 |
| P3 | gpt-image-2 | 关键帧参考/画质上界 | 非时序主线 |
| 停止 | SDXL + RGB RAFT | 旧版方案 | 不再扩大全量 |

## 9. 参考实现

- VACE: <https://github.com/ali-vilab/VACE>
- Wan2.1 VACE-14B: <https://huggingface.co/Wan-AI/Wan2.1-VACE-14B>
- CubeComposer: <https://github.com/TencentARC/CubeComposer>
- Seen-to-Scene: <https://github.com/InSeokJeon/Seen_to_Scene>
- Follow-Your-Canvas: <https://github.com/mayuelala/FollowYourCanvas>
- PanoWorld: <https://github.com/ostadabbas/PanoWorld>
