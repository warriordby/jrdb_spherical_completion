# JRDB Completion Quality Gate

正式主线只接受连续视频窗口，不再用离散单帧作为验收依据。首轮固定三类各 81 帧：

- 室内近人：`cubberly-auditorium-2019-04-22_1`
- 室外行走：`discovery-walk-2019-02-28_0`
- 反光/复杂地面：`indoor-coupa-cafe-2019-02-06_0`

硬门禁：

1. 输出为 `3760×720`，帧数、文件名和 15 FPS 元数据与输入一致。
2. 中央 `y=[120,600)` PNG 回读后逐像素一致，`max_abs_error=0`。
3. synthetic mask 顶部/底部为 255、中央为 0；合成区只进入 ignore，不写入 GT。
4. 生成区新增 person 检测数为 0；无文字、水印、重复人物、漂浮肢体与突现实体。
5. ERP 接缝误差不得成为相邻水平梯度的孤立尖峰；窗口边界连续 3 帧异常立即失败。

成对门禁初始阈值：

- 原观测区 HOTA、IDF1、MOTA 下降均不超过 1.0 个百分点。
- 原观测区 person recall 相对变化不超过 1%。
- 已匹配轨迹 ReID cosine distance 的 P95 增量不超过 0.05。
- 合成区 temporal warp error P95 不超过相邻观测边界带基线的 1.25 倍。

执行：

```bash
bash scripts/run_gpu_smoke.sh
```

当前 test 图像没有 MOT GT，HOTA/IDF1/MOTA 状态必须保持 `pending_ground_truth`；不得用伪标签把该项标成通过。三类 81 帧全部通过后运行 243 帧，再进入完整短序列。任何门禁失败都回到时序/球面增强阶段，不得启动 27,661 帧全量生产。
