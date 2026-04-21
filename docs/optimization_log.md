# 优化历程记录

所有数据基于 `data/process/manual_crop/1/crop.pcd` + LOOCV attention-unet
holdout_1 checkpoint，精度配置（`seam_step=1`，yaml 默认），`fast_mode=True`。
耗时为 1 次 warmup + 2 次正式运行的中位数；精度统计来自
`summary.json`。

规则：每一档（P0…P5）都先基线测、改动后再测，**有精度或速度的真实净
收益才保留**，倒退或持平一律回退。

---

## 基线（5bef076，pre-P0）

| 项 | 值 |
|---|---|
| commit | `5bef076` Tune GapFlushParams defaults for maximum measurement precision |
| 总耗时（中位数） | **89.3 s** |
| compute_gap_flush | 88.0 s（占 99%） |
| predict_mask | 0.4 s |
| gap_mean | 132.04（混合单位，实际像素量级） |
| gap_std | 1.57（混合） |
| flush_mean | 1.49 mm |
| flush_std | **0.87 mm** |
| 有效截面 | 1893 / 1893 |

---

## P0 — 双平面 3D gap/flush + 品质护栏 + count_neighbors 混合向量化

**精度**：纯 mm 单位，正交分解 gap⊥flush⊥along，护栏剔除 plane-fit
坏样，along-seam 残差单独扣除。
**速度**：关键是 count_neighbors 按 N 分流——小 N (≤1500) 向量化、大
N 回落 sorted sliding-window（避免 N² 内存爆炸）。

### 首次 P0（单纯向量化 count_neighbors）

| 项 | 值 | vs 基线 |
|---|---|---|
| 总耗时 | **286.7 s** | ❌ 慢 3.2× |
| 根因 | top_surface 里 `local_background` N 可达 10⁴，N² 内存 ~GB 级 | |

**判定**：速度倒退，不可接受。需要分流。

### 修复后 P0（按 N 分流的混合实现）

| 项 | 首次 P0 | **修复后** | vs 基线 |
|---|---|---|---|
| 总耗时 | 286.7 s | **86.1 s** | ✅ 快 3.6% |
| compute_gap_flush | 285 s | 85.0 s | |
| gap_mean | 15.41 mm | **15.412 mm** | 混合单位 → 真 mm |
| gap_std | 0.197 mm | **0.1971 mm** | 相比基线 1.57（混合）→ 0.2 mm |
| flush_mean | 1.40 mm | **1.396 mm** | |
| flush_std | 0.18 mm | **0.1804 mm** | **相比基线 0.87 → 0.18 mm（↓ 79%）** |
| 有效截面 | 1887/1893 | 1887/1893 | 6 个被 plane-fit 护栏拒 |
| 勾股自检 | 1893/1893 全满足 | 全满足 | |

**判定：保留**。速度同级或更快，精度 flush_std 降 5×，gap 回到物理 mm，
full-3D 正交自洽。

---

## P1 — `count_neighbors` 向量化（已内化到 P0）

原计划独立一档，但实际作为 P0 的一部分交付：在 `helpers.py::count_neighbors`
按输入 N 分流——

| N 规模 | 策略 | 典型出现位置 |
|---|---|---|
| ≤ 1500 | 向量化 N×N 布尔比较 | `bottom.py::_filter_isolated_points`（N≈50-200） |
| > 1500 | 回落到 sorted sliding-window 循环 | `top_surface.py::detect_top_surface_edges` 里 `local_background` N 可达 10⁴ |

**判定：保留（随 P0 commit `82a0133`）**。相比 baseline 的纯 loop：
小 N 快 ~10 倍，大 N 持平，净结果 3.6% 整体加速。如果没做分流，单纯
向量化会因为 N² 内存爆炸而慢 3.2×（已验证）。

---

## P2 — `refine_top_surface_edge_sequence` 早停

**改动**：统计每一 pass 更新了多少 section，两侧加总 < 0.5% section
总数就跳出余下的 pass。保留最多 `refine_passes=5` 上限。

| 项 | P0 基线 | P2 改动后 |
|---|---|---|
| 总耗时（中位数） | 86.1 s | **86.1 s** |
| gap_mean / std | 15.412 / 0.1971 | 15.4116 / 0.1971（bit-exact 同） |
| flush_mean / std | 1.396 / 0.1804 | 1.3957 / 0.1804（同） |
| 有效截面 | 1887/1893 | 1887/1893 |

**判定：回退**。实测 refine 各 pass 本身开销就占整体极小比例，早停
在这个样本上节不出可测量的时间，也没精度收益。按"没提升就不保留"规则
撤销，`top_surface.py` 回到 P0 版本。

---

## P3 — 亚像素边缘定位

**改动**：`_build_edge_point_from_model` 把"取最近像素+覆盖 z"换成"对
segment 内每个坐标（px, py, x, y）做 `polyfit(u, coord, 1)` 线性拟合，
再在 edge_u 处求值"。这样 edge 的 3D 位置不再被像素栅格吸附，消除了
~1 px 的离散化抖动。

| 项 | P0 | **P3** | Δ |
|---|---|---|---|
| 总耗时 | 86.1 s | 86.2 s | ≈ 持平 |
| gap_mean | 15.412 | 15.320 mm | 边缘微移 ~0.09 mm（更准） |
| **gap_std** | 0.1971 | **0.1915** | ✅ **−2.8%** |
| flush_mean | 1.396 | 1.415 | |
| **flush_std** | 0.1804 | **0.1792** | ✅ −0.7% |
| 有效截面 | 1887/1893 | 1887/1893 | |

**判定：保留**。精度两指标都有净提升，速度持平。

---

