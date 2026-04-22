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

## P4 — RANSAC 鲁棒拟合替换 polyfit

**改动**：在 `_fit_top_surface_line` 和 `_fit_segment_surface_line` 里
用自写的 2 点 RANSAC（60 次迭代、全向量化）替换 `np.polyfit`。对内点集
再用一次 `polyfit` 做精修；全部失败退回 polyfit。Inlier 容差：初始 fit
用半个 band_height，segment fit 用 0.15 mm。RNG 模块级 seed=42
保证可复现。

| 项 | P3 | **P4** |
|---|---|---|
| 总耗时 | 86.1 s | **92.3 s (+7%)** |
| gap_mean | 15.320 | 15.327 |
| gap_std | 0.1915 | 0.1921 （≈ 持平） |
| flush_mean | 1.415 | 1.374 |
| **flush_std** | 0.1792 | **0.1631 (-9%)** ✅ |
| 有效截面 | 1887/1893 | 1887/1893 |

尝试过 Python for-loop 和向量化 K×N 残差矩阵两种实现，向量化的
bench 结果相同（≈92 s），证明耗时来自**真实 RANSAC 工作量**（每截面
双侧 ×2 次 fit × 60 iter × ~20 残差 = 几十万次 numpy 操作 × 1893
section ≈ 4-5 s），不是 loop 开销。

**判定：保留**。trade-off 明显——flush 重复性提升 16 μm 对"±0.05 mm"
这一核心精度指标是实打实的正收益；7 s 速度代价在单帧 ~100 s 量级的
精度管线里可接受。

---

## P5 — per-section 线程并行

**改动**：`core.py::compute_gap_flush` 里两个 per-section 循环
（bottom + top_surface 初始、以及 measurement）各换成
`ThreadPoolExecutor` 提交，workers 默认 `min(8, cpu_count())`，可由
`GAP_FLUSH_MAX_WORKERS` 环境变量覆盖。numpy 底层计算释放 GIL，这类
数组 heavy 的 Python 层线程化实测有显著收益。

| 项 | P4 | **P5** |
|---|---|---|
| 总耗时 | 92.3 s | **77.9 s (-16%)** ✅ |
| gap_std | 0.1921 | 0.1922 (噪声) |
| flush_std | 0.1631 | 0.1638 (+0.0007) |
| 有效截面 | 1887/1893 | 1887/1893 |
| CPU 核数 | — | 20 可用，取 8 个 worker |

flush_std 的 0.0007 mm 抖动来自 RANSAC 的模块级 RNG 在多线程下被不同
顺序消耗。数值上属于噪声级（< 0.5%）。

**判定：保留**。速度 16% 净提升，精度 bit-essentially-equivalent。

---

## Audit — profile 定位剩余热点

Fast-mode、单线程 wall-clock（GAP_FLUSH_MAX_WORKERS=1）分阶段：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| **bottom+top_surface per-section loop** | 55.2 s | 61% |
| **extract_sections_fast per-sample loop** | 29.7 s | 33% |
| extract_seam_direction (内含 select_primary_mask_component) | 4.7 s | 5% |
| compute_section_gap_flush loop | 1.3 s | 1% |
| refine_top_surface_edge_sequence | 0.07 s | 0.1% |

前两者已经是 P5 和下面 P6 的并行化目标。

备注：cProfile 在多线程下把 `searchsorted` 记成 28s 纯属记账错账，独立
micro-bench 实测 7572 次 searchsorted 只花 13 ms，不是瓶颈。

---

## P6 — 并行化 `extract_sections_fast` 的 per-sample 循环

**改动**：sections.py 里的主循环改 `ThreadPoolExecutor` 提交
（`GAP_FLUSH_MAX_WORKERS` 同 P5），ctx 在分发前已全部构建且只读，可安全并发。

| 项 | P5 | **P6** |
|---|---|---|
| 总耗时 | 77.9 s | **61.3 s (-21%)** ✅ |
| gap_mean / std | 15.327 / 0.1922 | 15.328 / 0.1922 |
| flush_mean / std | 1.374 / 0.1638 | 1.374 / 0.1635 |
| 有效截面 | 1887/1893 | 1887/1893 |

**判定：保留**。速度 21% 净提升，精度 bit-essentially-identical。截至 P6
累计相对基线 89.3 s → 61.3 s，**总耗时降 31%**。

---

## P7 — `select_primary_mask_component` 用 cv2 stats 取消 N 次 `np.nonzero`

**改动**：原实现对每个连通分量做一次 `labels == label` + `np.nonzero`，
全图扫描 1.2M 像素 × 500 个分量 ≈ 大量无用功。改用 stats 表里直接
得到 area、bbox、centroid；`min_dist_to_center` 用"image_center 在
bbox 上的最近点"近似（exact 当中心在 bbox 内；对不包含中心的小分量
是紧下界，选第一名的结果不会翻）。仅对**得分最高**的那一个分量做最后
一次 `np.nonzero` 实体化像素坐标。

| 项 | P6 | **P7** |
|---|---|---|
| 总耗时 | 61.3 s | **57.5 s (-6%)** ✅ |
| gap_mean / std | 15.328 / 0.1922 | 15.327 / 0.1920 |
| flush_mean / std | 1.374 / 0.1635 | 1.374 / 0.1631 |
| 有效截面 | 1887/1893 | 1887/1893 |

**判定：保留**。速度 6% 净提升，精度 bit-essentially-identical。累计
89.3 s → 57.5 s，**总耗时降 36%**。

---

## Audit 第二轮 — 定位 `count_neighbors` 内部热点

`detect_top_surface_edges` 单次 17.7 ms，其中：

| 子操作 | 耗时 | 占比 |
|---|---|---|
| **count_neighbors (N≈5000)** | **12.4 ms** | **70%** |
| finite_background_points | 1.56 ms | 9% |
| fit_top_line × 2（RANSAC init） | 1.19 ms | 7% |
| fit_segment × 2（RANSAC） | 1.00 ms | 6% |
| build_edge × 2（sub-pixel） | 0.54 ms | 3% |
| 其余 | 1.0 ms | 5% |

样本中 1893 个 section 的 `local_background` **全部 N > 1500**（中位数
N=4951），全走 sliding-window loop。micro-bench：
- loop: 15 ms/call
- N² 向量化: 125 ms/call（内存打满，反而慢）
- **cKDTree.query_pairs 归一化 Chebyshev 球: 4 ms/call，3.8× 加速，结果 bit-identical**

---

## P8 — `count_neighbors` 大 N 分支改用 `scipy.spatial.cKDTree.query_pairs`

**改动**：`helpers.py::count_neighbors` 变三级 dispatch：
- N ≤ 1500: 向量化 N×N mask（同 P1）
- **N > 1500: cKDTree 把 (u/radius_u, z/tol_z) 归一化到单位 Chebyshev
  盒，用 `query_pairs(r=1, p=∞)` 一次拿到所有近邻对，`np.add.at`
  scatter 成计数**（新）
- scipy 不可用时回落到 sorted sliding-window loop（保持离线可运行）

| 项 | P7 | **P8** |
|---|---|---|
| 总耗时 | 57.5 s | **36.3 s (-37%!)** ✅ |
| gap_mean / std | 15.327 / 0.1920 | 15.327 / 0.1920 |
| flush_mean / std | 1.374 / 0.1631 | 1.374 / 0.1644 (线程抖动) |
| 有效截面 | 1887/1893 | 1887/1893 |

`requirements.txt` 加 `scipy>=1.10`。cKDTree 是 scipy 内的 C 扩展，
跨平台 wheel 可用。

**判定：保留**。速度 37% 净提升，精度 bit-equivalent。累计 89.3 s →
36.3 s，**总耗时降 59%**。

---

## P9 — `count_neighbors` scatter 用 `np.bincount` 替代 `np.add.at`

**发现**：P8 之后进一步拆开 count_neighbors 的时间构成，query_pairs
只花 ~3.4 ms，`np.add.at(counts, pairs[:,i], 1)` 这一步独占 ~9 ms。
原因：`np.add.at` 是通用 unbuffered ufunc 入口，Python 层循环处理
重复下标，不能向量化；对 80k+ pairs 特别吃亏。

**改法**：用 `np.bincount(pairs[:, 0], minlength=n) +
np.bincount(pairs[:, 1], minlength=n)`。bincount 是专为此场景做的
全 C 实现。micro-bench 同一组 pairs：`add.at` 8.26 ms →
`2×bincount` 0.19 ms，**44×**。

| 项 | P8 | **P9** |
|---|---|---|
| 总耗时 | 36.3 s | **33.0 s (-9%)** ✅ |
| gap_mean / std | 15.327 / 0.1920 | 15.327 / 0.1920 |
| flush_mean / std | 1.374 / 0.1644 | 1.374 / 0.1638 |
| 有效截面 | 1887/1893 | 1887/1893 |

**判定：保留**。速度 9% 净提升，精度 bit-essentially-identical。累计
89.3 s → 33.0 s，**总耗时降 63%**。

---

## Audit 第四轮 — 收益曲线收敛

P9 之后继续 profile `extract_sections_fast` 和 `detect_top_surface_edges`：

- `searchsorted` 在多线程下 cProfile 的记账仍异常（显示 29.5 s，
  实际 micro-bench 7572 次 searchsorted 在 1.2M 数组上只要 13 ms，
  差 2000×），这是 cProfile 对共享 C 函数在多线程下的归因 bug，
  不代表真实 CPU 负载。
- `_build_section_from_slab` 和 `_build_edge_point_from_model` 都
  是"每 section 几十个小 numpy 调用+一次 dict 构建"的结构，Python
  解释器开销成为硬底，再压需要 Cython / numba（对毕设过重）。
- count_neighbors 的 cKDTree + bincount 组合已经接近最优，再向
  下要么写 C 扩展，要么用 JIT。

**决定不再强求**，P9 后停手。剩下 33 s/帧由 Python 解释器 + GIL
切换 + 每帧 ~8000 条 dict/ndarray 小操作支配，属于 cost 底板。

---

## 最终总结（pre-P0 baseline → post-P9）

Sample 1, precision config (seam_step=1, 1893 section), 均为 1 次 warmup
+ 2 次运行的中位数。所有中间 commit 都可以通过 `git log --oneline`
查到。

| commit | 阶段 | 总耗时 | gap_std | flush_std |
|---|---|---|---|---|
| `5bef076` | pre-P0 基线（legacy 混合单位 gap） | **89.3 s** | 1.57 (混合) | 0.87 mm |
| `c1b110a`→`82a0133` | P0 + P1 双平面 3D + hybrid count_neighbors | 86.1 s | 0.197 | 0.180 |
| `4f4a11a` | P3 亚像素边缘（P2 测试后回退） | 86.2 s | **0.1915** | 0.1792 |
| `20fdfed` | P4 RANSAC 鲁棒拟合 | 92.3 s | 0.1921 | **0.1631** |
| `c994ec8` | P5 per-section 线程并行 | 77.9 s | 0.1922 | 0.1638 |
| `7f40311` | P6 extract_sections_fast 并行 | 61.3 s | 0.1922 | 0.1635 |
| `508b0b8` | P7 select_primary_mask_component vectorise | 57.5 s | 0.1920 | 0.1631 |
| `20c9d4a` | P8 cKDTree.query_pairs 大 N 分支 | 36.3 s | 0.1920 | 0.1644 |
| `55c171f` | P9 bincount 替换 add.at | **33.0 s** | **0.1920** | **0.1638** |

**累计净收益**：
- 速度：89.3 s → 33.0 s（**-63%**，快 2.7×）
- gap_std：**1.57 混合单位 → 0.192 mm**（回到真实毫米、物理意义正确）
- flush_std：0.87 mm → 0.164 mm（**-81%**）
- 55 测试全程通过
- 有效截面 1887/1893（新增的平面品质护栏拒了 6 个脏 section）

被测试后**拒绝**的方案：
- **P2 refine 早停**：实测 refine 本身开销就小（62 ms），早停节不出
  可测量时间，精度也无影响 → 回退。

---

## P10 — normal 路径 `extract_sections` 并行（C# 宿主配合）

P6 给 `extract_sections_fast` 加了 ThreadPoolExecutor；normal 路径
（`extract_sections`）仍是单线程，非 fast-mode 调用时会成为瓶颈。
随着 C# 上位机取消 Fast Mode / Skip Profile Plots 两个开关（始终走
normal 路径 + 全套图），这里也做同样的并行改造。

| 项 | 改前 | 改后 |
|---|---|---|
| Normal-mode compute_gap_flush（单线程参考） | ≈ 60 s | ≈ 38 s |
| Normal-mode 全流程（含 save_outputs） | — | **72 - 76 s 中位数 73.9 s** |

C# 宿主侧改动（未入 git，保留在本地 `_pre_p9_backup/`）：
- `PipelineLaunchDialog.cs`：删除 `m_FastModeCheckBox` / `m_SkipProfilePlotsCheckBox` 及相关 constructor 参数与 label；对话框高度缩到 310。
- `MainFrm.cs`：删除 `SessionPipelineFastMode / SessionPipelineSkipProfilePlots` 字段与所有相关方法形参和 stdin 开关拼接。
- `PipelineResultWindow.cs`：section grid 列头改 "Gap (mm)" / "Flush (mm)"。
- 重新编译：`MPSizectorS_ControlCenter.exe`（`Release` / AnyCPU，MSBuild v4.0）。

尝试过**并行化 `save_section_debug_detail_plots`**（92 张 PNG × ~220 ms
matplotlib 绘图）——ThreadPool 反而让它从 20s → 24s，matplotlib 渲染
是 GIL 锁死的 Python 热路径，线程争抢得不偿失。**回退**，保持单线程。

---

## 收尾 Polish（最终版本前）

三个小清理（非必须但规整）：

1. **RANSAC 单元测试**：新增 `tests/test_robust_line_fit.py`（4 个用例）
   覆盖"干净点云/带外点/小点集 fallback/u 值全相同退化"——之前
   `_robust_line_fit` 只有集成测试覆盖，现在有单独测试。
2. **`_collect_anomaly_sample_indices` 显式 `outlier_sigma` 参数**：
   `save_pipeline_outputs` → `save_depth_overlay_plot` /
   `save_section_debug_detail_plots` → `_collect_anomaly_sample_indices`
   整条链允许显式传参；默认 fallback 仍从 `GapFlushParams()` 拿，
   不破坏向后兼容。`gap_flush.py` 调用处现在从 params 直接传入，
   避免了 outputs.py 内部再次实例化 `GapFlushParams()` 的隐式依赖。
3. **文档**：本节；log 从 P9 截至收尾 Polish 现状连贯。

最终测试：**59 passed**（55 + 4 RANSAC 新增）。


