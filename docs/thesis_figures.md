# 论文图表生成指南

本文档描述如何从本仓库生成论文正文中的数据图（B 类代码图），以及哪些图需要先手工采集数据 / 拍照 / 截屏。

所有代码图：
- 由 `scripts/draw/` 和 `scripts/eval/param_sensitivity.py` 生成
- 保存在 `outputs/thesis_figures/`（该目录被 .gitignore 忽略，不入库）
- 统一样式：300 DPI PNG，宽度 ≥ 1200 px，SimSun/Microsoft YaHei 中文 + Times New Roman 英文/数字，轴标签 ≥ 10 pt，黑白可辨的色彩+线型+marker 组合
- 命名规范：`chN_figM_<desc>.png`

---

## 可无需额外数据直接生成的图（14 张）

| 图 | 文件名 | 脚本 | 数据来源 |
|---|---|---|---|
| 图 3.1 | `ch3_fig1_depth_image_demo.png` | `scripts/draw/ch3_fig1_depth_image_demo.py` | `data/process/manual_crop/1/crop.pcd` |
| 图 3.5 | `ch3_fig5_augmentation_demo.png` | `scripts/draw/ch3_fig5_augmentation_demo.py` | `data/real_train/{images,masks,valids}/1.png` |
| 图 3.6 | `ch3_fig6_patch_sampling.png` | `scripts/draw/ch3_fig6_patch_sampling.py` | 同上 |
| 图 3.7 / 6.4 | `ch3_fig7_train_curves.png` | `scripts/draw/ch3_fig7_train_curves.py` | `outputs/model/03301416_loocv_attention_unet/holdout_*/metrics/history.json` |
| 图 3.8 | `ch3_fig8_seg_triplet.png` | `scripts/draw/ch3_fig8_seg_triplet.py` | 数据+ holdout_1 best.pth |
| 图 4.2 | `ch4_fig2_pca_direction.png` | `scripts/draw/ch4_fig2_pca_direction.py` | PCD + checkpoint |
| 图 4.3 | `ch4_fig3_section_sampling.png` | `scripts/draw/ch4_fig3_section_sampling.py` | PCD + checkpoint |
| 图 4.4 | `ch4_fig4_top_surface_stages.png` | `scripts/draw/ch4_fig4_top_surface_stages.py` | PCD + checkpoint |
| 图 4.5 | `ch4_fig5_single_section_edges.png` | `scripts/draw/ch4_fig5_single_section_edges.py` | PCD + checkpoint |
| 图 4.8 | `ch4_fig8_gap_flush_profile.png` | `scripts/draw/ch4_fig8_gap_flush_profile.py --csv <run>/section_profile.csv` | 一次 pipeline 运行产出的 csv |
| 图 6.3 | `ch6_fig3_train_samples_grid.png` | `scripts/draw/ch6_fig3_train_samples_grid.py` | 5 张 `data/real_train/*/*.png` |
| 图 6.5 | `ch6_fig5_model_compare_bars.png` | `scripts/draw/ch6_fig5_model_compare_bars.py` | `outputs/model/{03312131_loocv_unet, 03301416_loocv_attention_unet}/loocv_summary.json` |
| 图 6.6 | `ch6_fig6_hard_cases.png` | `scripts/draw/ch6_fig6_hard_cases.py` | LOOCV run 里的 `holdout_*/previews/best_holdout.png` + `metrics/best_summary.json` |
| 图 6.10 | `ch6_fig10_param_sensitivity_<param>.png` | `scripts/eval/param_sensitivity.py --plot` | 动态生成（跑 5 次 pipeline） |
| 图 6.11 | `ch6_fig11_timing_breakdown.png` | `scripts/draw/ch6_fig11_timing_breakdown.py` | 动态生成（warmup + 3 次 pipeline） |

### 一键生成全部代码图

```bash
PCD="data/process/manual_crop/1/crop.pcd"
CKPT="outputs/model/03301416_loocv_attention_unet/holdout_1/checkpoints/best.pth"

# 先确保 outputs/pipeline/1_crop/section_profile.csv 存在（供图 4.8 使用）
python -m src.pipeline --pcd-path "$PCD" --checkpoint-path "$CKPT" --seam-step 2

# 静态图
python scripts/draw/ch3_fig1_depth_image_demo.py
python scripts/draw/ch3_fig5_augmentation_demo.py
python scripts/draw/ch3_fig6_patch_sampling.py
python scripts/draw/ch3_fig7_train_curves.py
python scripts/draw/ch3_fig8_seg_triplet.py
python scripts/draw/ch4_fig2_pca_direction.py
python scripts/draw/ch4_fig3_section_sampling.py
python scripts/draw/ch4_fig4_top_surface_stages.py
python scripts/draw/ch4_fig5_single_section_edges.py
python scripts/draw/ch4_fig8_gap_flush_profile.py --csv outputs/pipeline/1_crop/section_profile.csv
python scripts/draw/ch6_fig3_train_samples_grid.py
python scripts/draw/ch6_fig5_model_compare_bars.py
python scripts/draw/ch6_fig6_hard_cases.py

# 参数敏感性（图 6.10）
python scripts/eval/param_sensitivity.py --pcd "$PCD" --checkpoint "$CKPT" --param seam_step --values 1 2 4 8 16 --plot
python scripts/eval/param_sensitivity.py --pcd "$PCD" --checkpoint "$CKPT" --param top_surface_quantile --values 0.70 0.80 0.85 0.90 0.95 --plot

# 耗时分解（图 6.11）
python scripts/draw/ch6_fig11_timing_breakdown.py --pcd "$PCD" --checkpoint "$CKPT" --warmups 1 --repeats 3
```

## 需要先采集"算法 vs 人工测量"成对数据的图（3 张）

| 图 | 文件名 | 脚本 | 所需 CSV 列 |
|---|---|---|---|
| 图 6.7 | `ch6_fig7_8_error_histograms.png`（gap+flush 二联） | `scripts/draw/ch6_fig7_8_error_histograms.py --csv <path>` | `algo_gap, manual_gap, algo_flush, manual_flush` |
| 图 6.8 | 同上 | 同上 | 同上 |
| 图 6.9 | `ch6_fig9_scatter_bland_altman.png` | `scripts/draw/ch6_fig9_scatter_bland_altman.py --csv <path>` | 同上 |

CSV 模板：[`docs/algo_vs_manual_template.csv`](algo_vs_manual_template.csv)。建议每个样本至少 10-20 个成对测点，算法侧可从 `outputs/pipeline/<run>/section_profile.csv` 按 `sample_index` 挑出代表性 section；人工侧用卡尺 / 塞尺在同位置测量。

## 必须手工采集的内容（非代码产出）

| 图 | 来源 |
|---|---|
| 图 5.2 - 5.5 | 软件运行时界面截图（上位机） |
| 图 6.1 | 实验平台照片 |
| 图 6.2 | 蒙皮样件照片（每件附真值标注） |
