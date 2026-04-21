"""方案 B gap / flush 定义可视化（双平面 3D 参考系）.

4 联图逐层展开测量几何：
(a) 深度图全局 + 当前 section 位置；
(b) u-z 2D 旧视角（对照：展示 legacy mixed-unit gap 是怎么出来的）；
(c) 世界 3D 透视：左右两张拟合平面（半透明网格）+ 两个边缘点 +
    平均法向 + 把边缘差向量**正交分解**成 gap（沿切平面）与 flush（沿法向）；
(d) 投影到平均法平面后的侧视：把 gap 和 flush 画成直角三角形，
    勾股斜边就是两边缘点的 3D 距离。

每个面板里都标注实际数值，读者可以做 sqrt(gap² + flush²) == full_3d
的自检。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402 - registers 3d projection

from pipeline.seam_mapping.inference import build_depth_image_from_point_map, predict_mask_from_point_map  # noqa: E402
from pipeline.seam_mapping.io import load_point_map  # noqa: E402
from pipeline.seam_measurement.core import compute_gap_flush  # noqa: E402
from pipeline.seam_measurement.params import GapFlushParams  # noqa: E402

from _style import PALETTE, THESIS_FIGURES_DIR, savefig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="方案 B 3D gap / flush 定义可视化")
    parser.add_argument("--pcd", type=Path,
                        default=PROJECT_ROOT / "data" / "process" / "manual_crop" / "1" / "crop.pcd")
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / "outputs" / "model" / "03301416_loocv_attention_unet"
                        / "holdout_1" / "checkpoints" / "best.pth")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--section-index", type=int, default=-1,
                        help="Which section to highlight; -1 picks middle.")
    parser.add_argument("--out", type=Path,
                        default=THESIS_FIGURES_DIR / "ch4_fig_gap_3d_definition.png")
    return parser.parse_args()


def _plane_corners(centroid: np.ndarray, normal: np.ndarray, half_size: float) -> np.ndarray:
    """Return four corner xyz for a square patch of the plane centred on
    ``centroid`` with half-side ``half_size`` mm. Uses Gram-Schmidt to
    build an in-plane basis."""
    # Pick a helper vector not parallel to the normal
    helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(helper, normal))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_a = helper - np.dot(helper, normal) * normal
    tangent_a = tangent_a / max(1e-9, float(np.linalg.norm(tangent_a)))
    tangent_b = np.cross(normal, tangent_a)
    tangent_b = tangent_b / max(1e-9, float(np.linalg.norm(tangent_b)))
    corners = np.stack(
        [
            centroid + half_size * (+tangent_a + tangent_b),
            centroid + half_size * (+tangent_a - tangent_b),
            centroid + half_size * (-tangent_a - tangent_b),
            centroid + half_size * (-tangent_a + tangent_b),
        ],
        axis=0,
    ).astype(np.float32)
    return corners


def main() -> None:
    args = parse_args()
    point_map = load_point_map(args.pcd)
    depth_image, _ = build_depth_image_from_point_map(point_map)
    prediction = predict_mask_from_point_map(
        point_map=point_map,
        checkpoint_path=args.checkpoint,
        threshold=float(args.threshold),
    )
    pred_mask = np.asarray(prediction["pred_mask"], dtype=np.uint8)
    params = GapFlushParams()
    result = compute_gap_flush(pred_mask > 0, point_map, params=params)
    section_results = result["section_results"]
    valid_items = [(i, item) for i, item in enumerate(section_results)
                   if bool(item.get("measurement", {}).get("valid", False))]
    if not valid_items:
        raise RuntimeError("No valid measurements in pipeline output.")
    pick = len(valid_items) // 2 if args.section_index < 0 else min(
        int(args.section_index), len(valid_items) - 1
    )
    item = valid_items[pick][1]
    top_surface = item["top_surface"]
    measurement = item["measurement"]

    gap_new = float(measurement["gap"])
    flush_new = float(measurement["flush"])
    full_3d = float(measurement["full_3d_mm"])
    gap_legacy = float(measurement["gap_legacy_mixed"])
    flush_legacy = float(measurement["flush_legacy_mm"])
    left_xyz = np.asarray([measurement["left_xyz"][k] for k in ("x", "y", "z")], dtype=np.float32)
    right_xyz = np.asarray([measurement["right_xyz"][k] for k in ("x", "y", "z")], dtype=np.float32)
    left_plane_payload = measurement["left_plane"]
    right_plane_payload = measurement["right_plane"]
    left_plane_centroid = np.asarray(left_plane_payload["centroid"], dtype=np.float32)
    left_plane_normal = np.asarray(left_plane_payload["normal"], dtype=np.float32)
    right_plane_centroid = np.asarray(right_plane_payload["centroid"], dtype=np.float32)
    right_plane_normal = np.asarray(right_plane_payload["normal"], dtype=np.float32)
    n_avg = np.asarray(measurement["reference_normal"], dtype=np.float32)
    left_pt = measurement["left_point"]   # (u=px, z=mm)
    right_pt = measurement["right_point"]

    fig = plt.figure(figsize=(16.0, 11.5))
    ax_a = fig.add_subplot(2, 2, 1)
    ax_b = fig.add_subplot(2, 2, 2)
    ax_c = fig.add_subplot(2, 2, 3, projection="3d")
    ax_d = fig.add_subplot(2, 2, 4)

    # ----- (a) depth image + section position -----
    ax_a.imshow(depth_image, cmap="gray")
    for it in section_results:
        for side, colour in (("left_edge", PALETTE["accent"]),
                             ("right_edge", PALETTE["secondary"])):
            edge = it.get("top_surface", {}).get(side, {})
            pix = edge.get("pixels_xy") if edge else None
            if pix is None or len(pix) == 0:
                continue
            ax_a.plot(pix[0][0], pix[0][1], ".", color=colour, markersize=1.3, alpha=0.55)
    left_pix = top_surface["left_edge"]["pixels_xy"][0]
    right_pix = top_surface["right_edge"]["pixels_xy"][0]
    ax_a.plot(left_pix[0], left_pix[1], "o", color=PALETTE["accent"], markersize=10,
              markeredgecolor="white", markeredgewidth=1.4, label="左边缘 (选中)")
    ax_a.plot(right_pix[0], right_pix[1], "o", color=PALETTE["secondary"], markersize=10,
              markeredgecolor="white", markeredgewidth=1.4, label="右边缘 (选中)")
    ax_a.annotate("", xy=right_pix, xytext=left_pix,
                  arrowprops=dict(arrowstyle="<->", color="#0ea5e9", lw=1.8))
    ax_a.set_title("(a) 深度图 + 全截面顶面边缘 (绿/红)，青箭头 = 当前 section")
    ax_a.set_xlabel("列 / 像素")
    ax_a.set_ylabel("行 / 像素")
    ax_a.legend(loc="lower right", fontsize=9, framealpha=0.85)

    # ----- (b) u-z 2D legacy view -----
    filtered = item.get("filtered_points", {})
    if len(filtered.get("u", [])):
        ax_b.scatter(filtered["u"], filtered["z"], s=6, c=PALETTE["muted"], alpha=0.5, label="截面有效点")
    for label, colour, seg_key in (("左顶面选段", PALETTE["accent"], "left_segment"),
                                    ("右顶面选段", PALETTE["secondary"], "right_segment")):
        seg = top_surface.get(seg_key, {})
        if len(seg.get("u", [])):
            ax_b.scatter(seg["u"], seg["z"], s=14, c=colour, alpha=0.9, label=label)
    ax_b.scatter([left_pt["u"]], [left_pt["z"]], s=120, marker="*", c=PALETTE["accent"],
                 edgecolors="white", linewidths=0.9, zorder=6)
    ax_b.scatter([right_pt["u"]], [right_pt["z"]], s=120, marker="*", c=PALETTE["secondary"],
                 edgecolors="white", linewidths=0.9, zorder=6)
    y_arrow = max(left_pt["z"], right_pt["z"]) + 0.06
    ax_b.annotate("", xy=(right_pt["u"], y_arrow), xytext=(left_pt["u"], y_arrow),
                  arrowprops=dict(arrowstyle="<->", color="#9ca3af", lw=1.5))
    ax_b.text(0.5 * (left_pt["u"] + right_pt["u"]), y_arrow + 0.035,
              f"gap_legacy = {gap_legacy:.2f}  (混合单位, 不采用)",
              ha="center", fontsize=10, color="#6b7280")
    ax_b.annotate("", xy=(right_pt["u"], right_pt["z"]), xytext=(right_pt["u"], left_pt["z"]),
                  arrowprops=dict(arrowstyle="<->", color="#9ca3af", lw=1.2))
    ax_b.text(right_pt["u"] + 0.5, 0.5 * (left_pt["z"] + right_pt["z"]),
              f"flush_legacy = {flush_legacy:.3f} mm",
              fontsize=10, color="#6b7280")
    ax_b.set_title("(b) 截面 u-z 2D (legacy 旧视角) — 仅作对照")
    ax_b.set_xlabel("u / px (局部)")
    ax_b.set_ylabel("z / mm")
    ax_b.legend(loc="lower right", fontsize=9, framealpha=0.85)

    # ----- (c) 3D perspective with two fitted planes + decomposed vector -----
    # Collect surface points for context scatter
    def _points_from(key):
        pts = top_surface.get(key) or {}
        xyz = pts.get("xyz")
        if xyz is None:
            return np.empty((0, 3), dtype=np.float32)
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        finite = np.all(np.isfinite(arr), axis=1)
        return arr[finite]

    left_pts = _points_from("left_segment")
    if len(left_pts) < 6:
        left_pts = _points_from("left_candidates")
    right_pts = _points_from("right_segment")
    if len(right_pts) < 6:
        right_pts = _points_from("right_candidates")
    all_xyz = np.concatenate([left_pts, right_pts], axis=0) if len(left_pts) + len(right_pts) else None

    if all_xyz is not None and len(all_xyz):
        ax_c.scatter(left_pts[:, 0], left_pts[:, 1], left_pts[:, 2], s=12,
                     c=PALETTE["accent"], alpha=0.7, depthshade=False, label="左顶面采样点")
        ax_c.scatter(right_pts[:, 0], right_pts[:, 1], right_pts[:, 2], s=12,
                     c=PALETTE["secondary"], alpha=0.7, depthshade=False, label="右顶面采样点")
    ax_c.scatter([left_xyz[0]], [left_xyz[1]], [left_xyz[2]], s=110,
                 marker="*", c=PALETTE["accent"], depthshade=False, edgecolors="white", linewidths=1.0)
    ax_c.scatter([right_xyz[0]], [right_xyz[1]], [right_xyz[2]], s=110,
                 marker="*", c=PALETTE["secondary"], depthshade=False, edgecolors="white", linewidths=1.0)

    plane_half = max(3.0, 0.8 * full_3d)
    for centroid, normal, colour in (
        (left_plane_centroid, left_plane_normal, PALETTE["accent"]),
        (right_plane_centroid, right_plane_normal, PALETTE["secondary"]),
    ):
        corners = _plane_corners(centroid, normal, half_size=plane_half)
        xs = np.append(corners[:, 0], corners[0, 0])
        ys = np.append(corners[:, 1], corners[0, 1])
        zs = np.append(corners[:, 2], corners[0, 2])
        ax_c.plot_trisurf(xs[:-1], ys[:-1], zs[:-1], alpha=0.18, color=colour)
        ax_c.plot(xs, ys, zs, color=colour, lw=1.1, alpha=0.75)

    # Three-way decomposition: gap (cross-seam) + along-seam residual + flush (normal).
    parallel_signed = float(measurement.get("parallel_component_signed", 0.0))
    delta_vec = right_xyz - left_xyz
    t_along = np.asarray(measurement.get("seam_tangent_along_3d", [np.nan] * 3), dtype=np.float32)
    t_cross = np.asarray(measurement.get("seam_tangent_cross_3d", [np.nan] * 3), dtype=np.float32)
    along_comp = float(np.dot(delta_vec, t_along)) if np.all(np.isfinite(t_along)) else 0.0
    cross_comp = float(np.dot(delta_vec, t_cross)) if np.all(np.isfinite(t_cross)) else 0.0
    flush_signed = parallel_signed
    gap_along_res = float(measurement.get("gap_along_residual_mm", float("nan")))

    # Chain the three legs from left_xyz:
    #   left -> along -> cross -> flush == right
    p_after_along = left_xyz + along_comp * (t_along if np.all(np.isfinite(t_along)) else np.zeros(3, dtype=np.float32))
    p_after_cross = p_after_along + cross_comp * (t_cross if np.all(np.isfinite(t_cross)) else np.zeros(3, dtype=np.float32))
    # p_after_cross + flush_signed * n_avg should equal right_xyz

    if np.all(np.isfinite(t_along)):
        ax_c.plot([left_xyz[0], p_after_along[0]], [left_xyz[1], p_after_along[1]],
                  [left_xyz[2], p_after_along[2]], color="#f59e0b", lw=2.0,
                  label=f"along-seam 残差 = {gap_along_res:.3f} mm")
    ax_c.plot([p_after_along[0], p_after_cross[0]], [p_after_along[1], p_after_cross[1]],
              [p_after_along[2], p_after_cross[2]], color="#0ea5e9", lw=2.4,
              label=f"gap (cross-seam) = {gap_new:.3f} mm")
    ax_c.plot([p_after_cross[0], right_xyz[0]], [p_after_cross[1], right_xyz[1]],
              [p_after_cross[2], right_xyz[2]], color=PALETTE["purple"], lw=2.4,
              label=f"flush (normal) = {flush_new:.3f} mm")
    ax_c.plot([left_xyz[0], right_xyz[0]], [left_xyz[1], right_xyz[1]],
              [left_xyz[2], right_xyz[2]], color="#64748b", lw=1.3, ls="--",
              label=f"full 3D = {full_3d:.3f} mm  (勾股斜边)")
    # Explicit n_avg arrow from the mid-plane origin so the reader sees
    # which way the reference normal points in 3D. Scaled to be visible
    # next to the gap leg.
    n_arrow_scale = max(1.5, 0.3 * full_3d)
    n_origin = 0.5 * (left_plane_centroid + right_plane_centroid)
    ax_c.quiver(
        n_origin[0], n_origin[1], n_origin[2],
        n_avg[0], n_avg[1], n_avg[2],
        length=n_arrow_scale, color="#0f172a", arrow_length_ratio=0.2, lw=2.0,
        label=f"n_avg = ({n_avg[0]:+.3f}, {n_avg[1]:+.3f}, {n_avg[2]:+.3f})",
    )

    tilt_deg = float(np.degrees(np.arccos(np.clip(float(n_avg[2]), -1.0, 1.0))))
    ax_c.set_title(
        f"(c) 3D 双平面参考系   n_avg 倾角 = {tilt_deg:.2f}° (相对 +z)"
    )
    ax_c.set_xlabel("x / mm")
    ax_c.set_ylabel("y / mm")
    ax_c.set_zlabel("z / mm")
    ax_c.legend(loc="upper left", fontsize=7, framealpha=0.9)
    # Classical oblique 3D view so the +z axis (and any flush component
    # along it) remains clearly vertical in the rendered image, avoiding
    # the degenerate top-down projection where ``flush`` collapses to a
    # stub and visually masquerades as an xy-plane vector.
    ax_c.view_init(elev=18, azim=-55)
    # Equalise axis scales so the geometric proportions are faithful and
    # the reader can compare flush (mm, along z) with gap (mm, in xy)
    # without matplotlib auto-stretching one axis.
    all_points = np.vstack([left_pts, right_pts,
                            left_xyz[None, :], right_xyz[None, :],
                            n_origin[None, :]])
    pt_min = all_points.min(axis=0)
    pt_max = all_points.max(axis=0)
    span = float(max(pt_max - pt_min).max())
    mids = 0.5 * (pt_max + pt_min)
    half = max(span * 0.6, n_arrow_scale * 1.1)
    ax_c.set_xlim(mids[0] - half, mids[0] + half)
    ax_c.set_ylim(mids[1] - half, mids[1] + half)
    ax_c.set_zlim(mids[2] - half, mids[2] + half)

    # ----- (d) 2D orthographic view: cross-seam vs normal, absolute right-angle truth -----
    ax_d.set_aspect("equal", adjustable="datalim")
    # Right-angle indicator (small square at the corner)
    corner = 0.04 * max(gap_new, flush_new, 1.0)
    ax_d.plot([gap_new - corner, gap_new - corner, gap_new], [0, corner, corner],
              color="#0f172a", lw=1.0)
    ax_d.plot([0, gap_new], [0, 0], color="#0ea5e9", lw=2.6,
              label=f"gap (cross-seam) = {gap_new:.3f} mm")
    ax_d.plot([gap_new, gap_new], [0, flush_new], color=PALETTE["purple"], lw=2.6,
              label=f"flush (normal) = {flush_new:.3f} mm")
    ax_d.plot([0, gap_new], [0, flush_new], color="#64748b", lw=1.6, ls="--",
              label=f"full 3D (hypot) = {full_3d:.3f} mm")
    ax_d.scatter([0], [0], s=160, marker="*", c=PALETTE["accent"],
                 edgecolors="white", linewidths=0.9, zorder=5)
    ax_d.scatter([gap_new], [flush_new], s=160, marker="*", c=PALETTE["secondary"],
                 edgecolors="white", linewidths=0.9, zorder=5)
    ax_d.text(0.0, -0.08 * max(flush_new, 0.2), "左边缘", ha="center", va="top",
              color=PALETTE["accent"], fontsize=10)
    ax_d.text(gap_new, flush_new + 0.08 * max(flush_new, 0.2), "右边缘", ha="center", va="bottom",
              color=PALETTE["secondary"], fontsize=10)
    ax_d.set_xlabel("沿 cross-seam / mm  (gap 方向)")
    ax_d.set_ylabel("沿法向 / mm  (flush 方向)")
    ax_d.axhline(0.0, color="#94a3b8", linewidth=0.6, linestyle=":", alpha=0.6)
    ax_d.axvline(0.0, color="#94a3b8", linewidth=0.6, linestyle=":", alpha=0.6)
    check = (gap_new ** 2 + flush_new ** 2 + (gap_along_res if np.isfinite(gap_along_res) else 0.0) ** 2) ** 0.5
    along_txt = f"   (along-seam 残差 = {gap_along_res:.3f} mm 已扣除)" if np.isfinite(gap_along_res) else ""
    ax_d.set_title(
        f"(d) 正交 2D 视图  √(gap²+flush²+along²) = {check:.3f} ≈ full 3D = {full_3d:.3f} mm{along_txt}"
    )
    ax_d.grid(True, alpha=0.25)
    ax_d.legend(loc="upper left", fontsize=9, framealpha=0.9)

    skin_tilt_deg = float(np.degrees(np.arccos(np.clip(float(n_avg[2]), -1.0, 1.0))))
    fig.suptitle(
        f"方案 B: 双平面 3D gap/flush  |  section #{item['sample_index']}  |  "
        f"gap = {gap_new:.3f} mm   flush = {flush_new:.3f} mm   "
        f"full_3d = {full_3d:.3f} mm   "
        f"skin 倾角 = {skin_tilt_deg:.2f}°  (本样本近似水平，故 gap 方向 ≈ xy)",
        y=0.995, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
