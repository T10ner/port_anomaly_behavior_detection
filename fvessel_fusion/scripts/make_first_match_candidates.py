from pathlib import Path
import argparse
import re
import pandas as pd


def parse_camera_lon(camera_file: Path) -> float:
    text = camera_file.read_text(encoding="utf-8", errors="ignore")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) < 1:
        raise ValueError(f"无法从 {camera_file} 解析相机经度")
    return float(nums[0])


def to_ts(x):
    return pd.Timestamp(x)


def overlap_seconds(a_start, a_end, b_start, b_end):
    start = max(to_ts(a_start), to_ts(b_start))
    end = min(to_ts(a_end), to_ts(b_end))
    return max(0.0, (end - start).total_seconds())


def classify_visual_motion(delta_cx: float) -> str:
    if delta_cx >= 80:
        return "rightward"
    elif delta_cx <= -80:
        return "leftward"
    else:
        return "quasi_static"


def classify_ais_motion(delta_lon: float, mean_speed: float) -> str:
    if delta_lon >= 0.001 or mean_speed >= 1.0 and delta_lon > 0:
        return "eastward"
    elif delta_lon <= -0.001 or mean_speed >= 1.0 and delta_lon < 0:
        return "westward"
    else:
        return "quasi_static"


def classify_visual_zone(mean_cx: float, video_width: int = 2560) -> str:
    left_th = video_width / 3
    right_th = 2 * video_width / 3
    if mean_cx < left_th:
        return "left"
    elif mean_cx > right_th:
        return "right"
    else:
        return "center"


def classify_ais_zone(mean_lon: float, camera_lon: float, lon_thresh: float = 0.0045) -> str:
    diff = mean_lon - camera_lon
    if diff <= -lon_thresh:
        return "left"
    elif diff >= lon_thresh:
        return "right"
    else:
        return "center"


def motion_match_score(v_motion: str, a_motion: str) -> float:
    if v_motion == "quasi_static" and a_motion == "quasi_static":
        return 1.0
    if v_motion == "leftward" and a_motion == "westward":
        return 1.0
    if v_motion == "rightward" and a_motion == "eastward":
        return 1.0
    if v_motion == "quasi_static" or a_motion == "quasi_static":
        return 0.35
    return 0.0


def zone_match_score(v_zone: str, a_zone: str) -> float:
    if v_zone == a_zone:
        return 1.0

    # 相邻区域给半分
    adjacent = {
        ("left", "center"), ("center", "left"),
        ("center", "right"), ("right", "center")
    }
    if (v_zone, a_zone) in adjacent:
        return 0.5

    return 0.0


def distance_score(min_distance_m: float) -> float:
    if pd.isna(min_distance_m):
        return 0.0
    return max(0.0, 1.0 - min_distance_m / 2200.0)


def track_reliability_score(frame_count: int, duration_sec: float, mean_w: float, delta_cx: float) -> float:
    score = 1.0

    # 对短碎轨迹降权
    if duration_sec < 5:
        score *= 0.30
    elif duration_sec < 10:
        score *= 0.45
    elif duration_sec < 20:
        score *= 0.65
    elif duration_sec < 40:
        score *= 0.85

    if frame_count < 30:
        score *= 0.60
    elif frame_count < 60:
        score *= 0.80

    # 很小且几乎不动的轨迹进一步降权
    if mean_w < 35 and abs(delta_cx) < 30:
        score *= 0.70

    return max(0.05, min(score, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--video_width", type=int, default=2560)
    parser.add_argument("--lon_thresh", type=float, default=0.0045)
    parser.add_argument("--max_distance_m", type=float, default=2200.0)
    parser.add_argument(
        "--blacklist_mmsi",
        type=int,
        nargs="*",
        default=[110000000],
        help="要排除的 AIS MMSI，默认先排除 110000000"
    )
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)

    visual_csv = clip_dir / "processed" / "matching_prep" / "visual_track_summary_v2.csv"
    ais_csv = clip_dir / "processed" / "matching_prep" / "ais_track_summary.csv"
    closest_csv = clip_dir / "processed" / "closest_mmsi_summary.csv"
    camera_file = clip_dir / "camera_para.txt"

    if not visual_csv.exists():
        raise FileNotFoundError(f"找不到: {visual_csv}")
    if not ais_csv.exists():
        raise FileNotFoundError(f"找不到: {ais_csv}")
    if not closest_csv.exists():
        raise FileNotFoundError(f"找不到: {closest_csv}")
    if not camera_file.exists():
        raise FileNotFoundError(f"找不到: {camera_file}")

    camera_lon = parse_camera_lon(camera_file)

    visual_df = pd.read_csv(visual_csv, parse_dates=["start_time", "end_time"])
    ais_df = pd.read_csv(ais_csv, parse_dates=["start_time", "end_time"])
    closest_df = pd.read_csv(closest_csv)

    # 合并距离先验
    ais_df = ais_df.merge(
        closest_df[["mmsi", "min_distance_m", "mean_distance_m"]],
        on="mmsi",
        how="left"
    )

    # 只保留建议保留的视觉轨迹
    visual_df = visual_df[visual_df["suggest_keep"] == True].copy()

    # 排除黑名单 MMSI
    if args.blacklist_mmsi:
        ais_df = ais_df[~ais_df["mmsi"].isin(args.blacklist_mmsi)].copy()

    # 保留距离相机较近的 AIS
    ais_df = ais_df[ais_df["min_distance_m"] <= args.max_distance_m].copy()

    # 视觉侧标签
    visual_df["motion_label_v"] = visual_df["delta_cx"].apply(classify_visual_motion)
    visual_df["zone_label_v"] = visual_df["mean_cx"].apply(
        lambda x: classify_visual_zone(x, video_width=args.video_width)
    )
    visual_df["track_reliability"] = visual_df.apply(
        lambda r: track_reliability_score(
            frame_count=int(r["frame_count"]),
            duration_sec=float(r["duration_sec"]),
            mean_w=float(r["mean_w"]),
            delta_cx=float(r["delta_cx"]),
        ),
        axis=1
    )

    # AIS 侧标签
    ais_df["mean_lon"] = (ais_df["start_lon"] + ais_df["end_lon"]) / 2.0
    ais_df["motion_label_a"] = ais_df.apply(
        lambda r: classify_ais_motion(float(r["delta_lon"]), float(r["mean_speed"])),
        axis=1
    )
    ais_df["zone_label_a"] = ais_df["mean_lon"].apply(
        lambda x: classify_ais_zone(x, camera_lon=camera_lon, lon_thresh=args.lon_thresh)
    )

    candidate_rows = []

    for _, v in visual_df.iterrows():
        for _, a in ais_df.iterrows():
            ov_sec = overlap_seconds(v["start_time"], v["end_time"], a["start_time"], a["end_time"])
            if ov_sec <= 0:
                continue

            ov_ratio = ov_sec / max(1e-6, min(float(v["duration_sec"]), float(a["duration_sec"])))

            motion_sc = motion_match_score(v["motion_label_v"], a["motion_label_a"])
            zone_sc = zone_match_score(v["zone_label_v"], a["zone_label_a"])
            dist_sc = distance_score(float(a["min_distance_m"]))
            rel_sc = float(v["track_reliability"])

            # base score
            base_score = (
                    0.45 * ov_ratio +
                    0.20 * motion_sc +
                    0.20 * zone_sc +
                    0.15 * dist_sc
            )

            # 轨迹可靠性作为乘子
            final_score = base_score * rel_sc

            candidate_rows.append({
                "track_id": int(v["track_id"]),
                "track_duration_sec": float(v["duration_sec"]),
                "track_frame_count": int(v["frame_count"]),
                "track_mean_w": float(v["mean_w"]),
                "track_start": v["start_time"],
                "track_end": v["end_time"],
                "track_motion": v["motion_label_v"],
                "track_zone": v["zone_label_v"],
                "track_reliability": round(rel_sc, 4),

                "mmsi": int(a["mmsi"]),
                "ais_duration_sec": float(a["duration_sec"]),
                "ais_mean_speed": float(a["mean_speed"]),
                "ais_motion": a["motion_label_a"],
                "ais_zone": a["zone_label_a"],
                "ais_min_distance_m": float(a["min_distance_m"]),
                "ais_mean_distance_m": float(a["mean_distance_m"]),

                "overlap_sec": round(float(ov_sec), 2),
                "overlap_ratio": round(float(ov_ratio), 4),
                "motion_score": round(float(motion_sc), 4),
                "zone_score": round(float(zone_sc), 4),
                "distance_score": round(float(dist_sc), 4),
                "base_score": round(float(base_score), 4),
                "match_score": round(float(final_score), 4),
            })

    cand_df = pd.DataFrame(candidate_rows)
    if len(cand_df) == 0:
        raise RuntimeError("没有生成任何候选匹配，请检查输入文件")

    cand_df = cand_df.sort_values(
        ["track_id", "match_score", "overlap_sec"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    topk_df = cand_df.groupby("track_id").head(args.topk).copy()

    out_dir = clip_dir / "processed" / "matching_results_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_all = out_dir / "all_match_candidates_v2.csv"
    out_topk = out_dir / "topk_match_candidates_v2.csv"

    cand_df.to_csv(out_all, index=False, encoding="utf-8-sig")
    topk_df.to_csv(out_topk, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("第一版候选匹配 v2 完成")
    print("=" * 60)
    print(f"相机经度: {camera_lon}")
    print(f"排除 MMSI: {args.blacklist_mmsi}")
    print(f"最大距离阈值: {args.max_distance_m} m")
    print()

    print("Top-K 候选（前 30 行）:")
    print(
        topk_df[
            [
                "track_id", "track_motion", "track_zone", "track_duration_sec", "track_mean_w",
                "mmsi", "ais_motion", "ais_zone", "ais_mean_speed", "ais_min_distance_m",
                "overlap_sec", "overlap_ratio", "motion_score", "zone_score",
                "distance_score", "track_reliability", "match_score"
            ]
        ].head(30)
    )

    print("\n输出文件：")
    print(out_all)
    print(out_topk)


if __name__ == "__main__":
    main()
