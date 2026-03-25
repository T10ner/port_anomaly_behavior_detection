from pathlib import Path
import argparse
import re
from datetime import datetime, timedelta
import pandas as pd


def parse_video_start(video_name: str) -> datetime:
    """
    解析:
    2022_11_20_10_21_09_10_28_37_r.mp4
    取前半段开始时间 2022-11-20 10:21:09
    """
    pattern = r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_"
    m = re.match(pattern, video_name)
    if not m:
        raise ValueError(f"无法解析视频开始时间: {video_name}")
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True, help="clip目录")
    parser.add_argument("--fps", type=float, default=25.0, help="视频fps")
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    track_csv = clip_dir / "processed" / "track_runs" / "track_clip01.csv"

    video_files = sorted(clip_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"找不到视频: {clip_dir}")
    video_path = video_files[0]

    out_dir = clip_dir / "processed" / "visual_tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not track_csv.exists():
        raise FileNotFoundError(f"找不到 tracking csv: {track_csv}")

    df = pd.read_csv(track_csv)
    if len(df) == 0:
        raise RuntimeError("track_clip01.csv 为空")

    video_start = parse_video_start(video_path.name)

    # 加时间
    df["time_sec"] = df["frame_idx"] / args.fps
    df["video_datetime"] = df["time_sec"].apply(lambda x: video_start + timedelta(seconds=float(x)))

    # 去掉 track_id = -1
    df = df[df["track_id"] >= 0].copy()

    # 每条视觉轨迹汇总
    summary = (
        df.groupby("track_id")
        .agg(
            frame_count=("frame_idx", "size"),
            start_frame=("frame_idx", "min"),
            end_frame=("frame_idx", "max"),
            start_time=("video_datetime", "min"),
            end_time=("video_datetime", "max"),
            mean_conf=("conf", "mean"),
            mean_w=("w", "mean"),
            mean_h=("h", "mean"),
            mean_cx=("cx", "mean"),
            mean_cy=("cy", "mean"),
            min_cx=("cx", "min"),
            max_cx=("cx", "max"),
            min_cy=("cy", "min"),
            max_cy=("cy", "max"),
        )
        .reset_index()
    )

    summary["duration_sec"] = (summary["end_frame"] - summary["start_frame"] + 1) / args.fps
    summary["move_x"] = summary["max_cx"] - summary["min_cx"]
    summary["move_y"] = summary["max_cy"] - summary["min_cy"]

    # 筛掉太短的碎轨迹
    summary_filtered = summary[
        (summary["frame_count"] >= 15) &
        (summary["duration_sec"] >= 0.5)
        ].copy()

    # 排序：先看持续时间长、框较大的
    summary_filtered = summary_filtered.sort_values(
        ["duration_sec", "mean_w"], ascending=[False, False]
    ).reset_index(drop=True)

    out_all = out_dir / "track_points_with_time.csv"
    out_summary = out_dir / "track_summary_all.csv"
    out_summary_filtered = out_dir / "track_summary_filtered.csv"

    df.to_csv(out_all, index=False, encoding="utf-8-sig")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    summary_filtered.to_csv(out_summary_filtered, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("视觉轨迹构建完成")
    print("=" * 60)
    print(f"总检测点数: {len(df)}")
    print(f"唯一 track_id 数量: {df['track_id'].nunique()}")
    print(f"视频开始时间: {video_start}")
    print()
    print("过滤后的较稳定轨迹前20条：")
    print(summary_filtered.head(20))
    print()
    print("输出文件：")
    print(out_all)
    print(out_summary)
    print(out_summary_filtered)


if __name__ == "__main__":
    main()
