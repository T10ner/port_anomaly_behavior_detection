from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd


def fmt_dt(x):
    if pd.isna(x):
        return ""
    return pd.to_datetime(x).strftime("%H:%M:%S.%f")[:-3]


def visual_motion_label(delta_cx: float) -> str:
    if delta_cx >= 80:
        return "rightward"
    elif delta_cx <= -80:
        return "leftward"
    else:
        return "quasi_static"


def ais_motion_label(delta_lon: float, delta_lat: float) -> str:
    # 长江这个场景里先粗略用经度变化判断主方向
    if delta_lon >= 0.001:
        return "eastward"
    elif delta_lon <= -0.001:
        return "westward"
    else:
        return "quasi_static"


def build_visual_summary(track_points: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for track_id, g in track_points.groupby("track_id"):
        g = g.sort_values("frame_idx").reset_index(drop=True)

        first = g.iloc[0]
        last = g.iloc[-1]

        frame_count = len(g)
        start_frame = int(first["frame_idx"])
        end_frame = int(last["frame_idx"])
        duration_sec = (end_frame - start_frame + 1) / 25.0

        start_cx = float(first["cx"])
        end_cx = float(last["cx"])
        start_cy = float(first["cy"])
        end_cy = float(last["cy"])

        delta_cx = end_cx - start_cx
        delta_cy = end_cy - start_cy

        mean_w = float(g["w"].mean())
        mean_h = float(g["h"].mean())
        mean_conf = float(g["conf"].mean())
        mean_cx = float(g["cx"].mean())
        mean_cy = float(g["cy"].mean())

        min_cx = float(g["cx"].min())
        max_cx = float(g["cx"].max())
        min_cy = float(g["cy"].min())
        max_cy = float(g["cy"].max())

        motion = visual_motion_label(delta_cx)

        # 粗略质量标记
        edge_static_noise = (
                mean_w < 40 and
                abs(delta_cx) < 20 and
                (mean_cx < 100 or mean_cx > 2460)
        )

        suggest_keep = (
                frame_count >= 30 and
                duration_sec >= 2.0 and
                not edge_static_noise
        )

        rows.append({
            "track_id": int(track_id),
            "frame_count": frame_count,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": first["video_datetime"],
            "end_time": last["video_datetime"],
            "start_time_str": fmt_dt(first["video_datetime"]),
            "end_time_str": fmt_dt(last["video_datetime"]),
            "duration_sec": round(duration_sec, 2),
            "mean_conf": round(mean_conf, 4),
            "mean_w": round(mean_w, 2),
            "mean_h": round(mean_h, 2),
            "start_cx": round(start_cx, 2),
            "end_cx": round(end_cx, 2),
            "start_cy": round(start_cy, 2),
            "end_cy": round(end_cy, 2),
            "delta_cx": round(delta_cx, 2),
            "delta_cy": round(delta_cy, 2),
            "mean_cx": round(mean_cx, 2),
            "mean_cy": round(mean_cy, 2),
            "min_cx": round(min_cx, 2),
            "max_cx": round(max_cx, 2),
            "min_cy": round(min_cy, 2),
            "max_cy": round(max_cy, 2),
            "motion_label": motion,
            "edge_static_noise": edge_static_noise,
            "suggest_keep": suggest_keep,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["suggest_keep", "duration_sec", "mean_w"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def build_ais_summary(ais_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for mmsi, g in ais_df.groupby("mmsi"):
        g = g.sort_values("datetime").reset_index(drop=True)

        first = g.iloc[0]
        last = g.iloc[-1]

        start_time = pd.to_datetime(first["datetime"])
        end_time = pd.to_datetime(last["datetime"])
        duration_sec = (end_time - start_time).total_seconds()

        start_lon = float(first["lon"])
        end_lon = float(last["lon"])
        start_lat = float(first["lat"])
        end_lat = float(last["lat"])

        delta_lon = end_lon - start_lon
        delta_lat = end_lat - start_lat

        motion = ais_motion_label(delta_lon, delta_lat)

        rows.append({
            "mmsi": int(mmsi),
            "point_count": len(g),
            "start_time": start_time,
            "end_time": end_time,
            "start_time_str": fmt_dt(start_time),
            "end_time_str": fmt_dt(end_time),
            "duration_sec": round(duration_sec, 2),
            "mean_speed": round(float(g["speed"].mean()), 3),
            "max_speed": round(float(g["speed"].max()), 3),
            "start_lon": round(start_lon, 6),
            "end_lon": round(end_lon, 6),
            "start_lat": round(start_lat, 6),
            "end_lat": round(end_lat, 6),
            "delta_lon": round(delta_lon, 6),
            "delta_lat": round(delta_lat, 6),
            "motion_label": motion,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["duration_sec", "mean_speed"], ascending=[False, False]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True)
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)

    visual_csv = clip_dir / "processed" / "visual_tracks" / "track_points_with_time.csv"
    ais_csv = clip_dir / "processed" / "merged_ais_video_window.csv"

    if not visual_csv.exists():
        raise FileNotFoundError(f"找不到视觉轨迹点文件: {visual_csv}")
    if not ais_csv.exists():
        raise FileNotFoundError(f"找不到 AIS 文件: {ais_csv}")

    visual_df = pd.read_csv(visual_csv, parse_dates=["video_datetime"])
    ais_df = pd.read_csv(ais_csv, parse_dates=["datetime"])

    visual_summary = build_visual_summary(visual_df)
    ais_summary = build_ais_summary(ais_df)

    out_dir = clip_dir / "processed" / "matching_prep"
    out_dir.mkdir(parents=True, exist_ok=True)

    visual_out = out_dir / "visual_track_summary_v2.csv"
    ais_out = out_dir / "ais_track_summary.csv"

    visual_summary.to_csv(visual_out, index=False, encoding="utf-8-sig")
    ais_summary.to_csv(ais_out, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("匹配前摘要表已生成")
    print("=" * 60)

    print("\n[视觉轨迹 Top 15]")
    print(
        visual_summary[
            ["track_id", "suggest_keep", "frame_count", "duration_sec",
             "start_time_str", "end_time_str", "mean_w", "start_cx", "end_cx",
             "delta_cx", "motion_label", "edge_static_noise"]
        ].head(15)
    )

    print("\n[AIS 轨迹 Top 15]")
    print(
        ais_summary[
            ["mmsi", "point_count", "duration_sec", "start_time_str", "end_time_str",
             "mean_speed", "delta_lon", "delta_lat", "motion_label"]
        ].head(15)
    )

    print("\n输出文件：")
    print(visual_out)
    print(ais_out)


if __name__ == "__main__":
    main()
