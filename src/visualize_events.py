from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACK_FILE = PROJECT_ROOT / "data" / "processed" / "ais_valid_tracks.csv"
POINT_FILE = PROJECT_ROOT / "data" / "processed" / "anomaly_point_flags.csv"
EVENT_FILE = PROJECT_ROOT / "data" / "processed" / "anomaly_events.csv"

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_data():
    df_track = pd.read_csv(TRACK_FILE)
    df_point = pd.read_csv(POINT_FILE)
    df_event = pd.read_csv(EVENT_FILE)

    for df in [df_track, df_point]:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df_event["start_time"] = pd.to_datetime(df_event["start_time"], errors="coerce")
    df_event["end_time"] = pd.to_datetime(df_event["end_time"], errors="coerce")

    return df_track, df_point, df_event


def plot_all_tracks(df_track):
    plt.figure(figsize=(12, 10))

    for _, grp in df_track.groupby("track_id"):
        plt.plot(grp["lon"], grp["lat"], linewidth=0.4, alpha=0.25, color="gray")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("All AIS Tracks in LA / Long Beach")
    plt.tight_layout()

    out = FIG_DIR / "01_all_tracks_overview.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"已保存: {out}")


def plot_anomaly_points(df_point):
    plt.figure(figsize=(12, 10))

    # 背景轨迹
    for _, grp in df_point.groupby("track_id"):
        plt.plot(grp["lon"], grp["lat"], linewidth=0.35, alpha=0.18, color="lightgray")

    stop_df = df_point[df_point["abnormal_stop_flag"] == True]
    turn_df = df_point[df_point["sharp_turn_flag"] == True]
    drift_df = df_point[df_point["drift_like_flag"] == True]

    if not stop_df.empty:
        plt.scatter(stop_df["lon"], stop_df["lat"], s=6, alpha=0.45, label="abnormal_stop", c="orange")
    if not turn_df.empty:
        plt.scatter(turn_df["lon"], turn_df["lat"], s=8, alpha=0.60, label="sharp_turn", c="red")
    if not drift_df.empty:
        plt.scatter(drift_df["lon"], drift_df["lat"], s=8, alpha=0.60, label="drift_like", c="blue")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Candidate Anomaly Points")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / "02_anomaly_points_overview.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"已保存: {out}")


def get_top_event(df_event, event_type, rank=1):
    sub = df_event[df_event["event_type"] == event_type].copy()
    if sub.empty:
        return None

    sub = sub.sort_values(["duration_min", "max_score"], ascending=[False, False]).reset_index(drop=True)

    if rank > len(sub):
        return None

    return sub.iloc[rank - 1]


def plot_single_event(df_track, df_point, event_row, out_name, color):
    track_id = event_row["track_id"]
    start_time = event_row["start_time"]
    end_time = event_row["end_time"]
    event_type = event_row["event_type"]

    # 局部完整轨迹：从 df_track 取
    local_track = df_track[
        (df_track["track_id"] == track_id) &
        (df_track["time"] >= start_time - pd.Timedelta(minutes=30)) &
        (df_track["time"] <= end_time + pd.Timedelta(minutes=30))
        ].copy()

    if local_track.empty:
        print(f"{event_type} 没有找到局部轨迹")
        return

    # 局部异常点：从 df_point 取
    local_points = df_point[
        (df_point["track_id"] == track_id) &
        (df_point["time"] >= start_time - pd.Timedelta(minutes=30)) &
        (df_point["time"] <= end_time + pd.Timedelta(minutes=30))
        ].copy()

    flag_map = {
        "abnormal_stop": "abnormal_stop_flag",
        "sharp_turn": "sharp_turn_flag",
        "drift_like": "drift_like_flag",
    }
    flag_col = flag_map[event_type]

    flagged = local_points[local_points[flag_col] == True].copy()

    plt.figure(figsize=(10, 8))

    # 画完整局部轨迹
    plt.plot(local_track["lon"], local_track["lat"], linewidth=1.2, alpha=0.8, color="gray", label="local_track")

    # 画异常点
    if not flagged.empty:
        plt.scatter(flagged["lon"], flagged["lat"], s=25, alpha=0.8, c=color, label=event_type)

    # 起点终点
    first_row = local_track.iloc[0]
    last_row = local_track.iloc[-1]
    plt.scatter(first_row["lon"], first_row["lat"], s=50, c="green", marker="o", label="start")
    plt.scatter(last_row["lon"], last_row["lat"], s=50, c="black", marker="x", label="end")

    title = (
        f"{event_type}\n"
        f"track_id={track_id}, mmsi={event_row['mmsi']}\n"
        f"start={start_time}, end={end_time}, duration={event_row['duration_min']:.1f} min"
    )
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / out_name
    plt.savefig(out, dpi=220)
    plt.close()
    print(f"已保存: {out}")


def main():
    print("=== 第四步：异常事件可视化 ===")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not TRACK_FILE.exists():
        raise FileNotFoundError(f"找不到文件: {TRACK_FILE}")
    if not POINT_FILE.exists():
        raise FileNotFoundError(f"找不到文件: {POINT_FILE}")
    if not EVENT_FILE.exists():
        raise FileNotFoundError(f"找不到文件: {EVENT_FILE}")

    df_track, df_point, df_event = load_data()

    print(f"轨迹点数: {len(df_track)}")
    print(f"逐点异常表点数: {len(df_point)}")
    print(f"事件数: {len(df_event)}")

    plot_all_tracks(df_track)
    plot_anomaly_points(df_point)

    top_stop = get_top_event(df_event, "abnormal_stop", rank=1)
    top_turn = get_top_event(df_event, "sharp_turn", rank=1)
    top_drift = get_top_event(df_event, "drift_like", rank=1)

    if top_stop is not None:
        plot_single_event(df_track, df_point, top_stop, "03_top_abnormal_stop.png", "orange")

    if top_turn is not None:
        plot_single_event(df_track, df_point, top_turn, "04_top_sharp_turn.png", "red")

    if top_drift is not None:
        plot_single_event(df_track, df_point, top_drift, "05_top_drift_like.png", "blue")

    print("\n全部可视化完成。")


if __name__ == "__main__":
    main()
