from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ais_valid_tracks.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "track_summary.csv"

POINT_OUT = PROJECT_ROOT / "data" / "processed" / "anomaly_point_flags.csv"
EVENT_OUT = PROJECT_ROOT / "data" / "processed" / "anomaly_events.csv"


def haversine_m(lon1, lat1, lon2, lat2):
    """
    计算两点球面距离（米）
    """
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def angle_diff_deg(a, b):
    """
    计算航向角最小差值（0~180）
    """
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def load_data():
    df = pd.read_csv(INPUT_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    summary["start_time"] = pd.to_datetime(summary["start_time"], errors="coerce")
    summary["end_time"] = pd.to_datetime(summary["end_time"], errors="coerce")

    return df, summary


def add_prev_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["track_id", "time"]).copy()

    df["prev_time"] = df.groupby("track_id")["time"].shift(1)
    df["prev_lat"] = df.groupby("track_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("track_id")["lon"].shift(1)
    df["prev_sog"] = df.groupby("track_id")["sog"].shift(1)
    df["prev_cog"] = df.groupby("track_id")["cog"].shift(1)

    df["dt_min"] = (df["time"] - df["prev_time"]).dt.total_seconds() / 60.0

    valid_prev = (
            df["prev_lat"].notna() &
            df["prev_lon"].notna() &
            df["lat"].notna() &
            df["lon"].notna()
    )

    df["step_dist_m"] = np.nan
    df.loc[valid_prev, "step_dist_m"] = haversine_m(
        df.loc[valid_prev, "prev_lon"],
        df.loc[valid_prev, "prev_lat"],
        df.loc[valid_prev, "lon"],
        df.loc[valid_prev, "lat"],
    )

    df["calc_speed_mps"] = df["step_dist_m"] / (df["dt_min"] * 60.0)
    df["delta_sog"] = df["sog"] - df["prev_sog"]

    valid_cog = df["cog"].notna() & df["prev_cog"].notna()
    df["delta_cog_deg"] = np.nan
    df.loc[valid_cog, "delta_cog_deg"] = angle_diff_deg(
        df.loc[valid_cog, "cog"],
        df.loc[valid_cog, "prev_cog"]
    )

    return df


def apply_rules(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    第一版规则：
    1) abnormal_stop: 在非静止轨迹中出现持续低速点
    2) sharp_turn: 运动中短时间大转向
    3) drift_like: 在疑似静止轨迹中，低速但仍持续位移
    """
    use_cols = ["track_id", "likely_stationary", "mean_sog", "stop_ratio"]
    summary_small = summary[use_cols].copy()

    df = df.merge(summary_small, on="track_id", how="left")

    # 初始化
    df["abnormal_stop_flag"] = False
    df["sharp_turn_flag"] = False
    df["drift_like_flag"] = False

    # 规则 1：异常停车
    # 非明显静止轨迹里，低速/停止点先打标
    df["abnormal_stop_flag"] = (
            (df["sog"].fillna(999) <= 0.5) &
            (~df["likely_stationary"].fillna(False))
    )

    # 规则 2：异常转向
    # 船在动，并且短时间内转向幅度很大
    df["sharp_turn_flag"] = (
            (df["dt_min"].between(0.01, 5.0)) &
            (df["sog"].fillna(0) >= 2.0) &
            (df["delta_cog_deg"].fillna(0) >= 45.0)
    )

    # 规则 3：疑似漂移
    # 对“可能原本静止/停泊”的轨迹，在低速状态下仍有明显位移
    df["drift_like_flag"] = (
            (df["likely_stationary"].fillna(False)) &
            (df["sog"].fillna(999) <= 1.0) &
            (df["dt_min"].between(0.01, 5.0)) &
            (df["step_dist_m"].fillna(0) >= 20.0)
    )

    return df


def build_event_table(df: pd.DataFrame, flag_col: str, event_type: str,
                      max_gap_min: float = 5.0) -> pd.DataFrame:
    """
    把逐点标记聚合成事件段
    """
    sub = df[df[flag_col]].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "track_id", "mmsi", "event_type", "start_time", "end_time",
            "duration_min", "n_points", "max_score", "mean_score"
        ])

    score_map = {
        "abnormal_stop_flag": "sog",
        "sharp_turn_flag": "delta_cog_deg",
        "drift_like_flag": "step_dist_m",
    }
    score_col = score_map[flag_col]

    sub = sub.sort_values(["track_id", "time"]).copy()
    sub["prev_time_same_flag"] = sub.groupby("track_id")["time"].shift(1)
    sub["gap_min"] = (sub["time"] - sub["prev_time_same_flag"]).dt.total_seconds() / 60.0

    sub["new_event"] = (
            sub["gap_min"].isna() | (sub["gap_min"] > max_gap_min)
    ).astype(int)

    sub["event_seg"] = sub.groupby("track_id")["new_event"].cumsum()
    sub["event_id"] = (
            event_type + "_" +
            sub["track_id"].astype(str) + "_" +
            sub["event_seg"].astype(str)
    )

    events = sub.groupby("event_id").agg(
        track_id=("track_id", "first"),
        mmsi=("mmsi", "first"),
        start_time=("time", "min"),
        end_time=("time", "max"),
        n_points=("time", "size"),
        max_score=(score_col, "max"),
        mean_score=(score_col, "mean"),
    ).reset_index()

    events["event_type"] = event_type
    events["duration_min"] = (
                                     pd.to_datetime(events["end_time"]) - pd.to_datetime(events["start_time"])
                             ).dt.total_seconds() / 60.0

    return events[
        ["event_id", "track_id", "mmsi", "event_type", "start_time", "end_time",
         "duration_min", "n_points", "max_score", "mean_score"]
    ]


def post_filter_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    对事件再做一次筛选，去掉太碎的事件
    """
    if events.empty:
        return events

    keep = []

    for _, row in events.iterrows():
        et = row["event_type"]

        if et == "abnormal_stop":
            # 停车事件：至少持续 10 分钟
            if row["duration_min"] >= 10:
                keep.append(True)
            else:
                keep.append(False)

        elif et == "sharp_turn":
            # 转向事件：至少 1 个异常点即可保留
            if row["n_points"] >= 1:
                keep.append(True)
            else:
                keep.append(False)

        elif et == "drift_like":
            # 漂移事件：至少 3 个点，或持续 10 分钟
            if (row["n_points"] >= 3) or (row["duration_min"] >= 10):
                keep.append(True)
            else:
                keep.append(False)
        else:
            keep.append(False)

    return events[pd.Series(keep, index=events.index)].reset_index(drop=True)


def main():
    print("=== 第三步：第一版异常检测 ===")
    print(f"读取: {INPUT_FILE}")
    print(f"读取: {SUMMARY_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到文件: {INPUT_FILE}")
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"找不到文件: {SUMMARY_FILE}")

    df, summary = load_data()
    print(f"输入点数: {len(df)}")
    print(f"输入轨迹数: {df['track_id'].nunique()}")

    df = add_prev_features(df)
    df = apply_rules(df, summary)

    print("\n逐点异常数量：")
    print("abnormal_stop_flag =", int(df["abnormal_stop_flag"].sum()))
    print("sharp_turn_flag    =", int(df["sharp_turn_flag"].sum()))
    print("drift_like_flag    =", int(df["drift_like_flag"].sum()))

    stop_events = build_event_table(df, "abnormal_stop_flag", "abnormal_stop")
    turn_events = build_event_table(df, "sharp_turn_flag", "sharp_turn")
    drift_events = build_event_table(df, "drift_like_flag", "drift_like")

    all_events = pd.concat([stop_events, turn_events, drift_events], ignore_index=True)
    all_events = post_filter_events(all_events)

    print("\n聚合后事件数量：")
    if all_events.empty:
        print("没有检测到事件")
    else:
        print(all_events["event_type"].value_counts())

    POINT_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(POINT_OUT, index=False, encoding="utf-8-sig")
    all_events.to_csv(EVENT_OUT, index=False, encoding="utf-8-sig")

    print(f"\n已保存逐点标记: {POINT_OUT}")
    print(f"已保存事件表:   {EVENT_OUT}")

    if not all_events.empty:
        print("\n事件表示例前10行：")
        print(all_events.head(10))


if __name__ == "__main__":
    main()
