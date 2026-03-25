from pathlib import Path
import pandas as pd
import numpy as np
from shapely.geometry import Point

from regions import get_channel_polygon, get_turning_basin_polygon, get_anchorage_polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ais_valid_tracks.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "track_summary.csv"

POINT_OUT = PROJECT_ROOT / "data" / "processed" / "anomaly_point_flags.csv"
EVENT_OUT = PROJECT_ROOT / "data" / "processed" / "anomaly_events.csv"


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def angle_diff_deg(a, b):
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


def add_context_features(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    use_cols = [
        "track_id", "start_time", "end_time",
        "n_points", "mean_sog", "max_sog",
        "stop_ratio", "likely_stationary"
    ]
    summary_small = summary[use_cols].copy()

    df = df.merge(summary_small, on="track_id", how="left")

    df["time_from_start_min"] = (
                                        df["time"] - df["start_time"]
                                ).dt.total_seconds() / 60.0

    df["time_to_end_min"] = (
                                    df["end_time"] - df["time"]
                            ).dt.total_seconds() / 60.0

    # 每条轨迹的“中心点”，后面用于判断静止轨迹是不是只是 GPS 抖动
    centers = (
        df.groupby("track_id")
        .agg(center_lat=("lat", "median"), center_lon=("lon", "median"))
        .reset_index()
    )
    df = df.merge(centers, on="track_id", how="left")

    valid_center = (
            df["lat"].notna() & df["lon"].notna() &
            df["center_lat"].notna() & df["center_lon"].notna()
    )

    df["dist_to_center_m"] = np.nan
    df.loc[valid_center, "dist_to_center_m"] = haversine_m(
        df.loc[valid_center, "lon"],
        df.loc[valid_center, "lat"],
        df.loc[valid_center, "center_lon"],
        df.loc[valid_center, "center_lat"],
    )

    return df

def add_region_flags(df: pd.DataFrame) -> pd.DataFrame:
    channel = get_channel_polygon()
    turning = get_turning_basin_polygon()
    anchorage = get_anchorage_polygon()

    df = df.copy()

    df["in_channel"] = df.apply(
        lambda r: channel.contains(Point(r["lon"], r["lat"])),
        axis=1
    )

    df["in_turning_basin"] = df.apply(
        lambda r: turning.contains(Point(r["lon"], r["lat"])),
        axis=1
    )

    df["in_anchorage"] = df.apply(
        lambda r: anchorage.contains(Point(r["lon"], r["lat"])),
        axis=1
    )

    return df

def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["abnormal_stop_flag"] = False
    df["sharp_turn_flag"] = False
    df["drift_like_flag"] = False

    # 1) abnormal_stop：
    # 只在主航道内、且不在回转区时检测
    df["abnormal_stop_flag"] = (
            (df["in_channel"]) &
            (~df["in_turning_basin"]) &
            (~df["likely_stationary"].fillna(False)) &
            (df["max_sog"].fillna(0) >= 8.0) &
            (df["sog"].fillna(999) <= 0.3) &
            (df["time_from_start_min"].fillna(0) >= 20.0) &
            (df["time_to_end_min"].fillna(0) >= 20.0) &
            (
                    df["step_dist_m"].isna() |
                    (df["step_dist_m"].fillna(999) <= 15.0)
            )
    )

    # 2) sharp_turn：
    # 只在主航道内，排除正常回转区
    df["sharp_turn_flag"] = (
            (df["in_channel"]) &
            (~df["in_turning_basin"]) &
            (~df["likely_stationary"].fillna(False)) &
            (df["dt_min"].between(0.1, 3.0)) &
            (df["sog"].fillna(0) >= 4.0) &
            (df["delta_cog_deg"].fillna(0) >= 60.0) &
            (df["step_dist_m"].fillna(0) >= 30.0)
    )

    # 3) drift_like：
    # 只在锚地/等待区里检测
    df["drift_like_flag"] = (
            (df["in_anchorage"]) &
            (df["likely_stationary"].fillna(False)) &
            (df["sog"].fillna(999) <= 0.8) &
            (df["time_from_start_min"].fillna(0) >= 10.0) &
            (df["time_to_end_min"].fillna(0) >= 10.0) &
            (df["dist_to_center_m"].fillna(0) >= 60.0) &
            (df["step_dist_m"].fillna(999) <= 25.0)
    )

    return df

def build_event_table(df: pd.DataFrame, flag_col: str, event_type: str,
                      max_gap_min: float = 5.0) -> pd.DataFrame:
    sub = df[df[flag_col]].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "event_id", "track_id", "mmsi", "event_type", "start_time", "end_time",
            "duration_min", "n_points", "max_score", "mean_score"
        ])

    score_map = {
        "abnormal_stop_flag": "sog",
        "sharp_turn_flag": "delta_cog_deg",
        "drift_like_flag": "dist_to_center_m",
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
    if events.empty:
        return events

    keep = []

    for _, row in events.iterrows():
        et = row["event_type"]

        if et == "abnormal_stop":
            # 更严格：至少 15 分钟，且点数不少于 5
            keep.append((row["duration_min"] >= 15) and (row["n_points"] >= 5))

        elif et == "sharp_turn":
            # 转向：至少 2 个点，或者最大角度很大
            keep.append((row["n_points"] >= 2) or (row["max_score"] >= 90))

        elif et == "drift_like":
            # 漂移：至少 20 分钟且离中心够远
            keep.append(
                (row["duration_min"] >= 20) and
                (row["n_points"] >= 5) and
                (row["max_score"] >= 70)
            )
        else:
            keep.append(False)

    return events[pd.Series(keep, index=events.index)].reset_index(drop=True)


def main():
    print("=== 第五步：收紧规则后的异常检测 ===")
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
    df = add_context_features(df, summary)
    df = add_region_flags(df)
    df = apply_rules(df)

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
