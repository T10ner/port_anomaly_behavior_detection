from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ais_cleaned.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "track_summary.csv"
VALID_FILE = PROJECT_ROOT / "data" / "processed" / "ais_valid_tracks.csv"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def build_track_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("track_id")

    summary = grp.agg(
        mmsi=("mmsi", "first"),
        start_time=("time", "min"),
        end_time=("time", "max"),
        n_points=("time", "size"),
        min_lat=("lat", "min"),
        max_lat=("lat", "max"),
        min_lon=("lon", "min"),
        max_lon=("lon", "max"),
        mean_sog=("sog", "mean"),
        max_sog=("sog", "max"),
        vessel_type=("vessel_type", "first"),
        vessel_name=("vessel_name", "first"),
    ).reset_index()

    summary["duration_min"] = (
                                      pd.to_datetime(summary["end_time"]) - pd.to_datetime(summary["start_time"])
                              ).dt.total_seconds() / 60.0

    # 低速/静止比例
    stop_ratio = grp["sog"].apply(lambda s: (s.fillna(0) <= 0.5).mean()).reset_index(name="stop_ratio")
    summary = summary.merge(stop_ratio, on="track_id", how="left")

    # 粗略判断是否可能是停泊/静止轨迹
    summary["likely_stationary"] = (
            (summary["mean_sog"].fillna(0) < 1.0) &
            (summary["stop_ratio"].fillna(0) > 0.7)
    )

    return summary


def filter_valid_tracks(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    # 先做一个保守筛选：点数不少于20，持续时间不少于10分钟
    valid_track_ids = summary[
        (summary["n_points"] >= 20) &
        (summary["duration_min"] >= 10)
        ]["track_id"]

    df_valid = df[df["track_id"].isin(valid_track_ids)].copy()
    return df_valid


def main():
    print("=== 第二步：轨迹统计 ===")
    print(f"读取文件: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到文件: {INPUT_FILE}")

    df = load_data(INPUT_FILE)
    print(f"输入行数: {len(df)}")
    print(f"轨迹段数量: {df['track_id'].nunique()}")

    summary = build_track_summary(df)
    print(f"生成轨迹统计表，轨迹数: {len(summary)}")

    df_valid = filter_valid_tracks(df, summary)
    print(f"筛选后保留轨迹段数量: {df_valid['track_id'].nunique()}")
    print(f"筛选后保留点数: {len(df_valid)}")

    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    df_valid.to_csv(VALID_FILE, index=False, encoding="utf-8-sig")

    print("\n轨迹统计前10条：")
    print(summary.head(10))

    print(f"\n已保存: {SUMMARY_FILE}")
    print(f"已保存: {VALID_FILE}")


if __name__ == "__main__":
    main()
