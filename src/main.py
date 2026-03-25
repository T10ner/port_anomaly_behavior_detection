from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_AIS = PROJECT_ROOT / "data" / "raw" / "ais_sample.csv"
CLEANED_AIS = PROJECT_ROOT / "data" / "processed" / "ais_cleaned.csv"


def detect_time_column(columns):
    """自动识别时间列名"""
    candidates = ["BaseDateTime", "BaseDateT", "time", "Time", "timestamp", "Timestamp"]
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"没有找到时间列，当前列名为: {list(columns)}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名"""
    time_col = detect_time_column(df.columns)

    rename_map = {
        "MMSI": "mmsi",
        time_col: "time",
        "LAT": "lat",
        "LON": "lon",
        "SOG": "sog",
        "COG": "cog",
        "Heading": "heading",
        "VesselName": "vessel_name",
        "IMO": "imo",
        "CallSign": "call_sign",
        "VesselType": "vessel_type",
        "Status": "status",
        "Length": "length",
        "Width": "width",
        "Draft": "draft",
        "Cargo": "cargo",
        "TransceiverClass": "transceiver_class",
    }

    keep_cols = [c for c in rename_map.keys() if c in df.columns]
    df = df[keep_cols].copy()
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def clean_ais(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗"""
    # 时间解析
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # 数值列转数值
    numeric_cols = ["lat", "lon", "sog", "cog", "heading", "length", "width", "draft"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除关键字段为空的记录
    df = df.dropna(subset=["mmsi", "time", "lat", "lon"]).copy()

    # 基础范围过滤
    df = df[df["lat"].between(-90, 90)]
    df = df[df["lon"].between(-180, 180)]

    if "sog" in df.columns:
        df = df[(df["sog"].isna()) | ((df["sog"] >= 0) & (df["sog"] <= 60))]

    if "cog" in df.columns:
        df = df[(df["cog"].isna()) | ((df["cog"] >= 0) & (df["cog"] <= 360))]

    # 去重
    df = df.drop_duplicates(subset=["mmsi", "time", "lat", "lon"])

    # 排序
    df = df.sort_values(["mmsi", "time"]).reset_index(drop=True)

    return df


def split_tracks(df: pd.DataFrame, time_gap_minutes: int = 30) -> pd.DataFrame:
    """按时间间隔切分轨迹"""
    df = df.copy()

    df["time_diff_min"] = (
        df.groupby("mmsi")["time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    df["new_track_flag"] = (
            df["time_diff_min"].isna() | (df["time_diff_min"] > time_gap_minutes)
    ).astype(int)

    df["track_seg"] = df.groupby("mmsi")["new_track_flag"].cumsum()
    df["track_id"] = df["mmsi"].astype(str) + "_" + df["track_seg"].astype(str)

    return df


def main():
    print("=== 第一步：读取 AIS 文件 ===")
    print(f"文件路径: {RAW_AIS.resolve()}")

    if not RAW_AIS.exists():
        raise FileNotFoundError(f"找不到文件: {RAW_AIS}")

    # 读取
    df_raw = pd.read_csv(RAW_AIS)
    print(f"原始行数: {len(df_raw)}")
    print("原始列名:")
    print(list(df_raw.columns))

    # 标准化列名
    df_std = standardize_columns(df_raw)
    print("\n标准化后的列名:")
    print(list(df_std.columns))

    # 清洗
    df_clean = clean_ais(df_std)
    print(f"\n清洗后行数: {len(df_clean)}")
    print(f"MMSI 数量: {df_clean['mmsi'].nunique()}")

    # 切分轨迹
    df_track = split_tracks(df_clean, time_gap_minutes=30)
    print(f"轨迹段数量: {df_track['track_id'].nunique()}")

    # 输出预览
    print("\n前 10 行：")
    print(df_track.head(10))

    # 保存
    CLEANED_AIS.parent.mkdir(parents=True, exist_ok=True)
    df_track.to_csv(CLEANED_AIS, index=False, encoding="utf-8-sig")
    print(f"\n已保存清洗结果到: {CLEANED_AIS.resolve()}")


if __name__ == "__main__":
    main()
