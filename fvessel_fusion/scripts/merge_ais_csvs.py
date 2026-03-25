from pathlib import Path
import argparse
import re
from datetime import datetime
import pandas as pd


def find_video_file(clip_dir: Path) -> Path:
    mp4_files = sorted(clip_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"在 {clip_dir} 下没有找到 mp4")
    return mp4_files[0]


def parse_video_time_range(video_name: str):
    """
    解析类似:
    2022_11_20_10_21_09_10_28_37_r.mp4

    得到:
    start_dt = 2022-11-20 10:21:09
    end_dt   = 2022-11-20 10:28:37
    """
    pattern = r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})"
    m = re.search(pattern, video_name)
    if not m:
        raise ValueError(f"无法从视频文件名解析时间: {video_name}")

    y, mo, d, h1, mi1, s1, h2, mi2, s2 = map(int, m.groups())
    start_dt = datetime(y, mo, d, h1, mi1, s1)
    end_dt = datetime(y, mo, d, h2, mi2, s2)
    return start_dt, end_dt


def read_and_normalize_csv(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    # 列名统一成小写
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 删除无意义索引列
    unnamed_cols = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # 记录来源文件
    df["source_file"] = csv_file.name

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_dir",
        type=str,
        required=True,
        help="FVessel 单个 clip 目录"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录，默认保存到 clip_dir/processed"
    )
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    if not clip_dir.exists():
        raise FileNotFoundError(f"目录不存在: {clip_dir}")

    ais_dir = clip_dir / "ais"
    if not ais_dir.exists():
        raise FileNotFoundError(f"找不到 AIS 目录: {ais_dir}")

    video_file = find_video_file(clip_dir)
    video_start_dt, video_end_dt = parse_video_time_range(video_file.name)

    out_dir = Path(args.out_dir) if args.out_dir else (clip_dir / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(ais_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {ais_dir} 下没有找到 csv 文件")

    print("=" * 60)
    print("1) 基本信息")
    print("=" * 60)
    print(f"clip_dir: {clip_dir}")
    print(f"video_file: {video_file.name}")
    print(f"video_start: {video_start_dt}")
    print(f"video_end  : {video_end_dt}")
    print(f"AIS csv 数量: {len(csv_files)}")
    print(f"输出目录: {out_dir}")

    print("\n" + "=" * 60)
    print("2) 读取并合并 AIS csv")
    print("=" * 60)

    dfs = []
    for i, csv_file in enumerate(csv_files, start=1):
        try:
            df = read_and_normalize_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"[跳过] 读取失败: {csv_file.name} -> {e}")

        if i % 100 == 0 or i == len(csv_files):
            print(f"已处理 {i}/{len(csv_files)} 个 csv")

    if not dfs:
        raise RuntimeError("没有成功读取任何 AIS csv")

    merged = pd.concat(dfs, ignore_index=True)

    print(f"\n合并后总行数: {len(merged)}")
    print(f"列名: {list(merged.columns)}")

    # 检查关键字段
    required_cols = ["mmsi", "lon", "lat", "speed", "course", "heading", "type", "timestamp"]
    missing = [c for c in required_cols if c not in merged.columns]
    if missing:
        raise RuntimeError(f"缺少关键字段: {missing}")

    # 类型处理
    merged["timestamp"] = pd.to_numeric(merged["timestamp"], errors="coerce")
    merged["mmsi"] = pd.to_numeric(merged["mmsi"], errors="coerce")

    for c in ["lon", "lat", "speed", "course", "heading", "type"]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # 删除关键字段缺失的记录
    before_dropna = len(merged)
    merged = merged.dropna(subset=["mmsi", "lon", "lat", "timestamp"])
    print(f"删除关键字段缺失后: {len(merged)} （删去 {before_dropna - len(merged)} 行）")

    # 时间戳转 datetime
    # 这里 AIS 的 timestamp 是 UTC 毫秒时间戳
    # 视频文件名是本地时间（中国时区），所以需要转成 Asia/Shanghai
    merged["datetime_utc"] = pd.to_datetime(
        merged["timestamp"], unit="ms", utc=True, errors="coerce"
    )
    merged["datetime"] = merged["datetime_utc"].dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    merged = merged.dropna(subset=["datetime"])

    # 排序
    merged = merged.sort_values(["timestamp", "mmsi"]).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("3) 去重")
    print("=" * 60)

    # 先做严格去重
    dedup_cols_strict = ["mmsi", "timestamp", "lon", "lat", "speed", "course", "heading", "type"]
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=dedup_cols_strict).reset_index(drop=True)
    print(f"严格去重后: {len(merged)} （去掉 {before_dedup - len(merged)} 行重复）")

    # 保存 merged_all
    out_all = out_dir / "merged_ais_all.csv"
    merged.to_csv(out_all, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("4) 按视频时间窗口筛选")
    print("=" * 60)

    print("AIS 最早时间:", merged["datetime"].min())
    print("AIS 最晚时间:", merged["datetime"].max())
    print("视频开始时间:", video_start_dt)
    print("视频结束时间:", video_end_dt)

    # 视频时间窗口
    mask = (merged["datetime"] >= pd.Timestamp(video_start_dt)) & (merged["datetime"] <= pd.Timestamp(video_end_dt))
    merged_window = merged.loc[mask].copy().reset_index(drop=True)

    out_window = out_dir / "merged_ais_video_window.csv"
    merged_window.to_csv(out_window, index=False, encoding="utf-8-sig")

    # 统计信息
    print(f"视频时间窗口内 AIS 行数: {len(merged_window)}")
    print(f"唯一 MMSI 数量: {merged_window['mmsi'].nunique()}")

    if len(merged_window) > 0:
        print(f"窗口内最早时间: {merged_window['datetime'].min()}")
        print(f"窗口内最晚时间: {merged_window['datetime'].max()}")

    print("\n" + "=" * 60)
    print("5) 输出文件")
    print("=" * 60)
    print(f"全部合并结果: {out_all}")
    print(f"视频窗口结果: {out_window}")

    print("\n完成。下一步就是检查 merged_ais_video_window.csv 是否合理。")


if __name__ == "__main__":
    main()
