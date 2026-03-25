from pathlib import Path
import argparse
import re
import math
import pandas as pd
import matplotlib.pyplot as plt


def parse_camera_file(camera_file: Path):
    text = camera_file.read_text(encoding="utf-8", errors="ignore")
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    values = [float(x) for x in numbers]

    names = [
        "lon", "lat", "horizontal_orientation", "vertical_orientation",
        "camera_height", "horizontal_fov", "vertical_fov",
        "fx", "fy", "u0", "v0"
    ]

    result = {}
    for i, v in enumerate(values[:len(names)]):
        result[names[i]] = v
    return result


def haversine_m(lon1, lat1, lon2, lat2):
    """
    计算两点球面距离（米）
    """
    r = 6371000.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True, help="clip 目录")
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    processed_dir = clip_dir / "processed"
    csv_path = processed_dir / "merged_ais_video_window.csv"
    camera_file = clip_dir / "camera_para.txt"
    out_dir = processed_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")
    if not camera_file.exists():
        raise FileNotFoundError(f"找不到文件: {camera_file}")

    df = pd.read_csv(csv_path)
    camera = parse_camera_file(camera_file)

    cam_lon = camera["lon"]
    cam_lat = camera["lat"]

    if len(df) == 0:
        raise RuntimeError("merged_ais_video_window.csv 是空的，无法绘图")

    # 时间列处理
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 计算每个 AIS 点到相机的距离
    df["distance_to_camera_m"] = df.apply(
        lambda row: haversine_m(cam_lon, cam_lat, row["lon"], row["lat"]), axis=1
    )

    # 统计每艘船
    summary = (
        df.groupby("mmsi")
        .agg(
            point_count=("mmsi", "size"),
            min_distance_m=("distance_to_camera_m", "min"),
            mean_distance_m=("distance_to_camera_m", "mean"),
            start_time=("datetime", "min"),
            end_time=("datetime", "max"),
            mean_speed=("speed", "mean"),
        )
        .sort_values("min_distance_m")
        .reset_index()
    )

    summary_path = processed_dir / "closest_mmsi_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("AIS 概览统计")
    print("=" * 60)
    print(f"总 AIS 点数: {len(df)}")
    print(f"唯一 MMSI 数量: {df['mmsi'].nunique()}")
    print(f"相机位置: lon={cam_lon}, lat={cam_lat}")
    print("\n距离相机最近的前 10 艘船：")
    print(summary.head(10))

    # 图1：全部 AIS 点概览
    plt.figure(figsize=(10, 8))
    plt.scatter(df["lon"], df["lat"], s=8, alpha=0.5, label="AIS points")
    plt.scatter([cam_lon], [cam_lat], s=120, marker="*", label="Camera")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("AIS Points in Video Time Window")
    plt.legend()
    plt.tight_layout()
    fig1 = out_dir / "01_all_ais_points.png"
    plt.savefig(fig1, dpi=200)
    plt.close()

    # 图2：前10条最近船轨迹
    top_mmsi = summary.head(10)["mmsi"].tolist()
    df_top = df[df["mmsi"].isin(top_mmsi)].copy()

    plt.figure(figsize=(10, 8))
    for mmsi, g in df_top.groupby("mmsi"):
        g = g.sort_values("datetime")
        plt.plot(g["lon"], g["lat"], marker="o", markersize=2, linewidth=1, label=str(int(mmsi)))
    plt.scatter([cam_lon], [cam_lat], s=150, marker="*", label="Camera")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Top 10 Closest AIS Trajectories")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    fig2 = out_dir / "02_top10_closest_trajectories.png"
    plt.savefig(fig2, dpi=200)
    plt.close()

    # 图3：距离分布
    plt.figure(figsize=(10, 6))
    plt.hist(summary["min_distance_m"], bins=20)
    plt.xlabel("Minimum distance to camera (m)")
    plt.ylabel("Count of MMSI")
    plt.title("Distribution of Minimum Distance to Camera")
    plt.tight_layout()
    fig3 = out_dir / "03_min_distance_hist.png"
    plt.savefig(fig3, dpi=200)
    plt.close()

    print("\n输出文件：")
    print(fig1)
    print(fig2)
    print(fig3)
    print(summary_path)
    print("\n完成。下一步可以根据最近的几艘船去做视频目标检测与匹配。")


if __name__ == "__main__":
    main()
