from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from regions import get_channel_polygon, get_turning_basin_polygon, get_anchorage_polygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_FILE = PROJECT_ROOT / "data" / "processed" / "ais_valid_tracks.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "figures" / "06_regions_check.png"


def main():
    df = pd.read_csv(TRACK_FILE)

    channel = get_channel_polygon()
    turning = get_turning_basin_polygon()
    anchorage = get_anchorage_polygon()

    df["in_channel"] = df.apply(lambda r: channel.contains(Point(r["lon"], r["lat"])), axis=1)
    df["in_turning"] = df.apply(lambda r: turning.contains(Point(r["lon"], r["lat"])), axis=1)
    df["in_anchorage"] = df.apply(lambda r: anchorage.contains(Point(r["lon"], r["lat"])), axis=1)

    plt.figure(figsize=(12, 10))

    # 背景轨迹
    for _, grp in df.groupby("track_id"):
        plt.plot(grp["lon"], grp["lat"], linewidth=0.35, alpha=0.12, color="lightgray")

    # 区域内点
    ch = df[df["in_channel"]]
    tb = df[df["in_turning"]]
    an = df[df["in_anchorage"]]

    if not ch.empty:
        plt.scatter(ch["lon"], ch["lat"], s=3, alpha=0.20, label="channel", c="green")
    if not tb.empty:
        plt.scatter(tb["lon"], tb["lat"], s=4, alpha=0.25, label="turning_basin", c="red")
    if not an.empty:
        plt.scatter(an["lon"], an["lat"], s=4, alpha=0.25, label="anchorage", c="blue")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Manual Region Check")
    plt.legend()
    plt.tight_layout()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=220)
    plt.close()

    print(f"已保存: {OUT_FILE}")


if __name__ == "__main__":
    main()
