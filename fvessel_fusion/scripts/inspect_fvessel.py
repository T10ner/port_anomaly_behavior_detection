from pathlib import Path
import argparse
import re
import cv2
import pandas as pd


def parse_camera_file(camera_file: Path):
    """
    尽量宽松地解析 camera_para.txt
    目标：把里面的数字都提取出来
    预期顺序通常是：
    [Lon, Lat, Horizontal Orientation, Vertical Orientation,
     Camera Height, Horizontal FoV, Vertical FoV, fx, fy, u0, v0]
    """
    text = camera_file.read_text(encoding="utf-8", errors="ignore")
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    values = [float(x) for x in numbers]

    names = [
        "Lon", "Lat", "Horizontal_Orientation", "Vertical_Orientation",
        "Camera_Height", "Horizontal_FoV", "Vertical_FoV",
        "fx", "fy", "u0", "v0"
    ]

    result = {}
    for i, v in enumerate(values[:len(names)]):
        result[names[i]] = v

    return result, text


def inspect_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps and fps > 0 else None

    cap.release()

    return {
        "path": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def inspect_ais_folder(ais_dir: Path, preview_files=3):
    csv_files = sorted(ais_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"在 {ais_dir} 下没有找到 csv 文件")

    info = {
        "file_count": len(csv_files),
        "files": [],
        "all_columns_union": set(),
    }

    for csv_file in csv_files[:preview_files]:
        try:
            df = pd.read_csv(csv_file)
            info["files"].append({
                "file": str(csv_file.name),
                "rows": len(df),
                "columns": list(df.columns),
                "head": df.head(3).to_dict(orient="records"),
            })
            info["all_columns_union"].update(df.columns)
        except Exception as e:
            info["files"].append({
                "file": str(csv_file.name),
                "error": str(e)
            })

    return info, csv_files


def find_video_file(clip_dir: Path):
    mp4_files = sorted(clip_dir.glob("*.mp4"))
    if len(mp4_files) == 0:
        raise RuntimeError(f"在 {clip_dir} 下没有找到 mp4 视频")
    if len(mp4_files) > 1:
        print(f"[提醒] 发现多个 mp4，将默认使用第一个: {mp4_files[0].name}")
    return mp4_files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_dir",
        type=str,
        required=True,
        help="FVessel 单个 clip 的目录，例如 data/clip_01"
    )
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    if not clip_dir.exists():
        raise FileNotFoundError(f"目录不存在: {clip_dir}")

    print("=" * 60)
    print("1) 检查目录")
    print("=" * 60)
    print(f"clip_dir: {clip_dir.resolve()}")

    video_file = find_video_file(clip_dir)
    ais_dir = clip_dir / "ais"
    camera_file = clip_dir / "camera_para.txt"

    print(f"视频文件: {video_file.name}")
    print(f"AIS目录存在: {ais_dir.exists()} -> {ais_dir}")
    print(f"相机参数文件存在: {camera_file.exists()} -> {camera_file}")

    if not ais_dir.exists():
        raise FileNotFoundError(f"找不到 AIS 目录: {ais_dir}")
    if not camera_file.exists():
        raise FileNotFoundError(f"找不到相机参数文件: {camera_file}")

    print("\n" + "=" * 60)
    print("2) 视频信息")
    print("=" * 60)
    video_info = inspect_video(video_file)
    for k, v in video_info.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 60)
    print("3) 相机参数")
    print("=" * 60)
    camera_dict, raw_camera_text = parse_camera_file(camera_file)
    print("解析出的参数：")
    for k, v in camera_dict.items():
        print(f"{k}: {v}")

    print("\n原始 camera_para.txt 内容：")
    print(raw_camera_text[:1000])  # 防止太长

    print("\n" + "=" * 60)
    print("4) AIS 文件信息")
    print("=" * 60)
    ais_info, csv_files = inspect_ais_folder(ais_dir, preview_files=3)
    print(f"AIS csv 文件数量: {ais_info['file_count']}")
    print(f"预览到的字段合集: {sorted(list(ais_info['all_columns_union']))}")

    for item in ais_info["files"]:
        print("-" * 40)
        print(f"文件: {item['file']}")
        if "error" in item:
            print(f"读取失败: {item['error']}")
            continue
        print(f"行数: {item['rows']}")
        print(f"列名: {item['columns']}")
        print("前3行:")
        for row in item["head"]:
            print(row)

    print("\n" + "=" * 60)
    print("5) 基础判断")
    print("=" * 60)
    expected_cols = {"MMSI", "Lon", "Lat", "Speed", "Course", "Heading", "Timestamp"}
    found_cols = ais_info["all_columns_union"]
    missing_cols = expected_cols - found_cols

    if len(camera_dict) >= 7:
        print("[OK] camera_para.txt 基本可解析")
    else:
        print("[警告] camera_para.txt 解析出的参数偏少，后面可能需要手动处理")

    if len(missing_cols) == 0:
        print("[OK] AIS 关键字段齐全")
    else:
        print(f"[警告] AIS 缺少关键字段: {missing_cols}")

    print(f"[OK] 视频可读，AIS csv 数量 = {len(csv_files)}")
    print("\n检查完成。下一步就是合并 AIS 小 CSV。")


if __name__ == "__main__":
    main()
