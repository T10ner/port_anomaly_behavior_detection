from pathlib import Path
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True, help="clip目录")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="YOLO权重")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.20, help="置信度阈值")
    parser.add_argument("--device", type=str, default="0", help="设备，GPU一般填0")
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    video_files = sorted(clip_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"找不到视频: {clip_dir}")

    video_path = video_files[0]
    out_project = clip_dir / "processed" / "yolo_runs"
    out_project.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("开始 YOLO 推理")
    print("=" * 60)
    print(f"video: {video_path}")
    print(f"model: {args.model}")
    print(f"imgsz: {args.imgsz}")
    print(f"conf: {args.conf}")
    print(f"device: {args.device}")

    model = YOLO(args.model)

    results = model.predict(
        source=str(video_path),
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(out_project),
        name="detect_clip01",
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=True,
        stream=False
    )

    print("\n完成。结果保存在：")
    print(out_project / "detect_clip01")


if __name__ == "__main__":
    main()
