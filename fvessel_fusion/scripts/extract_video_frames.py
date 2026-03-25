from pathlib import Path
import argparse
import cv2
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True, help="clip目录")
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0, 60, 120, 180, 240, 300, 360, 420],
        help="要抽帧的秒数列表，例如 0 60 120"
    )
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    video_files = sorted(clip_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"在 {clip_dir} 下没有找到 mp4")

    video_path = video_files[0]
    out_dir = clip_dir / "processed" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    print("=" * 60)
    print("视频信息")
    print("=" * 60)
    print(f"video: {video_path.name}")
    print(f"fps: {fps}")
    print(f"frame_count: {frame_count}")
    print(f"duration_sec: {duration:.2f}")

    for sec in args.times:
        if sec < 0 or sec > duration:
            print(f"[跳过] {sec}s 超出视频范围")
            continue

        frame_idx = int(round(sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[失败] 读取 {sec}s 对应帧失败")
            continue

        text = f"time={sec:.1f}s, frame={frame_idx}"
        cv2.putText(
            frame,
            text,
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        out_path = out_dir / f"frame_{int(sec):04d}s.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"[保存] {out_path}")

    cap.release()
    print("\n完成。请先看这些帧里有哪些船、画面左右和远近关系。")


if __name__ == "__main__":
    main()
