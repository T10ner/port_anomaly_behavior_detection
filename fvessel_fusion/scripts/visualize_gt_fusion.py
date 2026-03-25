from pathlib import Path
import argparse
import cv2
import pandas as pd


def load_fusion_txt(fusion_path: Path) -> pd.DataFrame:
    df = pd.read_csv(fusion_path, header=None)
    df.columns = [
        "frame", "mmsi", "x", "y", "w", "h",
        "c1", "c2", "c3", "c4"
    ]
    return df


def draw_boxes(frame_img, rows):
    for _, row in rows.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["w"])
        h = int(row["h"])
        mmsi = str(int(row["mmsi"]))

        x2 = x + w
        y2 = y + h

        cv2.rectangle(frame_img, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_img,
            f"MMSI:{mmsi}",
            (x, max(30, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
    return frame_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True, help="clip目录")
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0, 180, 360],
        help="要可视化的秒数"
    )
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    gt_dir = clip_dir / "gt"

    video_files = sorted(clip_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"找不到视频: {clip_dir}")
    video_path = video_files[0]

    fusion_files = sorted(gt_dir.glob("*_gt_fusion.txt"))
    if not fusion_files:
        raise FileNotFoundError(f"找不到 gt_fusion.txt: {gt_dir}")
    fusion_path = fusion_files[0]

    out_dir = clip_dir / "processed" / "fusion_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_fusion_txt(fusion_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    print("=" * 60)
    print("视频与标注信息")
    print("=" * 60)
    print(f"video: {video_path.name}")
    print(f"fusion txt: {fusion_path.name}")
    print(f"fps: {fps}")
    print(f"frame_count: {frame_count}")
    print(f"duration_sec: {duration:.2f}")
    print(f"fusion rows: {len(df)}")
    print(f"unique mmsi in fusion: {df['mmsi'].nunique()}")

    for sec in args.times:
        if sec < 0 or sec > duration:
            print(f"[跳过] {sec}s 超出范围")
            continue

        frame_idx = int(round(sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[失败] 无法读取 {sec}s 的帧")
            continue

        # 先尝试精确匹配 frame
        rows = df[df["frame"] == frame_idx].copy()

        # 如果精确匹配不到，再找最近的一帧
        used_frame = frame_idx
        if len(rows) == 0:
            nearest_idx = (df["frame"] - frame_idx).abs().idxmin()
            used_frame = int(df.loc[nearest_idx, "frame"])
            rows = df[df["frame"] == used_frame].copy()

        frame = draw_boxes(frame, rows)

        cv2.putText(
            frame,
            f"time={sec:.1f}s  video_frame={frame_idx}  fusion_frame={used_frame}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        out_path = out_dir / f"fusion_{int(sec):04d}s.jpg"
        cv2.imwrite(str(out_path), frame)

        print(f"[保存] {out_path} | boxes={len(rows)} | fusion_frame={used_frame}")

    cap.release()
    print("\n完成。现在你可以直接看哪些 MMSI 出现在画面中。")


if __name__ == "__main__":
    main()
