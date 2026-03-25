from pathlib import Path
import argparse
from collections import defaultdict, deque
import math

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return fps, width, height, frame_count


def get_waterline_points():
    """
    仅用于辅助画参考线和做一个很弱的几何门。
    不再做整幅遮罩。
    """
    pts = np.array([
        [0, 1015],
        [220, 1008],
        [480, 1000],
        [760, 992],
        [1040, 982],
        [1320, 974],
        [1600, 970],
        [1880, 970],
        [2160, 973],
        [2560, 980],
    ], dtype=np.int32)
    return pts


def shoreline_y_at_x(x: float) -> float:
    pts = get_waterline_points()
    xs = pts[:, 0].astype(float)
    ys = pts[:, 1].astype(float)
    x = np.clip(x, xs.min(), xs.max())
    return float(np.interp(x, xs, ys))


def basic_boat_filter(x1, y1, x2, y2):
    """
    只保留很弱的几何约束，不做强遮罩。
    """
    w = x2 - x1
    h = y2 - y1
    area = w * h
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    y_bottom = y2

    # 太小的噪声直接去掉
    if w < 18 or h < 8:
        return False
    if area < 250:
        return False

    # 太高的小框，一般不是船
    if cy < 900 and w < 120:
        return False

    # 很弱的岸线门：底边至少要接近/低于岸线
    shore_y = shoreline_y_at_x(cx)
    if y_bottom < shore_y - 5:
        return False

    return True


def center_distance(p1, p2):
    return math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))


def is_static_false_positive(center_hist, width_hist, height_hist,
                             static_window=40,
                             min_history=20,
                             max_static_disp=10.0,
                             max_size_change=18.0):
    """
    判断一个 track 在最近一段时间里是否几乎不动。
    用来压建筑/岸边结构这种假目标。
    """
    if len(center_hist) < min_history:
        return False

    # 只看最近 static_window 个点
    centers = list(center_hist)[-static_window:]
    widths = list(width_hist)[-static_window:]
    heights = list(height_hist)[-static_window:]

    if len(centers) < min_history:
        return False

    disp = center_distance(centers[0], centers[-1])

    w_change = max(widths) - min(widths) if widths else 0.0
    h_change = max(heights) - min(heights) if heights else 0.0
    size_change = max(w_change, h_change)

    # 位移小、尺度变化也小 -> 更像建筑/固定结构
    if disp < max_static_disp and size_change < max_size_change:
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_dir", type=str, required=True, help="clip目录")
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--max_seconds", type=float, default=100.0, help="调试阶段只跑前多少秒")
    parser.add_argument("--min_hits_show", type=int, default=4, help="至少连续出现几次才显示")
    parser.add_argument("--static_window", type=int, default=40, help="静态判定窗口长度")
    parser.add_argument("--min_history", type=int, default=20, help="至少积累多少历史点才做静态判定")
    parser.add_argument("--max_static_disp", type=float, default=10.0, help="最近窗口内最大静态位移阈值")
    parser.add_argument("--max_size_change", type=float, default=18.0, help="最近窗口内最大尺度变化阈值")
    args = parser.parse_args()

    clip_dir = Path(args.clip_dir)
    video_files = sorted(clip_dir.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"找不到视频: {clip_dir}")
    video_path = video_files[0]

    out_dir = clip_dir / "processed" / "track_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_video = out_dir / "track_clip01_thin.mp4"
    out_csv = out_dir / "track_clip01.csv"

    fps, width, height, frame_count = get_video_info(video_path)
    max_frames = min(frame_count, int(args.max_seconds * fps))
    waterline_pts = get_waterline_points()

    print("=" * 60)
    print("开始 YOLO + Tracking（原图检测 + 静态目标抑制）")
    print("=" * 60)
    print(f"video: {video_path}")
    print(f"model: {args.model}")
    print(f"tracker: {args.tracker}")
    print(f"imgsz: {args.imgsz}")
    print(f"conf: {args.conf}")
    print(f"iou: {args.iou}")
    print(f"device: {args.device}")
    print(f"fps: {fps}, size: {width}x{height}, frames: {frame_count}")
    print(f"max_seconds: {args.max_seconds}")
    print(f"max_frames: {max_frames}")
    print(f"min_hits_show: {args.min_hits_show}")
    print(f"static_window: {args.static_window}")
    print(f"min_history: {args.min_history}")
    print(f"max_static_disp: {args.max_static_disp}")
    print(f"max_size_change: {args.max_size_change}")

    model = YOLO(args.model)

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    rows = []

    # 轨迹历史
    track_seen_count = defaultdict(int)
    track_center_hist = defaultdict(lambda: deque(maxlen=80))
    track_width_hist = defaultdict(lambda: deque(maxlen=80))
    track_height_hist = defaultdict(lambda: deque(maxlen=80))

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # 直接对原图做 tracking
        result = model.track(
            source=frame,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            persist=True,
            verbose=False
        )[0]

        vis_frame = frame.copy()
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().numpy()
            else:
                track_ids = np.full(len(xyxy), -1, dtype=int)

            for box, conf, cls_id, track_id in zip(xyxy, confs, clss, track_ids):
                cls_name = result.names[int(cls_id)]
                if cls_name != "boat":
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # 先过一层很弱的几何过滤
                if not basic_boat_filter(x1, y1, x2, y2):
                    continue

                tid = int(track_id)

                # 记录历史
                track_seen_count[tid] += 1
                track_center_hist[tid].append((cx, cy))
                track_width_hist[tid].append(w)
                track_height_hist[tid].append(h)

                # 至少出现几帧再显示
                if track_seen_count[tid] < args.min_hits_show:
                    continue

                # 静态目标抑制：几乎不动的 track 当作建筑/固定结构抑制掉
                static_fp = is_static_false_positive(
                    center_hist=track_center_hist[tid],
                    width_hist=track_width_hist[tid],
                    height_hist=track_height_hist[tid],
                    static_window=args.static_window,
                    min_history=args.min_history,
                    max_static_disp=args.max_static_disp,
                    max_size_change=args.max_size_change
                )
                if static_fp:
                    continue

                rows.append({
                    "frame_idx": frame_idx,
                    "track_id": tid,
                    "cls_name": cls_name,
                    "conf": float(conf),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "w": w,
                    "h": h,
                    "cx": cx,
                    "cy": cy
                })

                # 细框
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # 只给稍大一点的船画标签
                if w >= 120:
                    label = f"T{tid} {conf:.2f}"
                    cv2.putText(
                        vis_frame,
                        label,
                        (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

        # 画岸线参考线，方便你看
        cv2.polylines(vis_frame, [waterline_pts.reshape((-1, 1, 2))], isClosed=False, color=(255, 0, 0), thickness=2)

        writer.write(vis_frame)

        if frame_idx % 100 == 0:
            print(f"已处理 frame {frame_idx}/{max_frames}")

        frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n完成。输出文件：")
    print(out_video)
    print(out_csv)


if __name__ == "__main__":
    main()
