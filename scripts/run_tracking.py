#!/usr/bin/env python
"""Chạy detector YOLO cố định + một tracker (boxmot) trên một video / chuỗi ảnh.

Mọi nhóm trong lớp dùng CHUNG detector (``yolo26n.pt``), CHUNG kích thước ảnh đầu
vào (``IMG_SIZE``) và CHUNG mô hình Re-ID (``REID_WEIGHTS``) cho các tracker có
dùng ngoại hình. Biến được phép thay đổi là: ``--tracker``, ``--conf`` và ``--iou``
(ngưỡng của detector, không phải ngưỡng bên trong tracker).

Ví dụ:
    python scripts/run_tracking.py \\
        --source "$LAB_DATA/video_1/img1" \\
        --seq-name video_1 \\
        --tracker bytetrack --conf 0.3 --iou 0.5 \\
        --out runs/nop_bai --save-video --max-frames 150

Định dạng code theo Google Python Style Guide:
https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from boxmot import BotSort, ByteTrack, DeepOcSort, OcSort, StrongSort

# ---------------------------------------------------------------------------
# Các biến CỐ ĐỊNH cho cả lớp — KHÔNG sửa khi làm bài chính. Nếu muốn thử
# nghiệm thêm sau khi đã nộp, hãy ghi rõ trong báo cáo rằng đó là phần mở rộng.
# ---------------------------------------------------------------------------
DETECTOR_WEIGHTS = "yolo26n.pt"
IMG_SIZE = 640
PERSON_CLASS_ID = 0  # lớp "person" trong COCO — lab chỉ chấm người đi bộ
REID_WEIGHTS = Path("osnet_x0_25_msmt17.pt")  # tự tải về lần chạy đầu tiên

TRACKER_CHOICES = ["bytetrack", "ocsort", "botsort", "strongsort", "deepocsort"]
USES_APPEARANCE = {"botsort", "strongsort", "deepocsort"}
TRACKER_CLASSES = {
    "bytetrack": ByteTrack,
    "ocsort": OcSort,
    "botsort": BotSort,
    "strongsort": StrongSort,
    "deepocsort": DeepOcSort,
}


def make_tracker(name: str, device: str):
    """Tạo tracker BoxMOT với trọng số Re-ID cố định khi cần.

    Args:
        name: Tên tracker đã chọn.
        device: Thiết bị chạy Re-ID.

    Returns:
        Tracker tương ứng từ BoxMOT.

    Raises:
        KeyError: Khi tên tracker không hợp lệ.
    """
    kwargs = {"reid_weights": REID_WEIGHTS, "device": device, "half": False} if name in USES_APPEARANCE else {}
    return TRACKER_CLASSES[name](**kwargs)


def iter_frames(source: Path) -> Iterator[Tuple[int, np.ndarray]]:
    """Duyệt frame của thư mục ảnh ``img1/`` hoặc một file video.

    Args:
        source: Thư mục chứa ảnh ``.jpg`` đặt tên tăng dần, hoặc file ``.mp4``.

    Yields:
        Cặp ``(frame_index, frame_bgr)``. ``frame_index`` bắt đầu từ 0.

    Raises:
        FileNotFoundError: Khi thư mục không có ảnh ``.jpg``, hoặc không mở được file video.
    """
    if source.is_dir():
        frame_paths = sorted(source.glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"Không tìm thấy ảnh .jpg trong {source}")
        for i, path in enumerate(frame_paths):
            yield i, cv2.imread(str(path))
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Không mở được video {source}")
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
        cap.release()


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Sinh một màu BGR ổn định cho mỗi track ID.

    Args:
        track_id: Định danh track. Cùng ID luôn ra cùng màu.

    Returns:
        Bộ ba ``(B, G, R)``, mỗi kênh trong khoảng 64–254.
    """
    rng = np.random.default_rng(track_id * 9973 + 17)
    return tuple(int(c) for c in rng.integers(64, 255, size=3))


def detect(detector: YOLO, frame: np.ndarray, conf: float, iou: float, device: str) -> np.ndarray:
    """Chạy detector và trả về hộp người trên một frame.

    Args:
        detector: Mô hình YOLO đã nạp.
        frame: Ảnh BGR.
        conf: Ngưỡng confidence của detector.
        iou: Ngưỡng IoU cho NMS của detector.
        device: Thiết bị thực thi detector.

    Returns:
        Mảng ``(N, 6)`` với mỗi hàng là ``[x1, y1, x2, y2, conf, cls]``.
        Mảng rỗng ``(0, 6)`` khi không có hộp.
    """
    results = detector.predict(
        frame,
        conf=conf,
        iou=iou,
        imgsz=IMG_SIZE,
        classes=[PERSON_CLASS_ID],
        device=device,
        verbose=False,
    )[0]
    if results.boxes is None or len(results.boxes) == 0:
        return np.empty((0, 6))
    xyxy = results.boxes.xyxy.cpu().numpy()
    conf_arr = results.boxes.conf.cpu().numpy()
    cls_arr = results.boxes.cls.cpu().numpy()
    return np.hstack([xyxy, conf_arr[:, None], cls_arr[:, None]])


def run(args: argparse.Namespace) -> None:
    """Detect, track, rồi ghi file kết quả và video xem thử nếu được yêu cầu.

    Args:
        args: Tham số đã parse, gồm ``source``, ``seq_name``, ``tracker``,
            ``conf``, ``iou``, ``device``, ``out``, ``save_video``, ``fps``
            và ``max_frames``.
    """
    source = Path(args.source)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch chưa nhận GPU CUDA; hãy cài bản CUDA trước khi chạy bài nộp.")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mot_txt = out_dir / f"{args.seq_name}.txt"

    print(f"[nạp mô hình] detector={DETECTOR_WEIGHTS} imgsz={IMG_SIZE} tracker={args.tracker}")
    if args.tracker in USES_APPEARANCE:
        print(f"              tracker này dùng Re-ID: {REID_WEIGHTS.name} (tự tải nếu chưa có)")
    detector = YOLO(DETECTOR_WEIGHTS)
    tracker = make_tracker(args.tracker, args.device)

    writer = None
    rows = []
    n_frames = 0
    t0 = time.time()

    for frame_idx, frame in iter_frames(source):
        if frame is None:
            continue
        n_frames += 1

        dets = detect(detector, frame, conf=args.conf, iou=args.iou, device=args.device)
        tracks = tracker.update(dets, frame)

        for track in tracks:
            x1, y1, x2, y2, tid = track[0], track[1], track[2], track[3], track[4]
            tconf = track[5]
            rows.append(
                f"{frame_idx + 1},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},"
                f"{y2 - y1:.2f},{tconf:.4f},-1,-1,-1"
            )

        if args.save_video:
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                preview_path = out_dir / f"{args.seq_name}_preview.mp4"
                writer = cv2.VideoWriter(str(preview_path), fourcc, args.fps, (width, height))
            vis = frame.copy()
            for track in tracks:
                x1, y1, x2, y2, tid = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
                color = color_for_id(tid)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis, f"ID {tid}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )
            writer.write(vis)

        if args.max_frames and n_frames >= args.max_frames:
            break

    if writer is not None:
        writer.release()

    mot_txt.write_text("\n".join(rows) + ("\n" if rows else ""))

    dt = time.time() - t0
    fps = n_frames / dt if dt > 0 else 0.0
    print(
        f"\n[{args.seq_name}] tracker={args.tracker} conf={args.conf} iou={args.iou} "
        f"-> {n_frames} frame trong {dt:.1f}s ({fps:.1f} FPS)"
    )
    print(f"  File nộp bài: {mot_txt}")
    if args.save_video:
        print(f"  Video xem thử: {out_dir / f'{args.seq_name}_preview.mp4'}")


def parse_args() -> argparse.Namespace:
    """Khai báo tham số dòng lệnh.

    Returns:
        Namespace chứa các tham số đã parse.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Thư mục img1/ hoặc file .mp4")
    parser.add_argument("--seq-name", required=True, help="Tên video, ví dụ video_1 — dùng làm tên file .txt")
    parser.add_argument("--tracker", required=True, choices=TRACKER_CHOICES)
    parser.add_argument("--conf", type=float, default=0.3, help="Ngưỡng confidence của detector")
    parser.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU NMS của detector")
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda:0', ...")
    parser.add_argument("--out", default="runs", help="Thư mục xuất kết quả")
    parser.add_argument("--save-video", action="store_true", help="Xuất video xem thử")
    parser.add_argument("--fps", type=int, default=20, help="FPS của video xem thử")
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Giới hạn số frame (0 = toàn bộ). Dùng khi thử nhanh; bản nộp phải bỏ tham số này.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
