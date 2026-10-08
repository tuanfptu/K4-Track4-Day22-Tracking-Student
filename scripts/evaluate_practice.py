#!/usr/bin/env python
"""Chấm thử MỘT video luyện (video_1) để đọc HOTA / MOTA / IDF1.

Script chỉ nhận video_1. Bốn video còn lại không có nhãn trong gói học viên —
đánh giá chúng bằng mắt, ghi vào báo cáo.

Yêu cầu trước:
    git clone https://github.com/JonathonLuiten/TrackEval.git
    pip install -e TrackEval/

Ví dụ:
    python scripts/evaluate_practice.py \\
        --trackeval-root ~/TrackEval \\
        --lab-data-root "$LAB_DATA" \\
        --submission "$LAB_DATA/../runs/nop_bai/video_1.txt" \\
        --run-name nhom01_video1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PRACTICE_VIDEO = "video_1"


def _patch_numpy_aliases() -> None:
    """TrackEval còn gọi np.float / np.int (đã bỏ từ NumPy 1.24)."""
    import numpy as np

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]


def _load_eval_config(lab_data_root: Path) -> dict:
    """Đọc cấu hình chấm đi kèm nhãn video luyện.

    Args:
        lab_data_root: Thư mục lab_data giảng viên phát.

    Returns:
        Dict có khóa ``benchmark`` và có thể có ``split``.

    Raises:
        FileNotFoundError: Khi thiếu cả ``eval_config.json`` lẫn ``seqinfo.ini``.
    """
    config_path = lab_data_root / PRACTICE_VIDEO / "eval_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    seqinfo = lab_data_root / PRACTICE_VIDEO / "seqinfo.ini"
    if not seqinfo.exists():
        raise FileNotFoundError(f"Không thấy {config_path} hoặc {seqinfo}")
    # Gói bài tập hiện hành có seqinfo.ini nhưng không có eval_config.json.
    # Benchmark riêng tránh lẫn với dữ liệu TrackEval khác trên cùng máy.
    return {"benchmark": "LAB21", "split": "train"}


def stage(trackeval_root: Path, lab_data_root: Path, submission: Path, run_name: str, benchmark: str) -> None:
    """Copy nhãn video luyện và file nộp vào cây thư mục TrackEval.

    Args:
        trackeval_root: Thư mục gốc bản clone TrackEval.
        lab_data_root: Thư mục lab_data giảng viên phát.
        submission: File ``video_1.txt`` do ``run_tracking.py`` sinh ra.
        run_name: Tên lần chấm, dùng làm thư mục tracker.
        benchmark: Tên benchmark TrackEval, lấy từ ``eval_config.json``.

    Raises:
        FileNotFoundError: Khi thiếu nhãn, ``seqinfo.ini``, hoặc file nộp.
    """
    src = lab_data_root / PRACTICE_VIDEO
    gt_file = src / "gt" / "gt.txt"
    seqinfo = src / "seqinfo.ini"
    if not gt_file.exists() or not seqinfo.exists():
        raise FileNotFoundError(f"video luyện thiếu gt hoặc seqinfo trong {src}")
    if not submission.exists():
        raise FileNotFoundError(f"Không thấy file nộp {submission}")

    split_dir = f"{benchmark}-train"
    gt_dst = trackeval_root / "data" / "gt" / "mot_challenge" / split_dir / PRACTICE_VIDEO
    (gt_dst / "gt").mkdir(parents=True, exist_ok=True)
    shutil.copy(gt_file, gt_dst / "gt" / "gt.txt")
    shutil.copy(seqinfo, gt_dst / "seqinfo.ini")

    sub_dst = trackeval_root / "data" / "trackers" / "mot_challenge" / split_dir / run_name / "data"
    sub_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(submission, sub_dst / f"{PRACTICE_VIDEO}.txt")


def run_trackeval(trackeval_root: Path, run_name: str, benchmark: str, split: str) -> None:
    """Gọi script chấm của TrackEval, chỉ một video luyện.

    Args:
        trackeval_root: Thư mục gốc bản clone TrackEval.
        run_name: Tên lần chấm đã stage.
        benchmark: Tên benchmark TrackEval.
        split: Nhánh dữ liệu, thường là ``train``.

    Raises:
        subprocess.CalledProcessError: Khi TrackEval thoát với mã khác 0.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("trackeval_compat.py")),
        str(trackeval_root / "scripts" / "run_mot_challenge.py"),
        "--GT_FOLDER", str(trackeval_root / "data" / "gt" / "mot_challenge"),
        "--TRACKERS_FOLDER", str(trackeval_root / "data" / "trackers" / "mot_challenge"),
        "--BENCHMARK", benchmark,
        "--SPLIT_TO_EVAL", split,
        "--SEQ_INFO", PRACTICE_VIDEO,
        "--TRACKERS_TO_EVAL", run_name,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
    ]
    print("Đang chấm video luyện:\n  " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Chấm file ``video_1.txt`` và in HOTA, MOTA, IDF1.

    Raises:
        SystemExit: Khi file nộp không phải ``video_1.txt``.
    """
    _patch_numpy_aliases()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trackeval-root", required=True, type=Path)
    parser.add_argument("--lab-data-root", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path, help="File video_1.txt do run_tracking.py sinh ra")
    parser.add_argument("--run-name", required=True, help="Tên lần chấm, ví dụ nhom01_video1")
    args = parser.parse_args()

    if args.submission.name != f"{PRACTICE_VIDEO}.txt":
        raise SystemExit(
            f"Script này chỉ chấm {PRACTICE_VIDEO}.txt. "
            "video_2 đến video_5 không có nhãn — hãy đánh giá bằng mắt."
        )

    config = _load_eval_config(args.lab_data_root)
    benchmark = config["benchmark"]
    split = config.get("split", "train")
    stage(args.trackeval_root, args.lab_data_root, args.submission, args.run_name, benchmark)
    run_trackeval(args.trackeval_root, args.run_name, benchmark, split)
    print(
        "\nĐọc bảng phía trên: HOTA cân bằng phát hiện và giữ danh tính; "
        "MOTA phạt số lần đổi ID; IDF1 nhạy với track dài bị gán sai ID. "
        "Dùng các số này cho video_1 trong báo cáo. Không có bảng số cho video khác."
    )


if __name__ == "__main__":
    main()
