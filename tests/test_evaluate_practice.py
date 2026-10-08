"""Kiểm tra cấu hình chấm cho gói dữ liệu thực tế."""

from pathlib import Path

from evaluate_practice import _load_eval_config


def test_config_falls_back_to_seqinfo(tmp_path: Path) -> None:
    """Dùng benchmark riêng khi gói chỉ có seqinfo.ini."""
    video = tmp_path / "video_1"
    video.mkdir()
    (video / "seqinfo.ini").write_text("[Sequence]\nname=video_1\n", encoding="utf-8")
    assert _load_eval_config(tmp_path) == {"benchmark": "LAB21", "split": "train"}


def test_explicit_config_wins(tmp_path: Path) -> None:
    """Ưu tiên cấu hình giảng viên nếu có."""
    video = tmp_path / "video_1"
    video.mkdir()
    (video / "eval_config.json").write_text('{"benchmark":"CUSTOM"}', encoding="utf-8")
    assert _load_eval_config(tmp_path) == {"benchmark": "CUSTOM"}
