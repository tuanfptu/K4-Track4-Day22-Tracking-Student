"""Kiểm tra bộ chọn tracker không cần tải mô hình."""

import pytest

from run_tracking import TRACKER_CHOICES, make_tracker


@pytest.mark.parametrize("name", ["bytetrack", "ocsort"])
def test_motion_tracker_can_be_created(name: str) -> None:
    """Hai tracker chuyển động phải khởi tạo bằng API BoxMOT hiện hành."""
    tracker = make_tracker(name, "cpu")
    assert hasattr(tracker, "update")


def test_invalid_tracker_is_rejected() -> None:
    """Tên ngoài danh sách lựa chọn phải bị từ chối."""
    with pytest.raises(KeyError):
        make_tracker("invalid", "cpu")
    assert "invalid" not in TRACKER_CHOICES
