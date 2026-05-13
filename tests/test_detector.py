import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_bbox(x1=10, y1=10, x2=50, y2=30):
    """Returns a 4-corner EasyOCR-style bbox."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _sample_image(h=100, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture
def detector():
    with patch("src.detector.easyocr.Reader") as mock_cls:
        mock_reader = MagicMock()
        mock_cls.return_value = mock_reader
        from src.detector import PhiDetector
        d = PhiDetector()
        d.reader = mock_reader
        return d


def test_returns_boxes_above_threshold(detector):
    detector.reader.readtext.return_value = [(_make_bbox(), "PATIENT", 0.95)]
    result = detector.detect_text_phi(_sample_image())
    assert len(result) == 1
    x_min, y_min, x_max, y_max = result[0]
    assert x_min < x_max and y_min < y_max


def test_filters_low_confidence(detector):
    detector.reader.readtext.return_value = [(_make_bbox(), "NOISE", 0.1)]
    result = detector.detect_text_phi(_sample_image())
    assert result == []


def test_custom_confidence_threshold():
    with patch("src.detector.easyocr.Reader") as mock_cls:
        mock_reader = MagicMock()
        mock_cls.return_value = mock_reader
        from src.detector import PhiDetector
        d = PhiDetector(confidence_threshold=0.8)
        d.reader.readtext.return_value = [(_make_bbox(), "TEXT", 0.5)]
        result = d.detect_text_phi(_sample_image())
    assert result == []


def test_padding_clamped_to_image_bounds(detector):
    # Box very near the top-left corner — padding must not go negative
    detector.reader.readtext.return_value = [(_make_bbox(0, 0, 3, 3), "X", 0.9)]
    result = detector.detect_text_phi(_sample_image())
    assert len(result) == 1
    x_min, y_min, x_max, y_max = result[0]
    assert x_min >= 0 and y_min >= 0
    assert x_max <= 200 and y_max <= 100


def test_raises_on_float_image(detector):
    bad = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="8-bit"):
        detector.detect_text_phi(bad)


def test_raises_on_grayscale_image(detector):
    bad = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.detect_text_phi(bad)


def test_empty_image_returns_no_bboxes(detector):
    detector.reader.readtext.return_value = []
    result = detector.detect_text_phi(_sample_image())
    assert result == []
