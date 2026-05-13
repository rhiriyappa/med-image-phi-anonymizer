import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def image():
    return np.full((256, 256, 3), 128, dtype=np.uint8)


@pytest.fixture
def watermarker():
    from src.watermarker import ImageWatermarker
    return ImageWatermarker(watermark_mode="visible")


def test_visible_watermark_returns_same_shape(watermarker, image):
    result = watermarker.embed_watermark(image, "TEST")
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_visible_watermark_modifies_pixels(watermarker, image):
    result = watermarker.embed_watermark(image, "WATERMARK")
    assert not np.array_equal(result, image)


def test_visible_watermark_custom_position(watermarker, image):
    result = watermarker.embed_watermark(image, "POS", position=(10, 10))
    assert result.shape == image.shape


def test_raises_on_non_uint8_input(watermarker):
    bad = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="8-bit"):
        watermarker.embed_watermark(bad, "TEXT")


def test_fallback_to_visible_when_invisible_unavailable(image):
    with patch("src.watermarker.INVISIBLE_WM_AVAILABLE", False), \
         patch("src.watermarker.Watermark", None):
        from src.watermarker import ImageWatermarker
        wm = ImageWatermarker(watermark_mode="invisible", auto_fallback=True)
    assert wm.watermark_mode == "visible"


def test_no_fallback_raises_import_error():
    with patch("src.watermarker.INVISIBLE_WM_AVAILABLE", False), \
         patch("src.watermarker.Watermark", None):
        from src.watermarker import ImageWatermarker
        with pytest.raises(ImportError):
            ImageWatermarker(watermark_mode="invisible", auto_fallback=False)


def test_detect_watermark_wrong_mode_returns_empty(image):
    from src.watermarker import ImageWatermarker
    wm = ImageWatermarker(watermark_mode="visible")
    result = wm.detect_watermark(image, length=10)
    assert result == ""


def test_detect_watermark_raises_on_non_uint8():
    # INVISIBLE_WM_AVAILABLE is checked at call time, so the patch must stay active
    with patch("src.watermarker.INVISIBLE_WM_AVAILABLE", True), \
         patch("src.watermarker.Watermark", MagicMock()):
        from src.watermarker import ImageWatermarker
        wm = ImageWatermarker(watermark_mode="invisible", auto_fallback=False)
        bad = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="8-bit"):
            wm.detect_watermark(bad, length=5)
