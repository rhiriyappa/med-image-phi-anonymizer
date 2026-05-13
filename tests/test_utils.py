import numpy as np
import pytest
from PIL import Image

from src.utils import load_image, normalize_dicom, save_image


def test_normalize_full_range():
    arr = np.array([[0, 128, 255]], dtype=np.uint16)
    result = normalize_dicom(arr)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_normalize_flat_returns_zeros():
    arr = np.full((10, 10), 42.0, dtype=np.float32)
    result = normalize_dicom(arr)
    assert result.dtype == np.uint8
    assert result.max() == 0


def test_normalize_preserves_relative_order():
    arr = np.array([[0.0, 50.0, 100.0]], dtype=np.float32)
    result = normalize_dicom(arr)
    assert result[0, 0] < result[0, 1] < result[0, 2]


def test_load_image_rgb_png(tmp_path):
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    path = tmp_path / "test.png"
    img.save(path)

    arr, meta = load_image(str(path))

    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[2] == 3
    assert meta["source_path"] == str(path)


def test_load_image_grayscale_converted_to_rgb(tmp_path):
    img = Image.new("L", (32, 32), color=128)
    path = tmp_path / "gray.png"
    img.save(path)

    arr, _ = load_image(str(path))

    assert arr.shape[2] == 3


def test_load_image_rgba_converted_to_rgb(tmp_path):
    img = Image.new("RGBA", (32, 32), color=(10, 20, 30, 255))
    path = tmp_path / "rgba.png"
    img.save(path)

    arr, _ = load_image(str(path))

    assert arr.shape[2] == 3


def test_save_image_roundtrip(tmp_path):
    original = np.full((64, 64, 3), 123, dtype=np.uint8)
    path = tmp_path / "out.png"

    save_image(original, str(path))

    assert path.exists()
    loaded = np.array(Image.open(path))
    assert np.array_equal(original, loaded)


def test_load_missing_file_raises():
    with pytest.raises(Exception):
        load_image("/nonexistent/path/file.png")
