import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


@pytest.fixture
def anonymizer():
    with patch("src.anonymizer.StableDiffusionInpaintPipeline.from_pretrained") as mock:
        mock.return_value = MagicMock()
        from src.anonymizer import DiffusionAnonymizer
        return DiffusionAnonymizer()


def test_create_mask_size_matches_image(anonymizer):
    mask = anonymizer.create_mask((300, 400), [(10, 20, 50, 60)])
    assert mask.size == (400, 300)  # PIL size is (width, height)
    assert mask.mode == "L"


def test_create_mask_box_is_white(anonymizer):
    mask = anonymizer.create_mask((100, 100), [(10, 10, 50, 50)])
    arr = np.array(mask)
    assert arr[30, 30] == 255  # inside box
    assert arr[0, 0] == 0      # outside box


def test_create_mask_empty_bboxes_all_black(anonymizer):
    mask = anonymizer.create_mask((100, 100), [])
    assert np.array(mask).max() == 0


def test_create_mask_multiple_boxes(anonymizer):
    bboxes = [(5, 5, 20, 20), (60, 60, 90, 90)]
    mask = anonymizer.create_mask((100, 100), bboxes)
    arr = np.array(mask)
    assert arr[12, 12] == 255   # inside first box
    assert arr[75, 75] == 255   # inside second box
    assert arr[40, 40] == 0     # between boxes


def test_inpaint_phi_output_shape(anonymizer):
    """Verifies that inpainting returns an image at the original resolution."""
    mock_result = MagicMock()
    mock_result.images = [Image.new("RGB", (512, 512), color=(50, 50, 50))]
    anonymizer.pipe.return_value = mock_result

    img = Image.new("RGB", (200, 150), color=(100, 100, 100))
    mask = Image.new("L", (200, 150), 0)

    result = anonymizer.inpaint_phi(img, mask, "x-ray")

    assert isinstance(result, Image.Image)
    assert result.size == (200, 150)
