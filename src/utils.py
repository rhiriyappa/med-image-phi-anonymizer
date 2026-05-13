"""
Utility Functions Module

Helper functions for image loading, saving, and DICOM processing.
"""

import logging

import cv2
import numpy as np
import pydicom
from PIL import Image

logger = logging.getLogger(__name__)


def normalize_dicom(pixel_array: np.ndarray) -> np.ndarray:
    """
    Normalizes a DICOM pixel array to 0-255 via min-max scaling.
    Real-world applications require proper Window/Leveling.

    Args:
        pixel_array: DICOM pixel array (typically float32 or uint16).

    Returns:
        Normalized 8-bit (uint8) NumPy array.
    """
    logger.info("Normalizing DICOM pixel data...")
    pixel_array = pixel_array.astype(np.float32)

    if pixel_array.max() == pixel_array.min():
        return np.zeros_like(pixel_array, dtype=np.uint8)

    normalized = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
    return (normalized * 255.0).astype(np.uint8)


def load_image(input_path: str) -> tuple[np.ndarray, dict]:
    """
    Loads an image from DICOM or standard formats (PNG, JPG, etc.).

    Args:
        input_path: Path to the input image file (.dcm, .png, .jpg, etc.).

    Returns:
        A tuple of (image_np, metadata_dict).
        image_np is always 8-bit, 3-channel RGB.
    """
    logger.info("Loading image from: %s", input_path)
    metadata: dict = {"source_path": input_path}

    if input_path.lower().endswith(".dcm"):
        ds = pydicom.dcmread(input_path)
        pixel_array = ds.pixel_array

        metadata["PatientID"] = str(ds.get("PatientID", "N/A"))
        metadata["Modality"] = str(ds.get("Modality", "N/A"))

        image_8bit = normalize_dicom(pixel_array)

        if image_8bit.ndim == 2:
            image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image_8bit

        return image_rgb, metadata

    image = Image.open(input_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image), metadata


def save_image(image_np: np.ndarray, output_path: str) -> None:
    """
    Saves a NumPy array to an image file.

    Args:
        image_np: NumPy array (uint8, RGB).
        output_path: Destination file path.
    """
    img_pil = Image.fromarray(image_np)
    img_pil.save(output_path)
    logger.info("Saved output to: %s", output_path)
