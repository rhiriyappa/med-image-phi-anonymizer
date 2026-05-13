"""
PHI Detection Module

Text detection using EasyOCR to identify Protected Health Information (PHI)
regions in medical images.
"""

import logging

import numpy as np
import torch
import easyocr

logger = logging.getLogger(__name__)


class PhiDetector:
    """
    Detects text regions in an image using EasyOCR.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        confidence_threshold: float = 0.3,
    ):
        """
        Initializes the OCR reader.

        Args:
            languages: Language codes for OCR (default: ['en']).
            confidence_threshold: Minimum OCR confidence to include a detection (0–1).
        """
        if languages is None:
            languages = ["en"]
        self.confidence_threshold = confidence_threshold

        self.reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
        logger.info("PhiDetector (EasyOCR) initialized (threshold=%.2f).", confidence_threshold)

    def detect_text_phi(
        self, image_np: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """
        Detects text in a NumPy image array.

        Args:
            image_np: An 8-bit, 3-channel (RGB) NumPy array.

        Returns:
            A list of bounding boxes as (x_min, y_min, x_max, y_max).
        """
        if image_np.dtype != np.uint8:
            raise ValueError("Input image must be 8-bit (np.uint8).")
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError("Input image must be 3-channel RGB.")

        height, width = image_np.shape[:2]
        logger.info("Detecting PHI text regions...")

        results = self.reader.readtext(image_np)
        bboxes: list[tuple[int, int, int, int]] = []

        for bbox_coords, text, prob in results:
            if prob < self.confidence_threshold:
                continue

            tl, tr, br, bl = bbox_coords
            x_min = int(min(tl[0], bl[0]))
            y_min = int(min(tl[1], tr[1]))
            x_max = int(max(tr[0], br[0]))
            y_max = int(max(bl[1], br[1]))

            # Pad boxes to ensure full text coverage
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)

            bboxes.append((x_min, y_min, x_max, y_max))
            logger.info(
                "  Found: '%s' (conf=%.2f) at [%d, %d, %d, %d]",
                text, prob, x_min, y_min, x_max, y_max,
            )

        logger.info("Detected %d text region(s).", len(bboxes))
        return bboxes
