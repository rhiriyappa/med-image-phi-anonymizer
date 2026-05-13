"""
Image Watermarking Module

Supports invisible DWT-based watermarks via imwatermark and visible text
overlays via Pillow.
"""

import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

try:
    from imwatermark import Watermark
    INVISIBLE_WM_AVAILABLE = True
except ImportError:
    Watermark = None  # type: ignore[assignment,misc]
    INVISIBLE_WM_AVAILABLE = False
    logger.debug("invisible-watermark not installed; only visible watermarks available.")


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Returns the best available font at the requested size."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",                        # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",            # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # RHEL/CentOS
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


class ImageWatermarker:
    """
    Embeds and detects watermarks in images.
    Supports invisible (DWT) and visible (Pillow) modes.
    """

    def __init__(
        self,
        wm_type: str = "dwt",
        watermark_mode: str = "invisible",
        auto_fallback: bool = True,
    ):
        """
        Args:
            wm_type: Algorithm for invisible watermarks (default: 'dwt').
            watermark_mode: 'invisible' for DWT-based or 'visible' for Pillow overlay.
            auto_fallback: If True, fall back to visible mode when invisible is unavailable.
        """
        self.wm_type = wm_type
        self.watermark_mode = watermark_mode.lower()

        if self.watermark_mode == "invisible":
            if not INVISIBLE_WM_AVAILABLE:
                if auto_fallback:
                    logger.warning(
                        "invisible-watermark not installed; falling back to visible mode. "
                        "Install with: pip install invisible-watermark"
                    )
                    self.watermark_mode = "visible"
                else:
                    raise ImportError(
                        "Invisible watermarking requires the invisible-watermark package. "
                        "Install with: pip install invisible-watermark\n"
                        "Or pass watermark_mode='visible' to use Pillow-based watermarks."
                    )

        logger.info("ImageWatermarker initialized (mode=%s).", self.watermark_mode)

    def embed_watermark(
        self,
        image_np: np.ndarray,
        watermark_text: str,
        position: tuple[int, int] | None = None,
        opacity: float = 0.5,
        font_size: int = 40,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Embeds a watermark into a NumPy image array.

        Args:
            image_np: 8-bit, 3-channel (RGB) NumPy array.
            watermark_text: Text to embed.
            position: (x, y) for visible watermarks; centered if None.
            opacity: Opacity for visible watermarks (0.0–1.0).
            font_size: Font size in pixels for visible watermarks.
            color: RGB color tuple for visible watermarks.

        Returns:
            New NumPy array with the embedded watermark.
        """
        if image_np.dtype != np.uint8:
            raise ValueError("Input image must be 8-bit (np.uint8).")

        if self.watermark_mode == "invisible":
            return self._embed_invisible(image_np, watermark_text)
        return self._embed_visible(image_np, watermark_text, position, opacity, font_size, color)

    def _embed_invisible(self, image_np: np.ndarray, watermark_text: str) -> np.ndarray:
        """Embeds an invisible DWT watermark."""
        logger.info("Embedding invisible watermark...")
        bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        wm = Watermark(self.wm_type)
        wm.embed(bgr, watermark_text.encode("utf-8"))
        result = cv2.cvtColor(wm.img_wm, cv2.COLOR_BGR2RGB)
        logger.info("Invisible watermark embedded.")
        return result

    def _embed_visible(
        self,
        image_np: np.ndarray,
        watermark_text: str,
        position: tuple[int, int] | None,
        opacity: float,
        font_size: int,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Embeds a visible text watermark using Pillow."""
        logger.info("Embedding visible watermark...")
        img_pil = Image.fromarray(image_np)
        overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = _load_font(font_size)
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if position is None:
            x = (img_pil.width - text_w) // 2
            y = (img_pil.height - text_h) // 2
        else:
            x, y = position

        draw.text((x, y), watermark_text, font=font, fill=(*color, int(255 * opacity)))

        result = Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")
        logger.info("Visible watermark embedded.")
        return np.array(result)

    def detect_watermark(self, image_np: np.ndarray, length: int) -> str:
        """
        Detects an invisible DWT watermark of a known byte length.

        Args:
            image_np: 8-bit, 3-channel (RGB) NumPy array.
            length: Expected watermark length in bytes.

        Returns:
            The decoded watermark string, or "" if not found.
        """
        if self.watermark_mode != "invisible":
            logger.warning("Watermark detection only works in invisible mode.")
            return ""
        if not INVISIBLE_WM_AVAILABLE:
            logger.error("invisible-watermark package required for detection.")
            return ""
        if image_np.dtype != np.uint8:
            raise ValueError("Input image must be 8-bit (np.uint8).")

        logger.info("Detecting invisible watermark (expected %d byte(s))...", length)
        bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        wm = Watermark(self.wm_type)
        detected_bytes = wm.detect(bgr, length)
        detected = detected_bytes.decode("utf-8", errors="ignore")
        logger.info("Detected watermark: '%s'", detected)
        return detected
