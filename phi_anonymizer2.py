"""
Main CLI entry point for the PHI Anonymization Pipeline.

Detects text-based PHI in medical images, anonymizes it via diffusion
inpainting, and embeds an invisible provenance watermark.
"""

import argparse
import logging
import sys

import numpy as np
from PIL import Image

from src import DiffusionAnonymizer, ImageWatermarker, PhiDetector
from src.utils import load_image, save_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generative PHI Anonymizer and Watermarker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input DICOM or image file.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to save the processed output image.")
    parser.add_argument("-p", "--prompt",
                        default="medical scan, x-ray, anatomy, tissue",
                        help="Prompt to guide diffusion inpainting.")
    parser.add_argument("-w", "--watermark", default="ANONYMIZED_FOR_RESEARCH",
                        help="Watermark text to embed.")
    parser.add_argument("--watermark-mode", choices=["invisible", "visible"],
                        default="invisible",
                        help="'invisible' (DWT-based) or 'visible' (Pillow text overlay).")
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Minimum OCR confidence to treat text as PHI (0–1).")
    parser.add_argument("--detect-watermark", action="store_true",
                        help="Detect a watermark in the input instead of anonymizing.")
    args = parser.parse_args()

    if args.detect_watermark:
        watermarker = ImageWatermarker(watermark_mode="invisible", auto_fallback=False)
        img_np, _ = load_image(args.input)
        expected_len = len(args.watermark.encode("utf-8"))
        watermarker.detect_watermark(img_np, length=expected_len)
        return

    try:
        img_np, metadata = load_image(args.input)
        img_pil = Image.fromarray(img_np)

        detector = PhiDetector(confidence_threshold=args.confidence_threshold)
        bboxes = detector.detect_text_phi(img_np)

        if not bboxes:
            logger.info("No PHI detected — skipping inpainting.")
            final_np = img_np
        else:
            anonymizer = DiffusionAnonymizer()
            mask_pil = anonymizer.create_mask(img_np.shape[:2], bboxes)
            inpainted_pil = anonymizer.inpaint_phi(img_pil, mask_pil, args.prompt)
            final_np = np.array(inpainted_pil)

        watermarker = ImageWatermarker(watermark_mode=args.watermark_mode)
        final_np = watermarker.embed_watermark(final_np, args.watermark)

        save_image(final_np, args.output)

        print("\n--- Pipeline Summary ---")
        print(f"Source:         {args.input}")
        print(f"Modality:       {metadata.get('Modality', 'N/A')}")
        print(f"PHI regions:    {len(bboxes)}")
        print(f"Prompt:         {args.prompt}")
        print(f"Watermark mode: {watermarker.watermark_mode}")
        print(f"Watermark text: {args.watermark}")
        print(f"Output:         {args.output}")
        print("\nSUCCESS — Review output for missed PHI or artifacts before use.")

    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
