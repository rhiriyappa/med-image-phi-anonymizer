"""
Diffusion-Based Anonymization Module

Provides diffusion inpainting to fill detected PHI regions in medical images.
"""

import logging

import numpy as np
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

logger = logging.getLogger(__name__)


class DiffusionAnonymizer:
    """
    Uses a Stable Diffusion inpainting model to fill masked PHI regions.
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting"):
        """
        Initializes the inpainting pipeline.

        Args:
            model_id: Hugging Face model identifier for the inpainting model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing DiffusionAnonymizer on %s...", self.device)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipe = self.pipe.to(self.device)
        logger.info("Inpainting pipeline loaded.")

    def create_mask(
        self,
        image_shape: tuple[int, int],
        bboxes: list[tuple[int, int, int, int]],
    ) -> Image.Image:
        """
        Creates a white-on-black binary mask from bounding boxes.

        Args:
            image_shape: (height, width) of the original image.
            bboxes: List of (xmin, ymin, xmax, ymax) boxes.

        Returns:
            A PIL Image in 'L' (grayscale) mode; white = region to inpaint.
        """
        logger.info("Creating redaction mask for %d bounding box(es)...", len(bboxes))
        mask = Image.new("L", (image_shape[1], image_shape[0]), 0)
        draw = ImageDraw.Draw(mask)
        for x_min, y_min, x_max, y_max in bboxes:
            draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
        return mask

    def inpaint_phi(
        self,
        image_pil: Image.Image,
        mask_pil: Image.Image,
        prompt: str,
    ) -> Image.Image:
        """
        Runs diffusion inpainting on the masked regions.

        Args:
            image_pil: Original PIL Image (RGB).
            mask_pil: Mask PIL Image (grayscale L).
            prompt: Text prompt to guide the inpainting.

        Returns:
            Inpainted PIL Image at the original resolution.
        """
        logger.info("Running diffusion inpainting (prompt: '%s')...", prompt)
        original_size = image_pil.size

        # SD inpainting requires 512x512 input
        image_512 = image_pil.resize((512, 512), Image.Resampling.LANCZOS)
        mask_512 = mask_pil.resize((512, 512), Image.Resampling.NEAREST)

        negative_prompt = "text, letters, words, watermark, logo, signature, writing"

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                image=image_512,
                mask_image=mask_512,
                negative_prompt=negative_prompt,
                guidance_scale=7.5,
                num_inference_steps=30,
            )

        inpainted_512 = result.images[0]
        inpainted = inpainted_512.resize(original_size, Image.Resampling.LANCZOS)

        # Composite: keep original pixels outside mask, use inpainted inside
        original_np = np.array(image_pil)
        inpainted_np = np.array(inpainted)
        mask_np = np.array(mask_pil.convert("RGB")) / 255.0

        final_np = (original_np * (1 - mask_np) + inpainted_np * mask_np).astype(np.uint8)
        logger.info("Inpainting complete.")
        return Image.fromarray(final_np)
