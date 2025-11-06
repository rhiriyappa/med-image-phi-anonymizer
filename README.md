### Project: Generative Anonymization and Watermarking for Medical Imaging

Goal: To protect Patient Health Information (PHI) in medical imaging workflows using diffusion-based redaction (inpainting) and robust generative watermarking.

#### 1. Problem Statement

Medical images (like DICOM files from X-Rays, CTs, and MRIs) often contain burned-in PHI, such as patient names, IDs, dates of birth, and clinic names. Traditional anonymization by drawing black boxes (redaction) is destructive, irreversible, and can obscure diagnostically relevant information. It also clearly signals that data has been removed.

#### 2. Proposed Solution

This project implements a multi-stage Python pipeline to "generatively" anonymize images:

PHI Detection: Automatically identify regions of text in the image using Optical Character Recognition (OCR).

Diffusion-Based Redaction: Instead of a black box, use a diffusion inpainting model (like Stable Diffusion Inpainting) to "fill in" the detected PHI regions with plausible, AI-generated image data that matches the surrounding anatomy. The result is a clean image that looks as if the text was never there.

Generative Watermarking: Embed an invisible, robust watermark (e.g., in the wavelet domain) into the anonymized image. This watermark can be used to track image provenance, verify integrity, or prove ownership, which is crucial when sharing data for research.

#### 3. Technical Architecture & Workflow

Load Image: The system ingests a DICOM (.dcm) file or standard image (.png, .jpg).

Pre-process: The image (potentially 16-bit DICOM data) is normalized and converted to an 8-bit, 3-channel (RGB) format suitable for deep learning models.

Detect PHI (PhiDetector): An OCR model (easyocr) scans the image and returns bounding boxes for all detected text.

Create Mask (DiffusionAnonymizer): These bounding boxes are used to create a binary (black and white) mask. White areas correspond to the PHI that needs to be redacted.

Inpaint PHI (DiffusionAnonymizer): The original image and the mask are passed to a diffusion inpainting pipeline (diffusers). The model is guided by a prompt (e.g., "x-ray, medical scan, bone, tissue") to fill the masked areas realistically.

Embed Watermark (ImageWatermarker): A unique, invisible watermark (e.g., "Anonymized by Clinic-X, 2025") is embedded into the inpainted image using a discrete wavelet transform (DWT) method.

Save Output: The final, anonymized, and watermarked image is saved to disk.

#### 4. Tech Stack

Python 3.8+

PyTorch: For deep learning operations.

Hugging Face diffusers: For the diffusion inpainting model.

transformers & accelerate: Required by diffusers.

easyocr: For robust, off-the-shelf text detection.

pydicom: To read and parse DICOM files.

opencv-python: For image processing (creating masks, color conversion).

imwatermark: For robust, invisible watermarking.

Pillow (PIL): For image object handling.

#### 5. Ethical Considerations & Limitations

*Model Failure*: This is not a foolproof medical device. The OCR may miss some text, or the inpainting model may create non-realistic artifacts.

*Human-in-the-Loop (HITL)*: *CRITICAL*. A human (e.g., a radiologist or technician) must review and validate the final anonymized image before it is used for any purpose, especially research or publication.

*Data Security*: The raw, non-anonymized images must be handled in a secure, HIPAA-compliant (or equivalent) environment. This script should be run on a secure, local machine, not a public cloud service.

*Inpainting Prompts*: The quality of the inpainted region depends on the text prompt. A prompt that is too generic or too specific may lead to poor results.
