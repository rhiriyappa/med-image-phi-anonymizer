# Generative Anonymization and Watermarking for Medical Imaging

A Python pipeline that detects and removes burned-in PHI (Protected Health Information) from medical images using diffusion-based inpainting, then embeds an invisible provenance watermark.

> **This is a proof-of-concept.** A human reviewer (radiologist or technician) must validate every output before clinical or research use.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Ethical Considerations & Limitations](#ethical-considerations--limitations)
- [References](#references)

---

## Problem Statement

Medical images (DICOM files from X-rays, CTs, MRIs) frequently contain burned-in PHI — patient names, IDs, dates of birth, and clinic names. Traditional black-box redaction is:

- **Destructive and irreversible** — permanently alters the image
- **Diagnostically disruptive** — may obscure anatomy near the text region
- **Visually obvious** — signals to any observer that data was removed

---

## Solution Overview

This project implements a three-stage pipeline:

1. **PHI Detection** — EasyOCR scans the image and returns bounding boxes for all text regions.
2. **Diffusion-Based Redaction** — A Stable Diffusion inpainting model fills detected text regions with anatomically plausible content guided by a text prompt (e.g. `"x-ray, medical scan, bone, tissue"`). The result looks as if the text was never there.
3. **Generative Watermarking** — An invisible DWT-based watermark (e.g. `"Anonymized by Clinic-X, 2025"`) is embedded to track provenance and verify integrity.

---

## Architecture

```
Input Image (DICOM / PNG / JPG)
        │
        ▼
┌───────────────┐
│  load_image() │  Normalize DICOM → 8-bit RGB
└──────┬────────┘
       │
       ▼
┌──────────────┐
│ PhiDetector  │  EasyOCR → bounding boxes of text regions
└──────┬───────┘
       │  bboxes
       ▼
┌─────────────────────┐
│ DiffusionAnonymizer │  Create binary mask → SD inpainting
└──────┬──────────────┘
       │  inpainted image
       ▼
┌─────────────────┐
│ ImageWatermarker│  Embed invisible DWT watermark
└──────┬──────────┘
       │
       ▼
  Output Image (PNG)
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `torch` | Deep learning runtime |
| `diffusers` | Stable Diffusion inpainting pipeline |
| `transformers` / `accelerate` | Required by diffusers |
| `easyocr` | Robust OCR for PHI detection |
| `pydicom` | Read and parse DICOM files |
| `opencv-python` | Image processing, color conversion |
| `invisible-watermark` | DWT-based invisible watermarking |
| `Pillow` | PIL image handling and visible watermarks |

---

## Project Structure

```
anonymize_watermark_medical_imaging/
├── src/
│   ├── __init__.py          # Package exports
│   ├── detector.py          # PhiDetector — OCR-based text detection
│   ├── anonymizer.py        # DiffusionAnonymizer — SD inpainting
│   ├── watermarker.py       # ImageWatermarker — invisible & visible watermarks
│   └── utils.py             # load_image, save_image, normalize_dicom
├── tests/
│   └── fixtures/            # Sample images for testing
├── phi_anonymizer2.py       # CLI entry point
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- At least 8 GB RAM (16 GB recommended for CPU-only inpainting)
- GPU optional but strongly recommended for inpainting speed

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd anonymize_watermark_medical_imaging

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> The first run will download the Stable Diffusion inpainting model (~5 GB) from Hugging Face.

---

## Usage

### Anonymize a medical image

```bash
uv run phi_anonymizer2.py \
    -i test/fixtures/test_xray.png \
    -o data/test_xray_anonymized.png \
    -w "ANONYMIZED" \
    --watermark-mode visible
```

### Specify a custom invisible watermark

```bash
python phi_anonymizer2.py \
    -i input.png \
    -o output.png \
    -w "ANONYMIZED_FOR_RESEARCH"
```

### Use a visible Pillow-based watermark instead

```bash
python phi_anonymizer2.py \
    -i input.png \
    -o output.png \
    -w "ANONYMIZED" \
    --watermark-mode visible
```

### Detect an embedded invisible watermark

```bash
python phi_anonymizer2.py \
    --input output.png \
    --output /dev/null \
    --watermark "ANONYMIZED_FOR_RESEARCH" \
    --detect-watermark
```

### All CLI options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | required | Path to input image or DICOM file |
| `--output` | `-o` | required | Path to save the processed image |
| `--prompt` | `-p` | `"medical scan, x-ray, anatomy, tissue"` | Inpainting guidance prompt |
| `--watermark` | `-w` | `"ANONYMIZED_FOR_RESEARCH"` | Watermark text to embed |
| `--watermark-mode` | | `invisible` | `invisible` (DWT) or `visible` (Pillow overlay) |
| `--detect-watermark` | | false | Detect rather than embed a watermark |

---

## Configuration

### Inpainting model

The default model is `runwayml/stable-diffusion-inpainting`. To use a different model, pass `model_id` when constructing `DiffusionAnonymizer` directly:

```python
from src import DiffusionAnonymizer
anonymizer = DiffusionAnonymizer(model_id="stabilityai/stable-diffusion-2-inpainting")
```

### OCR confidence threshold

The default confidence threshold for text detection is `0.3`. Lower values detect more text at the cost of more false positives. Configurable in `src/detector.py:54`.

---

## Ethical Considerations & Limitations

| Concern | Detail |
|---|---|
| **Model failure** | OCR may miss text; inpainting may produce artifacts near complex anatomy. |
| **Human-in-the-Loop** | **CRITICAL** — A qualified human must review every output before use in research or clinical settings. |
| **Data security** | Raw, non-anonymized images must remain in a secure, HIPAA-compliant (or equivalent) environment. Do not run this on a public cloud service with real patient data. |
| **Inpainting quality** | Output quality depends heavily on the inpainting prompt. Generic prompts may produce unrealistic fills. |
| **Watermark robustness** | DWT watermarks can survive moderate JPEG compression but are not forensically guaranteed. |
| **Not a medical device** | This tool has not been validated for regulatory compliance and must not be used as a standalone anonymization solution. |

---

## References

1. [Stable Diffusion Inpainting — AI by Group](https://medium.com/aibygroup/lets-understand-stable-diffusion-inpainting-fdd0b1c3a925)
2. [Radiology Masterclass — Chest X-ray samples](https://www.radiologymasterclass.co.uk/tutorials/chest/chest_system/chest_system_01#top_1st_img)
3. [NIH Chest X-rays dataset — Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
4. [Pseudo-PHI DICOM Data — Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/pseudo-phi-dicom-data/)
