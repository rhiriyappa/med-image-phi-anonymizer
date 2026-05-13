"""
PHI Anonymizer Package

A modular package for detecting and anonymizing Protected Health Information (PHI)
in medical images using OCR, diffusion inpainting, and invisible watermarks.
"""

from .detector import PhiDetector
from .anonymizer import DiffusionAnonymizer
from .watermarker import ImageWatermarker
from .utils import load_image, save_image, normalize_dicom

__all__ = [
    'PhiDetector',
    'DiffusionAnonymizer',
    'ImageWatermarker',
    'load_image',
    'save_image',
    'normalize_dicom',
]

__version__ = '1.0.0'
__author__ = 'Ram Hiriyappa'
__description__ = 'A modular package for PHI anonymization and watermarking in medical images'
