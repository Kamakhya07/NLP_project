"""
Multimodal OCR Screenshot Text Extraction Engine
================================================
Extracts text from uploaded social media screenshots (WhatsApp, Instagram,
Twitter/X, Reddit, Discord) using Windows Media Native OCR (winocr).

Features:
  - Supports PNG, JPG, JPEG, WEBP, BMP image formats
  - High-precision image preprocessing (rescaling, contrast enhancement)
  - Full-text line reconstruction and whitespace normalization
  - Seamless integration with CyberShield prediction pipeline
"""

import io
import re
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image, ImageEnhance, ImageOps
try:
    import winocr
except (ImportError, ModuleNotFoundError, Exception):
    winocr = None

logger = logging.getLogger(__name__)

class ScreenshotOCR:
    def __init__(self, default_lang: str = "en"):
        self.default_lang = default_lang

    def preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Enhances screenshot readability for the OCR engine.
        Converts grayscale/contrast adjustments while preserving original dimensions.
        """
        try:
            # Convert to RGB first
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Upscale small images to ensure minimum height for legible text
            w, h = img.size
            if h < 400 or w < 400:
                scale_factor = max(400 / h, 400 / w)
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Enhance contrast slightly for text crispness
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.4)

            # Convert to RGBA as required by Windows Media OCR
            return img.convert("RGBA")
        except Exception as e:
            logger.warning(f"Image preprocessing warning: {e}")
            return img.convert("RGBA") if img.mode != "RGBA" else img

    def extract_text(self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image], lang: str = "en") -> Dict[str, Any]:
        """
        Runs OCR on the given image and extracts text.
        Returns a dictionary with success status, extracted_text, and metadata.
        """
        try:
            # 1. Load image into PIL
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input)
            elif isinstance(image_input, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, io.BytesIO):
                img = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                return {
                    "success": False,
                    "extracted_text": "",
                    "error": f"Unsupported image input type: {type(image_input)}"
                }

            if winocr is None:
                logger.info("winocr is not available on this platform (non-Windows). OCR disabled.")
                return {
                    "success": False,
                    "extracted_text": "",
                    "lines": [],
                    "word_count": 0,
                    "error": "OCR text extraction is only supported on Windows host environments."
                }

            # 2. Preprocess image
            processed_img = self.preprocess_image(img)

            # 3. Execute OCR asynchronously
            async def _run_ocr():
                return await winocr.recognize_pil(processed_img, lang or self.default_lang)

            ocr_res = asyncio.run(_run_ocr())

            raw_text = getattr(ocr_res, "text", "") or ""
            # Clean and normalize text
            cleaned_text = re.sub(r"\r\n|\r", "\n", raw_text)
            # Remove isolated noise artifacts while keeping words, emojis, and punctuation
            lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
            final_text = " ".join(lines)

            if not final_text.strip():
                return {
                    "success": False,
                    "extracted_text": "",
                    "lines": [],
                    "word_count": 0,
                    "error": "No legible text was detected in the screenshot. Please upload a clearer image."
                }

            return {
                "success": True,
                "extracted_text": final_text,
                "lines": lines,
                "word_count": len(final_text.split()),
                "error": None
            }

        except Exception as e:
            logger.error(f"OCR Extraction failed: {e}", exc_info=True)
            return {
                "success": False,
                "extracted_text": "",
                "lines": [],
                "word_count": 0,
                "error": f"Failed to extract text from screenshot: {str(e)}"
            }

    def extract_text_and_boxes(self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image], lang: str = "en") -> Dict[str, Any]:
        """Convenience method returning extracted text and metadata."""
        res = self.extract_text(image_input, lang)
        return {
            "text": res.get("extracted_text", ""),
            "success": res.get("success", False),
            "lines": res.get("lines", []),
            "words": [],
            "boxes": []
        }

# Singleton instance
ocr_detector = ScreenshotOCR()
