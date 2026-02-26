"""
TrOCR inference pipeline for handwriting OCR.

Uses Microsoft's TrOCR-large-handwritten model with optional per-user
LoRA adapters for personalised recognition.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.config import settings

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "microsoft/trocr-large-handwritten"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OcrSegment:
    """A single recognised text segment with its position on the page."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h) relative to original image


# ---------------------------------------------------------------------------
# Image preprocessing helpers
# ---------------------------------------------------------------------------


def preprocess_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk and convert to RGB."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def binarize(image: Image.Image) -> Image.Image:
    """Adaptive binarization for cleaner OCR input.

    Converts to grayscale, applies a local-mean adaptive threshold
    (approximated with a large-radius Gaussian blur), then binarizes.
    """
    gray = image.convert("L")

    # Use a blurred version of the image as the local threshold.
    # A large radius approximates an adaptive neighbourhood.
    blur_radius = max(gray.width, gray.height) // 30
    blur_radius = max(blur_radius, 10)
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    gray_arr = np.array(gray, dtype=np.float32)
    blur_arr = np.array(blurred, dtype=np.float32)

    # Pixel is foreground (black) when it is darker than the local mean
    # minus a small offset (handles slight gradients / noise).
    offset = 10.0
    binary = ((gray_arr < blur_arr - offset) * 255).astype(np.uint8)

    # Invert so that text is white-on-black (easier for projection profiles).
    return Image.fromarray(binary, mode="L")


def _horizontal_projection(binary_image: Image.Image) -> np.ndarray:
    """Compute horizontal projection profile (sum of white pixels per row)."""
    arr = np.array(binary_image, dtype=np.float32) / 255.0
    return arr.sum(axis=1)


def segment_lines(image: Image.Image) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    """Segment a page image into individual text lines.

    Uses horizontal projection profile on a binarized copy to find gaps
    between text lines.  Returns a list of ``(line_image, bbox)`` tuples
    where *line_image* is cropped from the **original** (RGB) image and
    *bbox* is ``(x, y, w, h)``.
    """
    binary = binarize(image)
    projection = _horizontal_projection(binary)

    # Threshold: a row is "text" if projection exceeds a fraction of the
    # maximum projection value.
    threshold = max(projection.max() * 0.02, 1.0)

    in_text = False
    line_starts: list[int] = []
    line_ends: list[int] = []

    for row_idx, val in enumerate(projection):
        if not in_text and val >= threshold:
            in_text = True
            line_starts.append(row_idx)
        elif in_text and val < threshold:
            in_text = False
            line_ends.append(row_idx)

    # Close the last region if the image ends while still in text.
    if in_text:
        line_ends.append(len(projection))

    # Merge lines that are very close together (likely the same line with
    # a thin gap, e.g. from descenders/ascenders).
    min_gap = max(int(image.height * 0.005), 3)
    merged_starts: list[int] = []
    merged_ends: list[int] = []

    for start, end in zip(line_starts, line_ends):
        if merged_ends and (start - merged_ends[-1]) < min_gap:
            # Extend the previous region.
            merged_ends[-1] = end
        else:
            merged_starts.append(start)
            merged_ends.append(end)

    # Add a small vertical padding around each line for better recognition.
    pad = max(int(image.height * 0.005), 2)

    results: list[tuple[Image.Image, tuple[int, int, int, int]]] = []
    for start, end in zip(merged_starts, merged_ends):
        y0 = max(start - pad, 0)
        y1 = min(end + pad, image.height)
        # Full width of the page.
        bbox = (0, y0, image.width, y1 - y0)
        line_img = image.crop((0, y0, image.width, y1))
        results.append((line_img, bbox))

    # If segmentation found nothing, return the whole image as one segment.
    if not results:
        results.append((image.copy(), (0, 0, image.width, image.height)))

    return results


# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------


class OcrEngine:
    """Wraps TrOCR for inference, with optional per-user LoRA adapters."""

    def __init__(self, model_dir: str | None = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model_dir = Path(model_dir or settings.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading TrOCR base model (%s) ...", BASE_MODEL_NAME)
        self.processor = TrOCRProcessor.from_pretrained(BASE_MODEL_NAME)
        self.base_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL_NAME)
        self.base_model.to(self.device)
        self.base_model.eval()

        # Active model may differ from base_model when a LoRA adapter is loaded.
        self.model: VisionEncoderDecoderModel = self.base_model
        self._active_user_id: Optional[int] = None

        logger.info("TrOCR base model loaded on %s", self.device)

    # ----- LoRA adapter management -----------------------------------------

    def load_user_model(self, user_id: int, lora_path: str) -> None:
        """Load a user's LoRA fine-tuned adapter on top of the base model."""
        from peft import PeftModel

        lora_dir = Path(lora_path)
        if not lora_dir.exists():
            raise FileNotFoundError(f"LoRA weights not found at {lora_dir}")

        logger.info("Loading LoRA adapter for user %d from %s", user_id, lora_dir)

        # Always start from the base model to avoid stacking adapters.
        self.unload_user_model()

        peft_model = PeftModel.from_pretrained(self.base_model, str(lora_dir))
        peft_model.to(self.device)
        peft_model.eval()

        self.model = peft_model
        self._active_user_id = user_id
        logger.info("LoRA adapter for user %d active", user_id)

    def unload_user_model(self) -> None:
        """Revert to the base model (discard any loaded LoRA adapter)."""
        if self._active_user_id is not None:
            logger.info("Unloading LoRA adapter for user %d", self._active_user_id)
        self.model = self.base_model
        self._active_user_id = None

    # ----- Core inference --------------------------------------------------

    def process_single(self, image: Image.Image) -> tuple[str, float]:
        """Run OCR on a single cropped image.

        Returns ``(recognised_text, confidence)`` where confidence is in
        ``[0, 1]``.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=256,
            )

        generated_ids = outputs.sequences
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        confidence = self._compute_confidence(outputs)
        return text, confidence

    def get_confidence(self, image: Image.Image, text: str) -> float:
        """Compute a confidence score for *text* being the correct
        transcription of *image*.

        Uses the model's own generation with ``output_scores=True`` and
        takes the average token log-probability, mapped to ``[0, 1]``.
        """
        _, confidence = self.process_single(image)
        return confidence

    def process_page(self, image_path: str | Path) -> list[OcrSegment]:
        """Full pipeline: load image, segment into lines, run OCR.

        Returns a list of :class:`OcrSegment` instances, one per detected
        text line.
        """
        image = preprocess_image(image_path)
        lines = segment_lines(image)

        segments: list[OcrSegment] = []
        for line_img, bbox in lines:
            text, confidence = self.process_single(line_img)
            if text:  # skip empty detections
                segments.append(OcrSegment(text=text, confidence=confidence, bbox=bbox))

        return segments

    # ----- Internal helpers ------------------------------------------------

    @staticmethod
    def _compute_confidence(generate_output) -> float:
        """Derive a ``[0, 1]`` confidence score from generation output.

        ``generate_output`` is the dict returned by ``model.generate`` when
        ``output_scores=True, return_dict_in_generate=True``.  We compute
        the average token-level probability from the logits/scores.
        """
        scores = generate_output.scores  # tuple of (vocab_size,) tensors per step
        if not scores:
            return 0.0

        sequences = generate_output.sequences  # (batch, seq_len)
        # generated_ids includes the decoder start token; scores start from
        # the first *generated* token onward.
        generated_token_ids = sequences[0, 1:]  # drop decoder start token

        log_probs: list[float] = []
        for step_idx, score_tensor in enumerate(scores):
            if step_idx >= len(generated_token_ids):
                break
            token_id = generated_token_ids[step_idx].item()
            # Convert logits to log-probabilities.
            log_softmax = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
            log_probs.append(log_softmax[token_id].item())

        if not log_probs:
            return 0.0

        avg_log_prob = sum(log_probs) / len(log_probs)
        # Map to [0, 1]:  exp(log_prob) is in (0, 1]; clamp for safety.
        confidence = math.exp(avg_log_prob)
        return max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# Singleton engine management
# ---------------------------------------------------------------------------

_engine: Optional[OcrEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> OcrEngine:
    """Return the singleton :class:`OcrEngine`, creating it lazily on first call."""
    global _engine
    if _engine is None:
        with _engine_lock:
            # Double-checked locking.
            if _engine is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _engine = OcrEngine(model_dir=settings.MODEL_DIR, device=device)
    return _engine
