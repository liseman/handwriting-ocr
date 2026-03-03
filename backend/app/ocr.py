"""
OCR inference pipeline for handwriting OCR.

Primary engine: Google Gemini Flash (multimodal LLM) — dramatically better
accuracy on camera photos of handwriting, handles rotation natively.
Fallback engine: TrOCR-large-handwritten with optional LoRA adapters.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from pillow_heif import register_heif_opener
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Enable HEIC/HEIF support in Pillow
register_heif_opener()

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


def preprocess_image(image_path: str | Path, rotation: int = 0) -> Image.Image:
    """Load an image from disk, convert to RGB, and apply manual rotation."""
    img = Image.open(image_path)
    # EXIF rotation is now baked into the file at upload time,
    # so we only need to handle manual user rotation here.
    if img.mode != "RGB":
        img = img.convert("RGB")
    if rotation:
        # PIL rotate is counter-clockwise; user rotation is clockwise
        img = img.rotate(-rotation, expand=True)
    return img


def binarize(image: Image.Image) -> Image.Image:
    """Adaptive binarization for cleaner OCR input.

    Converts to grayscale, applies a local-mean adaptive threshold
    (approximated with a large-radius Gaussian blur), then binarizes.
    Uses morphological opening to remove noise from camera photos.
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
    # minus offset.  Use a generous offset (25) so camera noise, paper
    # texture, and shadows don't get classified as ink.
    offset = 25.0
    binary = ((gray_arr < blur_arr - offset) * 255).astype(np.uint8)

    # Morphological opening (erosion → dilation) removes small noise specks
    # while preserving text strokes.  Use a 3×3 kernel.
    from PIL import ImageMorph

    binary_img = Image.fromarray(binary, mode="L")

    # Use min/max filters as a simpler morphological open.
    # Erosion = min filter, Dilation = max filter.
    kernel_size = max(3, min(image.width, image.height) // 500)
    if kernel_size % 2 == 0:
        kernel_size += 1
    binary_img = binary_img.filter(ImageFilter.MinFilter(kernel_size))
    binary_img = binary_img.filter(ImageFilter.MaxFilter(kernel_size))

    return binary_img


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

    # Fallback: if we have a single segment covering most of the page,
    # the binarization likely failed to find gaps (common on camera photos).
    # Split into estimated line-height strips so TrOCR gets manageable chunks.
    if len(results) == 1:
        seg_img, seg_bbox = results[0]
        seg_height = seg_bbox[3]
        if seg_height > image.height * 0.6:
            # Estimate line height as ~3-4% of page height (typical for
            # handwriting on letter-size paper photographed at full res).
            est_line_height = max(int(image.height * 0.035), 40)
            results = []
            y = 0
            while y < image.height:
                y1 = min(y + est_line_height, image.height)
                bbox = (0, y, image.width, y1 - y)
                line_img = image.crop((0, y, image.width, y1))
                results.append((line_img, bbox))
                y = y1

    return results


def detect_content_bounds(
    image: Image.Image, padding_percent: float = 0.03,
) -> tuple[int, int, int, int] | None:
    """Detect the bounding box of actual content (ink) in an image.

    Uses horizontal and vertical projection profiles on a binarised copy to
    find where ink exists, trimming empty margins/background.

    Returns ``(x, y, w, h)`` with padding, or ``None`` if the detected
    region covers almost the entire image (no meaningful crop).
    """
    binary = binarize(image)
    arr = np.array(binary, dtype=np.float32) / 255.0

    h_proj = arr.sum(axis=1)         # per-row ink
    v_proj = arr.sum(axis=0)         # per-column ink

    h_thresh = max(h_proj.max() * 0.01, 1.0)
    v_thresh = max(v_proj.max() * 0.01, 1.0)

    rows = np.where(h_proj >= h_thresh)[0]
    cols = np.where(v_proj >= v_thresh)[0]

    if len(rows) == 0 or len(cols) == 0:
        return None  # no content detected

    y_min, y_max = int(rows[0]), int(rows[-1])
    x_min, x_max = int(cols[0]), int(cols[-1])

    pad_y = int(image.height * padding_percent)
    pad_x = int(image.width * padding_percent)

    x = max(0, x_min - pad_x)
    y = max(0, y_min - pad_y)
    x2 = min(image.width, x_max + pad_x)
    y2 = min(image.height, y_max + pad_y)
    w = x2 - x
    h = y2 - y

    # Skip if the crop covers >90% of the image — not a meaningful crop.
    if w * h > image.width * image.height * 0.90:
        return None

    return (x, y, w, h)


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

    def process_page(
        self,
        image_path: str | Path,
        rotation: int = 0,
        crop: dict | None = None,
    ) -> list[OcrSegment]:
        """Full pipeline: load image, segment into lines, run OCR.

        Parameters
        ----------
        crop : dict | None
            Optional ``{"x": int, "y": int, "w": int, "h": int}`` crop region
            applied *after* rotation.  Returned bbox coords are offset so they
            align with the full (rotated) image.

        Returns a list of :class:`OcrSegment` instances, one per detected
        text line.
        """
        image = preprocess_image(image_path, rotation=rotation)

        # Apply crop region if provided.
        crop_offset_x = 0
        crop_offset_y = 0
        if crop:
            cx, cy, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
            image = image.crop((cx, cy, cx + cw, cy + ch))
            crop_offset_x = cx
            crop_offset_y = cy

        lines = segment_lines(image)

        segments: list[OcrSegment] = []
        for line_img, bbox in lines:
            text, confidence = self.process_single(line_img)
            if text:  # skip empty detections
                # Offset bbox coords by crop origin so overlays align with full image.
                bx, by, bw, bh = bbox
                segments.append(OcrSegment(
                    text=text,
                    confidence=confidence,
                    bbox=(bx + crop_offset_x, by + crop_offset_y, bw, bh),
                ))

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


# ---------------------------------------------------------------------------
# Gemini Flash OCR engine (primary, high-quality)
# ---------------------------------------------------------------------------

GEMINI_ROTATION_PROMPT = """Look at this image of handwritten text. The text might be rotated.

What clockwise rotation in degrees (0, 90, 180, or 270) would make the text read normally (upright, left-to-right)?

Reply with ONLY a single number: 0, 90, 180, or 270"""

GEMINI_OCR_PROMPT = """Transcribe ALL handwritten text in this image line by line.

CRITICAL: For each line, provide the EXACT bounding box that tightly fits the ink of that specific line. Each line has a DIFFERENT y position and DIFFERENT height based on where the ink actually is. Do NOT use uniform spacing or a grid.

Return a JSON array. Each element: {"text": "...", "box": [y1, x1, y2, x2]}
Coordinates are normalized 0-1000 (0=top/left edge, 1000=bottom/right edge).

- y1 = top of the tallest ascender on that line
- y2 = bottom of the lowest descender on that line
- x1 = left edge of the first character
- x2 = right edge of the last character

Output ONLY the JSON array. If no text: return []"""

GEMINI_PAGE_CORNERS_PROMPT = """This is a camera photo. Is there a paper page or notebook visible with non-paper background (desk, table, hands, objects) around it?

If YES — the paper does NOT fill the entire image and background is clearly visible on at least 2 sides — return the 4 corners of the paper as JSON:
{"top_left": [x, y], "top_right": [x, y], "bottom_right": [x, y], "bottom_left": [x, y]}
Coordinates are normalized 0-1000 (0=top/left, 1000=bottom/right).

If NO — the paper fills nearly all of the image, or no clear paper boundary exists — return exactly: null

Important: only return corners if there is significant non-paper background visible (more than a thin border). A notebook photo that fills 90%+ of the frame should return null.
Output ONLY valid JSON (the object or null), nothing else."""


def perspective_warp_page(
    image_path: str, corners: list[tuple[int, int]],
) -> str | None:
    """Apply a perspective warp to extract the page quadrilateral as a rectangle.

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.
    corners : list[tuple[int, int]]
        Four corner points [top_left, top_right, bottom_right, bottom_left]
        in pixel coordinates.

    Returns the new image path on success, or None if the warp would produce
    a too-small image (< 100px in either dimension).
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        logger.warning("perspective_warp_page: could not read %s", image_path)
        return None

    tl, tr, br, bl = corners

    # Compute output dimensions from the max width/height of the quadrilateral.
    width_top = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    width_bot = math.hypot(br[0] - bl[0], br[1] - bl[1])
    out_w = int(max(width_top, width_bot))

    height_left = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    height_right = math.hypot(br[0] - tr[0], br[1] - tr[1])
    out_h = int(max(height_left, height_right))

    if out_w < 100 or out_h < 100:
        logger.warning(
            "perspective_warp_page: output too small (%dx%d), skipping",
            out_w, out_h,
        )
        return None

    src_pts = np.float32([tl, tr, br, bl])
    dst_pts = np.float32([
        [0, 0], [out_w - 1, 0],
        [out_w - 1, out_h - 1], [0, out_h - 1],
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (out_w, out_h))

    # Save to a new UUID filename (busts browser cache).
    directory = os.path.dirname(image_path)
    ext = os.path.splitext(image_path)[1] or ".jpg"
    new_path = os.path.join(directory, f"{uuid.uuid4().hex}{ext}")
    cv2.imwrite(new_path, warped)

    # Remove old file.
    try:
        os.remove(image_path)
    except OSError:
        pass

    logger.info(
        "perspective_warp_page: %s → %s (%dx%d)",
        image_path, new_path, out_w, out_h,
    )
    return new_path


def deskew_page(image_path: str) -> str | None:
    """Detect and correct small text/page skew using Hough line detection.

    Uses probabilistic Hough transform to find near-horizontal lines
    (text baselines, ruled notebook lines) and computes the median angle.
    Corrects skew up to ±10°; skips if < 0.5° (already straight enough).

    Returns the new image path, or None if no correction was needed.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection for Hough lines.
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines — long minimum length to catch ruled lines/text baselines.
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100,
        minLineLength=w // 8,
        maxLineGap=20,
    )

    if lines is None:
        logger.info("deskew_page: no Hough lines found, skipping")
        return None

    # Collect angles of near-horizontal lines (within ±15° of horizontal).
    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if abs(a) < 15:
            angles.append(a)

    if len(angles) < 5:
        logger.info("deskew_page: only %d near-horizontal lines, skipping", len(angles))
        return None

    # Use median angle as the skew estimate (robust to outliers).
    angle = float(np.median(angles))

    if abs(angle) < 0.5:
        logger.info("deskew_page: skew %.2f° too small, skipping", angle)
        return None
    if abs(angle) > 10:
        logger.warning("deskew_page: skew %.2f° too large, skipping", angle)
        return None

    # Rotate by -angle to correct the skew.
    # cv2.getRotationMatrix2D positive angle = counter-clockwise.
    correction = -angle
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, correction, 1.0)

    # Expand canvas so rotated image isn't clipped.
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Save to new UUID filename.
    directory = os.path.dirname(image_path)
    ext = os.path.splitext(image_path)[1] or ".jpg"
    new_path = os.path.join(directory, f"{uuid.uuid4().hex}{ext}")
    cv2.imwrite(new_path, rotated)

    try:
        os.remove(image_path)
    except OSError:
        pass

    logger.info("deskew_page: corrected %.2f° — %s → %s", angle, image_path, new_path)
    return new_path


GEMINI_SINGLE_PROMPT = """You are an expert handwriting OCR system. This image shows a cropped region of handwritten text.

Transcribe ALL the handwritten text in this image precisely.
Capture every word, punctuation mark, and number.
Return ONLY the transcribed text, nothing else.
If no text is visible, respond with EMPTY."""


@dataclass
class GeminiOcrResult:
    """Result from Gemini OCR on a full page."""
    rotation: int
    segments: list[OcrSegment]


class GeminiOcrEngine:
    """OCR engine using Google Gemini Flash multimodal API (new google-genai SDK)."""

    def __init__(self) -> None:
        from google import genai

        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = "gemini-2.5-flash"
        logger.info("Gemini OCR engine initialized (%s)", self.model_name)

    def _call(self, prompt: str, image: Image.Image, max_tokens: int = 16384, temperature: float = 0.0) -> str:
        """Send a prompt + image to Gemini and return the text response."""
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        text = response.text
        if text is None:
            logger.warning("Gemini returned None text, finish_reason=%s",
                           response.candidates[0].finish_reason if response.candidates else "N/A")
            return ""
        return text.strip()

    def detect_rotation(self, image: Image.Image) -> int:
        """Ask Gemini what rotation is needed to make text upright.

        Uses thinking disabled + temperature 0 for consistent results.
        """
        from google.genai import types

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[GEMINI_ROTATION_PROMPT, image],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=16,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw = (response.text or "").strip()
            for token in raw.split():
                token = token.strip("°.,")
                if token in ("0", "90", "180", "270"):
                    return int(token)
            logger.warning("Could not parse rotation from Gemini: %r", raw)
            return 0
        except Exception:
            logger.exception("Gemini rotation detection failed")
            return 0

    def detect_page_corners(
        self, image: Image.Image,
    ) -> list[tuple[int, int]] | None:
        """Ask Gemini to detect the 4 corners of the paper page.

        Returns a list of 4 (x, y) pixel-coordinate tuples
        [top_left, top_right, bottom_right, bottom_left], or None if
        corners can't be detected or the image is already just the page.
        """
        import json as _json
        from google.genai import types

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[GEMINI_PAGE_CORNERS_PROMPT, image],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            raw = (response.text or "").strip()
        except Exception:
            logger.exception("Gemini page corner detection failed")
            return None

        logger.info("Gemini page corners raw: %.300s", raw)

        if raw.lower() == "null" or not raw:
            return None

        # Strip markdown fences if present.
        json_str = raw
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            data = _json.loads(json_str)
        except _json.JSONDecodeError:
            logger.warning("Could not parse page corners JSON: %r", raw)
            return None

        if data is None:
            return None

        img_w, img_h = image.size
        try:
            corners = []
            for key in ("top_left", "top_right", "bottom_right", "bottom_left"):
                pt = data[key]
                px = int(pt[0] / 1000.0 * img_w)
                py = int(pt[1] / 1000.0 * img_h)
                px = max(0, min(px, img_w - 1))
                py = max(0, min(py, img_h - 1))
                corners.append((px, py))
            return corners
        except (KeyError, TypeError, IndexError, ValueError):
            logger.warning("Invalid page corners structure: %r", data)
            return None

    def process_page(
        self,
        image_path: str | Path,
        rotation: int = 0,
        crop: dict | None = None,
    ) -> GeminiOcrResult:
        """Send full page image to Gemini for OCR.

        Uses Gemini for text recognition (high accuracy) and CV-based
        line detection for bounding boxes (pixel-precise alignment).

        Strategy: Gemini provides accurate text but unreliable bounding
        boxes (tends to return a uniform grid). So we use Gemini text
        paired with CV-detected line positions (with ruled-line removal
        to handle notebook paper).
        """

        image = preprocess_image(image_path, rotation=rotation)

        crop_offset_x = 0
        crop_offset_y = 0
        if crop:
            cx, cy, cw, ch = crop["x"], crop["y"], crop["w"], crop["h"]
            image = image.crop((cx, cy, cx + cw, cy + ch))
            crop_offset_x = cx
            crop_offset_y = cy

        img_width, img_height = image.size

        # ── Step 1: Get text from Gemini ─────────────────────────────
        try:
            raw_text = self._call(GEMINI_OCR_PROMPT, image, temperature=0.0)
            logger.info(
                "Gemini OCR raw response (%d chars): %.300s",
                len(raw_text), raw_text,
            )
        except Exception:
            logger.exception("Gemini OCR API call failed for %s", image_path)
            return GeminiOcrResult(rotation=0, segments=[])

        if not raw_text or raw_text.strip() == "[]":
            return GeminiOcrResult(rotation=0, segments=[])

        # Extract text lines from Gemini response.
        entries = self._parse_gemini_json(raw_text)
        if entries is not None:
            text_lines = [
                e.get("text", "").strip()
                for e in entries
                if isinstance(e, dict) and e.get("text", "").strip()
            ]
        else:
            # Plain text fallback — skip if it looks like broken JSON.
            stripped = raw_text.strip()
            if (stripped.startswith("[") or stripped.startswith("{")
                    or stripped.startswith("```")):
                logger.warning("Gemini returned unparseable JSON, returning empty")
                return GeminiOcrResult(rotation=0, segments=[])
            text_lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

        if not text_lines:
            return GeminiOcrResult(rotation=0, segments=[])

        # ── Step 2: Build bboxes with CV-corrected Gemini coordinates ──
        # Gemini's spacing is correct but the absolute y-position is
        # offset.  CV detects the first ink peak and computes the
        # shift needed to align boxes with actual text.
        segments = self._build_cv_aligned_segments(
            text_lines, entries, image, img_width, img_height,
            crop_offset_x, crop_offset_y,
        )

        logger.info(
            "Gemini OCR: %d text lines → %d segments for %s",
            len(text_lines), len(segments), image_path,
        )
        return GeminiOcrResult(rotation=0, segments=segments)

    @staticmethod
    def _build_cv_aligned_segments(
        text_lines: list[str],
        entries: list | None,
        image: Image.Image,
        img_width: int,
        img_height: int,
        crop_offset_x: int = 0,
        crop_offset_y: int = 0,
    ) -> list[OcrSegment]:
        """Build segments using Gemini boxes shifted by a CV-derived offset.

        Gemini returns accurate text and correctly-spaced bounding
        boxes, but the absolute y-position is typically offset from
        the real ink.  We detect the first ink peak near Gemini's
        first entry and compute a constant vertical shift that
        brings all boxes into alignment.
        """
        import cv2

        n = len(text_lines)
        if n == 0:
            return []

        # ── Extract Gemini per-line coordinates ──────────────────────
        gemini_y_top: list[int | None] = []
        gemini_y_bot: list[int | None] = []
        per_line_x: list[tuple[int, int]] = []
        x_min_all, x_max_all = img_width, 0

        if entries is not None:
            for e in entries:
                if isinstance(e, dict) and isinstance(e.get("box"), list) and len(e["box"]) == 4:
                    y1, x1, y2, x2 = e["box"]
                    yt = int(y1 / 1000.0 * img_height)
                    yb = int(y2 / 1000.0 * img_height)
                    gemini_y_top.append(yt)
                    gemini_y_bot.append(yb)
                    px1 = max(0, int(x1 / 1000.0 * img_width))
                    px2 = min(img_width, int(x2 / 1000.0 * img_width))
                    per_line_x.append((px1, px2))
                    x_min_all = min(x_min_all, px1)
                    x_max_all = max(x_max_all, px2)
                else:
                    gemini_y_top.append(None)
                    gemini_y_bot.append(None)
                    per_line_x.append((0, img_width))

        # Pad if fewer coords than text lines.
        while len(gemini_y_top) < n:
            gemini_y_top.append(None)
            gemini_y_bot.append(None)
        while len(per_line_x) < n:
            per_line_x.append((0, img_width))

        # ── Compute y-offset from CV projection ──────────────────────
        # Build handwriting-only projection in the text column to find
        # where ink actually starts, then shift Gemini boxes to match.
        y_offset = 0
        first_gc = None
        for i in range(n):
            gt = gemini_y_top[i]
            gb = gemini_y_bot[i]
            if gt is not None and gb is not None:
                first_gc = (gt + gb) // 2
                break

        if first_gc is not None:
            # Text column bounds.
            if x_max_all <= x_min_all:
                x_min_all, x_max_all = 0, img_width
            pad_x = int(img_width * 0.03)
            x_col_min = max(0, x_min_all - pad_x)
            x_col_max = min(img_width, x_max_all + pad_x)

            gray = np.array(image.convert("L"))
            h, w = gray.shape

            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            line_len = max(w // 4, 100)
            horiz_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (line_len, 1),
            )
            ruled = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
            ruled = cv2.dilate(
                ruled,
                cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
                iterations=1,
            )
            handwriting = cv2.subtract(binary, ruled)

            proj = handwriting[:, x_col_min:x_col_max].astype(
                np.float32,
            ).sum(axis=1) / 255.0
            proj = cv2.GaussianBlur(
                proj.reshape(-1, 1), (1, 19), 3,
            ).flatten()

            threshold = max(proj.max() * 0.10, 1.0)

            # Search for the nearest strong peak to the first Gemini
            # entry center.  Look within ±100px of that position.
            search_r = 100
            s_top = max(0, first_gc - search_r)
            s_bot = min(len(proj), first_gc + search_r)
            best_peak = first_gc
            best_dist = search_r + 1

            for j in range(max(1, s_top), min(len(proj) - 1, s_bot)):
                if (proj[j] > threshold
                        and proj[j] > proj[j - 1]
                        and proj[j] >= proj[j + 1]):
                    d = abs(j - first_gc)
                    if d < best_dist:
                        best_dist = d
                        best_peak = j

            y_offset = best_peak - first_gc
            logger.info(
                "CV offset: first_peak=%d, gemini_center=%d, "
                "offset=%d, text_col x=%d-%d",
                best_peak, first_gc, y_offset, x_col_min, x_col_max,
            )
        else:
            h = img_height

        # ── Build segments with offset Gemini coordinates ────────────
        segments: list[OcrSegment] = []
        avg_line_h = max(img_height // max(n, 1), 30)

        for i, text in enumerate(text_lines):
            lx1, lx2 = per_line_x[i]

            gt = gemini_y_top[i]
            gb = gemini_y_bot[i]
            if gt is not None and gb is not None:
                by = gt + y_offset
                bh = gb - gt
            else:
                by = avg_line_h * i
                bh = avg_line_h

            # Clamp to image bounds.
            by = max(0, min(by, img_height - 1))
            bh = max(20, min(bh, img_height - by))

            # Horizontal extent from Gemini + padding.
            hpad = max(5, img_width // 200)
            bx = max(0, lx1 - hpad)
            bw = min(img_width, lx2 + hpad) - bx

            segments.append(OcrSegment(
                text=text,
                confidence=0.95,
                bbox=(
                    bx + crop_offset_x,
                    by + crop_offset_y,
                    bw,
                    bh,
                ),
            ))

        return segments

    @staticmethod
    def _parse_gemini_json(raw_text: str) -> list | None:
        """Parse Gemini JSON response, stripping markdown fences. Returns None on failure."""
        import json as _json
        import re

        json_str = raw_text.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_str = "\n".join(lines)

        try:
            entries = _json.loads(json_str)
        except _json.JSONDecodeError:
            # Try to extract a JSON array from the response (Gemini sometimes
            # wraps it in extra text or has trailing commas).
            match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if match:
                try:
                    # Remove trailing commas before ] which is invalid JSON
                    cleaned = re.sub(r',\s*([}\]])', r'\1', match.group())
                    entries = _json.loads(cleaned)
                    if isinstance(entries, list):
                        return entries
                except _json.JSONDecodeError:
                    pass
            logger.warning("Failed to parse Gemini JSON response")
            return None

        if not isinstance(entries, list):
            return None
        return entries

    def process_single(self, image: Image.Image) -> tuple[str, float]:
        """Run OCR on a single cropped image region.

        Returns (text, confidence).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            text = self._call(GEMINI_SINGLE_PROMPT, image, max_tokens=1024)
            if text == "EMPTY":
                return "", 0.0
            return text, 0.95
        except Exception:
            logger.exception("Gemini single OCR failed")
            return "", 0.0


# Singleton Gemini engine
_gemini_engine: Optional[GeminiOcrEngine] = None
_gemini_lock = threading.Lock()


def has_gemini() -> bool:
    """Return True if a Gemini API key is configured."""
    return bool(settings.GEMINI_API_KEY)


def get_gemini_engine() -> GeminiOcrEngine:
    """Return the singleton GeminiOcrEngine, creating it lazily."""
    global _gemini_engine
    if _gemini_engine is None:
        with _gemini_lock:
            if _gemini_engine is None:
                _gemini_engine = GeminiOcrEngine()
    return _gemini_engine
