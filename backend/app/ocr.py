"""
OCR inference pipeline for handwriting OCR.

Primary engine: Google Gemini Flash (multimodal LLM) — dramatically better
accuracy on camera photos of handwriting, handles rotation natively.
Fallback engine: TrOCR-large-handwritten with optional LoRA adapters.
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
# Line position detection (used by Gemini engine for bbox placement)
# ---------------------------------------------------------------------------


def _smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing via convolution (no scipy dependency)."""
    size = int(sigma * 6) | 1  # ensure odd
    x = np.arange(size) - size // 2
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _find_line_positions(
    image: Image.Image, expected_lines: int,
) -> list[tuple[int, int, int, int]]:
    """Find text line bounding boxes using ink detection + projection peak finding.

    Uses paper-aware ink detection (handles spine removal, paper vs
    background separation) and smoothed horizontal projection profiles
    to locate individual text lines via peak detection.

    Pipeline:
    1. Paper-aware ink detection → clean binary mask
    2. Smoothed horizontal projection (adaptive sigma)
    3. Peak detection → one peak per text line
    4. Midpoint boundaries between adjacent peaks
    5. Per-line horizontal extent within detected page bounds

    Returns a list of ``(x, y, w, h)`` bounding boxes.
    """
    import cv2

    img_width, img_height = image.size

    if expected_lines <= 0:
        return []

    # Paper-aware ink detection — handles spine removal, noise, and
    # paper vs background separation.
    try:
        ink = _detect_ink_on_paper(image)
    except Exception:
        logger.warning("Ink detection failed, using even distribution")
        return _even_distribute(img_width, img_height, expected_lines)

    binary = (ink * 255).astype(np.uint8)
    if binary.sum() == 0:
        return _even_distribute(img_width, img_height, expected_lines)

    # Horizontal projection profile.
    proj = binary.astype(np.float32).sum(axis=1) / 255.0

    # Adaptive Gaussian sigma: small for dense pages (preserves individual
    # peaks), larger for sparse pages (smooths noise).
    sigma = max(3, min(15, 150 / max(expected_lines, 1)))
    ksize = int(sigma * 6) | 1
    smooth = cv2.GaussianBlur(proj.reshape(-1, 1), (1, ksize), 0).flatten()

    threshold = max(smooth.max() * 0.05, 0.5)
    text_rows = np.where(smooth > threshold)[0]
    if len(text_rows) == 0:
        return _even_distribute(img_width, img_height, expected_lines)

    text_extent = int(text_rows[-1]) - int(text_rows[0])
    min_dist = max(int(text_extent / (expected_lines * 1.5)), 15)

    # Peak detection: local maxima above threshold, separated by min_dist.
    peaks: list[int] = []
    for i in range(1, len(smooth) - 1):
        if (smooth[i] > threshold
                and smooth[i] > smooth[i - 1]
                and smooth[i] >= smooth[i + 1]):
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)

    if not peaks:
        return _even_distribute(img_width, img_height, expected_lines)

    # Filter very weak peaks (< 5% of strongest) — removes noise.
    peak_vals = [float(smooth[p]) for p in peaks]
    max_pv = max(peak_vals)
    peaks = [p for p, v in zip(peaks, peak_vals) if v >= max_pv * 0.05]

    if not peaks:
        return _even_distribute(img_width, img_height, expected_lines)

    # Use midpoints between adjacent peaks as line boundaries.
    # For each peak's y-strip, detect separate text regions (handles
    # two-page notebook photos where lines from both pages overlap in y).
    pad = max(int(img_height * 0.003), 3)
    # Gap must be >10% of image width to count as a page boundary
    # (spine gap is typically 200-600px on a 4032px-wide image).
    gap_thresh = img_width * 0.10
    min_region_width = img_width * 0.05  # ignore regions < 5% of image
    result: list[tuple[int, int, int, int]] = []

    for i, peak in enumerate(peaks):
        # Top boundary.
        if i == 0:
            if len(peaks) > 1:
                top = max(0, peak - (peaks[1] - peak) // 2)
            else:
                top = max(0, int(text_rows[0]))
        else:
            top = (peaks[i - 1] + peak) // 2

        # Bottom boundary.
        if i == len(peaks) - 1:
            if len(peaks) > 1:
                bot = min(img_height, peak + (peak - peaks[-2]) // 2)
            else:
                bot = min(img_height, int(text_rows[-1]))
        else:
            bot = (peak + peaks[i + 1]) // 2

        top = max(0, top - pad)
        bot = min(img_height, bot + pad)

        # Per-line horizontal extent — detect separate text regions
        # (e.g. left page text and right page text at the same y).
        strip = binary[top:bot, :]
        cpj = strip.sum(axis=0) / 255.0
        cth = max(cpj.max() * 0.05, 0.3)
        ic = np.where(cpj >= cth)[0]

        if len(ic) == 0:
            result.append((0, top, img_width, bot - top))
            continue

        # Split into separate regions at large horizontal gaps (spine).
        diffs = np.diff(ic)
        split_idx = np.where(diffs > gap_thresh)[0]

        if len(split_idx) > 0:
            boundaries = [0] + [si + 1 for si in split_idx] + [len(ic)]
            for j in range(len(boundaries) - 1):
                region = ic[boundaries[j] : boundaries[j + 1]]
                region_width = int(region[-1]) - int(region[0])
                if region_width < min_region_width:
                    continue
                x_left = max(int(region[0]) - pad, 0)
                x_right = min(int(region[-1]) + pad, img_width)
                result.append((x_left, top, x_right - x_left, bot - top))
        else:
            x_left = max(int(ic[0]) - pad, 0)
            x_right = min(int(ic[-1]) + pad, img_width)
            result.append((x_left, top, x_right - x_left, bot - top))

    logger.info(
        "Line detection: expected %d lines, found %d peaks → %d bboxes "
        "(sigma=%.1f)",
        expected_lines, len(peaks), len(result), sigma,
    )
    return result


def _detect_ink_on_paper(
    image: Image.Image, sigma_mult: float = 3.0,
) -> np.ndarray:
    """Detect ink pixels within the paper region of a camera photo.

    Parameters
    ----------
    sigma_mult : float
        Number of standard deviations below paper median to threshold ink.
        3.0 = strict (fewer pixels, cleaner), 2.0 = relaxed (more pixels).

    Returns a float32 array (0.0/1.0) where 1.0 = ink pixel on paper.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)

    # Find paper region: bright pixels (notebook paper is white/cream).
    paper_mask = gray > 170
    pm = Image.fromarray((paper_mask * 255).astype(np.uint8), mode="L")
    pm = pm.filter(ImageFilter.MaxFilter(21))  # morphological close
    pm = pm.filter(ImageFilter.MinFilter(21))
    paper_mask = np.array(pm) > 128

    if paper_mask.sum() < gray.size * 0.05:
        binary = binarize(image)
        return np.array(binary, dtype=np.float32) / 255.0

    paper_pixels = gray[paper_mask]
    paper_median = float(np.median(paper_pixels))
    paper_std = float(np.std(paper_pixels))
    ink_thresh = paper_median - sigma_mult * paper_std

    ink_mask = (gray < ink_thresh) & paper_mask

    ink_img = Image.fromarray((ink_mask * 255).astype(np.uint8), mode="L")
    ink_img = ink_img.filter(ImageFilter.MinFilter(3))
    ink_img = ink_img.filter(ImageFilter.MaxFilter(3))

    ink_arr = np.array(ink_img, dtype=np.float32) / 255.0

    # Remove narrow vertical features (notebook spine, edges).
    # Only remove groups narrower than ~50px — wide groups are text regions.
    if sigma_mult >= 2.5:
        img_h, img_w = ink_arr.shape
        vert_proj = ink_arr.sum(axis=0)
        spine_threshold = img_h * 0.03
        spine_cols = np.where(vert_proj > spine_threshold)[0]
        if len(spine_cols) > 0:
            groups: list[tuple[int, int]] = []
            start = int(spine_cols[0])
            for i in range(1, len(spine_cols)):
                if spine_cols[i] - spine_cols[i - 1] > 10:
                    groups.append((start, int(spine_cols[i - 1])))
                    start = int(spine_cols[i])
            groups.append((start, int(spine_cols[-1])))
            margin = 15
            max_spine_width = 50  # real spine/edges are thin
            for s, e in groups:
                if (e - s) > max_spine_width:
                    continue  # skip wide groups — they're text, not spine
                x0 = max(0, s - margin)
                x1 = min(img_w, e + margin + 1)
                ink_arr[:, x0:x1] = 0

    return ink_arr


def _even_distribute(
    img_width: int, img_height: int, n: int,
) -> list[tuple[int, int, int, int]]:
    """Distribute n lines evenly across the full image height."""
    if n <= 0:
        return []
    margin = int(img_height * 0.05)
    total = max(img_height - 2 * margin, 1)
    h = max(total // n, 10)
    return [(0, margin + i * h, img_width, h) for i in range(n)]


def _merge_to_count(
    regions: list[tuple[int, int]], target: int,
    img_height: int,
) -> list[tuple[int, int]]:
    """Reduce regions to target count by merging small gaps or dropping tiny regions.

    When gaps are small (likely parts of the same text block), merge.
    When gaps are large (different text blocks), drop the smallest regions.

    Returns list of (y, h) pairs.
    """
    merged = list(regions)
    max_merge_gap = img_height * 0.1  # don't merge across >10% of image height

    while len(merged) > target and len(merged) > 1:
        # Find smallest gap between adjacent regions.
        min_gap = float("inf")
        min_idx = 0
        for i in range(len(merged) - 1):
            gap = merged[i + 1][0] - merged[i][1]
            if gap < min_gap:
                min_gap = gap
                min_idx = i

        if min_gap <= max_merge_gap:
            # Merge across the small gap.
            new_region = (merged[min_idx][0], merged[min_idx + 1][1])
            merged = merged[:min_idx] + [new_region] + merged[min_idx + 2:]
        else:
            # Gap too large — drop the smallest region instead.
            smallest_idx = min(range(len(merged)), key=lambda i: merged[i][1] - merged[i][0])
            merged = merged[:smallest_idx] + merged[smallest_idx + 1:]

    pad = max(int(img_height * 0.003), 2)
    return [
        (max(s - pad, 0), min(e - s + 2 * pad, img_height - max(s - pad, 0)))
        for s, e in merged
    ]


def _split_to_count(
    regions: list[tuple[int, int]], target: int,
    img_height: int,
) -> list[tuple[int, int]]:
    """Split large regions to get closer to target count.

    Returns list of (y, h) pairs.
    """
    total_height = sum(e - s for s, e in regions)
    if total_height <= 0:
        margin = int(img_height * 0.05)
        total = max(img_height - 2 * margin, 1)
        h = max(total // target, 10)
        return [(margin + i * h, h) for i in range(target)]

    result: list[tuple[int, int]] = []
    remaining = target
    for idx, (start, end) in enumerate(regions):
        height = end - start
        if idx == len(regions) - 1:
            count = remaining
        else:
            count = max(1, round(height / total_height * target))
            count = min(count, remaining)
        remaining -= count

        sub_h = max(height // count, 10) if count > 0 else height
        for j in range(count):
            y = start + j * sub_h
            result.append((y, sub_h))

    return result


# ---------------------------------------------------------------------------
# Gemini Flash OCR engine (primary, high-quality)
# ---------------------------------------------------------------------------

GEMINI_ROTATION_PROMPT = """Look at this image of handwritten text. The text might be rotated.

What clockwise rotation in degrees (0, 90, 180, or 270) would make the text read normally (upright, left-to-right)?

Reply with ONLY a single number: 0, 90, 180, or 270"""

GEMINI_OCR_PROMPT = """You are an expert handwriting OCR system. Transcribe ALL the handwritten text in this image.

Rules:
- Output each line of handwritten text on its own line
- Be extremely precise — capture every word, number, and punctuation mark
- Preserve the line-by-line structure of the original handwriting
- Output ONLY the transcribed text, nothing else (no labels, no numbering)
- If no handwritten text is visible, respond with EMPTY"""

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

    def _call(self, prompt: str, image: Image.Image, max_tokens: int = 16384) -> str:
        """Send a prompt + image to Gemini and return the text response."""
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.1,
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

    def process_page(
        self,
        image_path: str | Path,
        rotation: int = 0,
        crop: dict | None = None,
    ) -> GeminiOcrResult:
        """Send full page image to Gemini for OCR.

        Uses Gemini for transcription, then matches text lines to actual
        line positions found via horizontal projection profiles.
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

        # Transcribe with Gemini (handles any orientation natively).
        try:
            raw_text = self._call(GEMINI_OCR_PROMPT, image)
            logger.info("Gemini OCR raw response (%d chars): %.300s", len(raw_text), raw_text)
        except Exception:
            logger.exception("Gemini OCR API call failed for %s", image_path)
            return GeminiOcrResult(rotation=0, segments=[])

        if raw_text == "EMPTY" or not raw_text:
            return GeminiOcrResult(rotation=0, segments=[])

        text_lines = [line for line in raw_text.split("\n") if line.strip()]
        if not text_lines:
            return GeminiOcrResult(rotation=0, segments=[])

        # Step 3: Find actual line positions via projection profiles.
        line_bboxes = _find_line_positions(image, len(text_lines))

        segments: list[OcrSegment] = []
        for i, text in enumerate(text_lines):
            if i < len(line_bboxes):
                bx, by, bw, bh = line_bboxes[i]
            else:
                # Fallback: put extra lines after the last detected position.
                last_y = line_bboxes[-1][1] + line_bboxes[-1][3] if line_bboxes else 0
                bx, by, bw, bh = 0, last_y, img_width, 30
            segments.append(OcrSegment(
                text=text.strip(),
                confidence=0.95,
                bbox=(
                    bx + crop_offset_x,
                    by + crop_offset_y,
                    bw,
                    bh,
                ),
            ))

        logger.info(
            "Gemini OCR: %d lines, %d bboxes for %s",
            len(text_lines), len(line_bboxes), image_path,
        )
        return GeminiOcrResult(rotation=0, segments=segments)

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
