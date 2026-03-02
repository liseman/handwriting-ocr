"""OCR processing routes -- trigger OCR on pages/documents, fetch results."""

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Document, OcrResult, Page, User, UserModel
from app.schemas import MessageResponse, OcrResultOut

router = APIRouter(prefix="/ocr", tags=["ocr"])


# ── OCR engine stub ──────────────────────────────────────────────────────────
# The actual OCR inference is implemented in a separate module.
# This function will be replaced once the model pipeline is ready.


async def _run_ocr_on_page(page_id: int, user_id: int) -> None:
    """Run OCR inference on a single page and persist results.

    Uses Gemini Flash API when available (much better quality on camera photos).
    Falls back to TrOCR pipeline when no Gemini API key is configured.

    Pipeline:
    1. Auto-rotate: Gemini detects orientation; TrOCR tries all 4 rotations.
    2. Auto-crop: detect content bounds and store as page crop fields.
    3. Run OCR (with crop if set).
    4. Index results in Whoosh for full-text search.
    """
    import asyncio
    import logging

    from app.database import async_session  # local import to avoid circulars
    from app.ocr import (
        detect_content_bounds, get_engine, get_gemini_engine,
        has_gemini, preprocess_image,
    )
    from app.routes.documents import _bake_rotation
    from app.routes.search import index_ocr_result

    logger = logging.getLogger(__name__)

    async with async_session() as db:
        # Mark page as processing.
        stmt = select(Page).where(Page.id == page_id)
        result = await db.execute(stmt)
        page = result.scalar_one_or_none()
        if page is None:
            return

        page.processing_status = "processing"
        await db.flush()
        await db.commit()

        doc_stmt = select(Document).where(Document.id == page.document_id)
        doc_result = await db.execute(doc_stmt)
        document = doc_result.scalar_one_or_none()
        if document is None:
            return

        model_stmt = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .order_by(UserModel.version.desc())
            .limit(1)
        )
        model_result = await db.execute(model_stmt)
        user_model = model_result.scalar_one_or_none()
        model_version = f"user-v{user_model.version}" if user_model else "base"

        try:
            if has_gemini():
                # ── Gemini Flash pipeline ──────────────────────────────
                gemini = get_gemini_engine()
                model_version = "gemini-flash"

                logger.info("Running Gemini OCR on page %d", page_id)

                # Auto-rotate: detect orientation and bake into file.
                # page.rotation != 0 means we already auto-rotated this
                # page — skip detection to prevent double-rotation.
                if page.rotation == 0:
                    from app.ocr import preprocess_image as _pi
                    raw_img = await asyncio.to_thread(_pi, page.image_path, 0)
                    detected_rot = await asyncio.to_thread(
                        gemini.detect_rotation, raw_img,
                    )
                    if detected_rot != 0:
                        logger.info(
                            "Auto-rotate page %d: Gemini detected %d°",
                            page_id, detected_rot,
                        )
                        new_path = _bake_rotation(page.image_path, detected_rot)
                        page.image_path = new_path
                        page.rotation = detected_rot  # flag: already rotated
                        await db.flush()

                # Process with rotation=0 — file is already correctly
                # oriented (either originally or after baking above).
                gemini_result = await asyncio.to_thread(
                    gemini.process_page,
                    page.image_path, 0,
                )

                segments = gemini_result.segments

            else:
                # ── TrOCR fallback pipeline ────────────────────────────
                engine = get_engine()

                if user_model and user_model.lora_path:
                    engine.load_user_model(user_id, user_model.lora_path)
                else:
                    engine.unload_user_model()

                # Auto-rotation: try all 4 orientations.
                best_segments = None
                best_avg_conf = -1.0
                best_rotation = 0

                for candidate_rot in [0, 90, 180, 270]:
                    try:
                        segs = await asyncio.to_thread(
                            engine.process_page,
                            page.image_path, candidate_rot,
                        )
                    except Exception:
                        logger.warning(
                            "Auto-rotate: rotation=%d failed on page %d",
                            candidate_rot, page_id, exc_info=True,
                        )
                        continue

                    if not segs:
                        continue

                    avg_conf = sum(s.confidence for s in segs) / len(segs)
                    logger.info(
                        "Auto-rotate page %d: rotation=%d  avg_conf=%.4f  segments=%d",
                        page_id, candidate_rot, avg_conf, len(segs),
                    )
                    if avg_conf > best_avg_conf:
                        best_avg_conf = avg_conf
                        best_segments = segs
                        best_rotation = candidate_rot

                logger.info(
                    "Auto-rotate page %d: winner rotation=%d (conf=%.4f)",
                    page_id, best_rotation, best_avg_conf,
                )

                if best_rotation != 0:
                    new_path = _bake_rotation(page.image_path, best_rotation)
                    page.image_path = new_path
                    page.rotation = 0
                    await db.flush()

                # Auto-crop.
                image = preprocess_image(page.image_path, rotation=0)
                bounds = detect_content_bounds(image)
                if bounds is not None:
                    page.crop_x, page.crop_y, page.crop_w, page.crop_h = bounds
                else:
                    page.crop_x = page.crop_y = page.crop_w = page.crop_h = None
                await db.flush()

                # Final OCR with crop.
                crop = None
                if page.crop_x is not None:
                    crop = {
                        "x": page.crop_x, "y": page.crop_y,
                        "w": page.crop_w, "h": page.crop_h,
                    }

                segments = await asyncio.to_thread(
                    engine.process_page,
                    page.image_path, 0, crop,
                )

            # ── Persist results ────────────────────────────────────
            for segment in segments:
                bbox_x, bbox_y, bbox_w, bbox_h = segment.bbox
                ocr_row = OcrResult(
                    page_id=page.id,
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_w=bbox_w,
                    bbox_h=bbox_h,
                    text=segment.text,
                    confidence=segment.confidence,
                    model_version=model_version,
                )
                db.add(ocr_row)
                await db.flush()

                try:
                    index_ocr_result(
                        ocr_result_id=ocr_row.id,
                        user_id=user_id,
                        page_id=page.id,
                        document_id=document.id,
                        text=segment.text,
                    )
                except Exception:
                    logger.warning(
                        "Failed to index OCR result %d", ocr_row.id, exc_info=True,
                    )

            page.processing_status = "done"

        except Exception:
            logger.exception("OCR processing failed for page %d", page_id)
            page.processing_status = "error"

        await db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _verify_page_ownership(
    page_id: int, user_id: int, db: AsyncSession
) -> Page:
    """Return the Page if it belongs to *user_id*, else raise 404."""
    stmt = (
        select(Page)
        .join(Document, Page.document_id == Document.id)
        .where(Page.id == page_id, Document.user_id == user_id)
    )
    result = await db.execute(stmt)
    page = result.scalar_one_or_none()
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")
    return page


async def _verify_document_ownership(
    document_id: int, user_id: int, db: AsyncSession
) -> Document:
    """Return the Document if it belongs to *user_id*, else raise 404."""
    stmt = (
        select(Document)
        .options(selectinload(Document.pages))
        .where(Document.id == document_id, Document.user_id == user_id)
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/process/{page_id}", response_model=MessageResponse)
async def process_page(
    page_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Trigger OCR processing on a single page.

    Processing runs as a background task so the response returns immediately.
    """
    page = await _verify_page_ownership(page_id, current_user.id, db)

    # Clear any previous OCR results for this page so we get a fresh run.
    old_results_stmt = select(OcrResult).where(OcrResult.page_id == page.id)
    old_results = await db.execute(old_results_stmt)
    for old in old_results.scalars():
        await db.delete(old)

    page.processing_status = "processing"
    await db.flush()

    background_tasks.add_task(_run_ocr_on_page, page.id, current_user.id)
    return MessageResponse(message=f"OCR processing started for page {page.id}")


@router.get("/results/{page_id}", response_model=List[OcrResultOut])
async def get_ocr_results(
    page_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    """Return all OCR results for a page."""
    await _verify_page_ownership(page_id, current_user.id, db)

    stmt = (
        select(OcrResult)
        .where(OcrResult.page_id == page_id)
        .order_by(OcrResult.bbox_y, OcrResult.bbox_x)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


class ProcessBboxRequest(BaseModel):
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


@router.post("/process-bbox/{page_id}", response_model=OcrResultOut)
async def process_bbox(
    page_id: int,
    body: ProcessBboxRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> OcrResult:
    """Run OCR on a user-drawn bounding box region and return the result synchronously."""
    from app.ocr import get_engine, get_gemini_engine, has_gemini, preprocess_image
    from app.routes.search import index_ocr_result
    import logging

    logger = logging.getLogger(__name__)

    page = await _verify_page_ownership(page_id, current_user.id, db)

    # Find document for search indexing.
    doc_stmt = select(Document).where(Document.id == page.document_id)
    doc_result = await db.execute(doc_stmt)
    document = doc_result.scalar_one_or_none()

    # Load the image with rotation + crop applied.
    image = preprocess_image(page.image_path, rotation=page.rotation)
    if page.crop_x is not None and page.crop_y is not None and page.crop_w is not None and page.crop_h is not None:
        image = image.crop((page.crop_x, page.crop_y, page.crop_x + page.crop_w, page.crop_y + page.crop_h))
        crop_x_off = page.crop_x
        crop_y_off = page.crop_y
    else:
        crop_x_off = 0
        crop_y_off = 0

    # Crop to the user-drawn bbox.
    bx = body.bbox_x - crop_x_off
    by = body.bbox_y - crop_y_off
    bbox_img = image.crop((bx, by, bx + body.bbox_w, by + body.bbox_h))

    # Check user model.
    model_stmt = (
        select(UserModel)
        .where(UserModel.user_id == current_user.id)
        .order_by(UserModel.version.desc())
        .limit(1)
    )
    model_result = await db.execute(model_stmt)
    user_model = model_result.scalar_one_or_none()

    if has_gemini():
        gemini = get_gemini_engine()
        text, confidence = gemini.process_single(bbox_img)
        model_version = "gemini-flash"
    else:
        model_version = f"user-v{user_model.version}" if user_model else "base"
        engine = get_engine()
        if user_model and user_model.lora_path:
            engine.load_user_model(current_user.id, user_model.lora_path)
        else:
            engine.unload_user_model()
        text, confidence = engine.process_single(bbox_img)

    ocr_row = OcrResult(
        page_id=page.id,
        bbox_x=body.bbox_x,
        bbox_y=body.bbox_y,
        bbox_w=body.bbox_w,
        bbox_h=body.bbox_h,
        text=text or "",
        confidence=confidence,
        model_version=model_version,
    )
    db.add(ocr_row)
    await db.flush()

    # Index in Whoosh.
    if document and text:
        try:
            index_ocr_result(
                ocr_result_id=ocr_row.id,
                user_id=current_user.id,
                page_id=page.id,
                document_id=document.id,
                text=text,
            )
        except Exception:
            logger.warning("Failed to index OCR result %d in search", ocr_row.id, exc_info=True)

    await db.commit()
    await db.refresh(ocr_row)
    return ocr_row


@router.post("/process-document/{document_id}", response_model=MessageResponse)
async def process_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Trigger OCR processing on every page in a document."""
    document = await _verify_document_ownership(document_id, current_user.id, db)

    if not document.pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no pages",
        )

    for page in document.pages:
        # Clear old results.
        old_stmt = select(OcrResult).where(OcrResult.page_id == page.id)
        old_results = await db.execute(old_stmt)
        for old in old_results.scalars():
            await db.delete(old)

        page.processing_status = "processing"
        background_tasks.add_task(_run_ocr_on_page, page.id, current_user.id)

    await db.flush()

    return MessageResponse(
        message=f"OCR processing started for {len(document.pages)} page(s) in document '{document.name}'"
    )


class ProcessingStatusItem(BaseModel):
    page_id: int
    document_id: int
    document_name: str
    status: str


class ProcessingStatusResponse(BaseModel):
    pages: list[ProcessingStatusItem]


@router.get("/processing-status", response_model=ProcessingStatusResponse)
async def processing_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ProcessingStatusResponse:
    """Return any pages currently being processed or recently completed for the user.

    Frontend polls this to show notifications when background OCR finishes.
    """
    stmt = (
        select(Page, Document.id, Document.name)
        .join(Document, Page.document_id == Document.id)
        .where(
            Document.user_id == current_user.id,
            Page.processing_status.in_(["processing", "done", "error"]),
        )
    )
    result = await db.execute(stmt)
    rows = result.all()

    items = []
    for page, doc_id, doc_name in rows:
        items.append(ProcessingStatusItem(
            page_id=page.id,
            document_id=doc_id,
            document_name=doc_name,
            status=page.processing_status or "idle",
        ))

        # Auto-clear "done" and "error" statuses after they've been read.
        if page.processing_status in ("done", "error"):
            page.processing_status = "idle"

    if any(p.processing_status == "idle" for p, _, _ in rows):
        await db.flush()

    return ProcessingStatusResponse(pages=items)
