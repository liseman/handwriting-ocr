"""OCR processing routes -- trigger OCR on pages/documents, fetch results."""

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
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

    Uses the real TrOCR engine from ``app.ocr``.  The engine segments the
    page into lines and runs recognition on each line, optionally using
    the user's LoRA adapter if one has been trained.

    Results are also indexed in Whoosh for full-text search.
    """
    import logging

    from app.database import async_session  # local import to avoid circulars
    from app.ocr import get_engine
    from app.routes.search import index_ocr_result

    logger = logging.getLogger(__name__)

    async with async_session() as db:
        stmt = select(Page).where(Page.id == page_id)
        result = await db.execute(stmt)
        page = result.scalar_one_or_none()
        if page is None:
            return

        # Find the document to get the user_id for search indexing.
        doc_stmt = select(Document).where(Document.id == page.document_id)
        doc_result = await db.execute(doc_stmt)
        document = doc_result.scalar_one_or_none()
        if document is None:
            return

        # Check for the user's latest fine-tuned model (if any).
        model_stmt = (
            select(UserModel)
            .where(UserModel.user_id == user_id)
            .order_by(UserModel.version.desc())
            .limit(1)
        )
        model_result = await db.execute(model_stmt)
        user_model = model_result.scalar_one_or_none()

        model_version = f"user-v{user_model.version}" if user_model else "base"

        # Load the OCR engine and optionally attach the user's LoRA adapter.
        try:
            engine = get_engine()

            if user_model and user_model.lora_path:
                engine.load_user_model(user_id, user_model.lora_path)
            else:
                engine.unload_user_model()

            segments = engine.process_page(page.image_path)

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
                await db.flush()  # get the id

                # Index in Whoosh for full-text search.
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
                        "Failed to index OCR result %d in search", ocr_row.id, exc_info=True
                    )

        except Exception:
            logger.exception("OCR processing failed for page %d", page_id)

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

        background_tasks.add_task(_run_ocr_on_page, page.id, current_user.id)

    await db.flush()

    return MessageResponse(
        message=f"OCR processing started for {len(document.pages)} page(s) in document '{document.name}'"
    )
