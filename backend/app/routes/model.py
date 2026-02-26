"""Model management routes -- status, trigger training, export LoRA weights."""

import io
import os
import zipfile

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Correction, Document, OcrResult, Page, User, UserModel
from app.schemas import ModelStatusOut, TrainRequest, TrainResponse

router = APIRouter(prefix="/model", tags=["model"])


# ── Fine-tuning background task ──────────────────────────────────────────────


async def _run_finetuning(user_id: int, new_version: int, epochs: int, learning_rate: float) -> None:
    """Fine-tune the base OCR model using the user's accumulated corrections.

    Queries all Correction rows for this user, builds training data from
    the corresponding page image crops + corrected text, then delegates to
    the ``FineTuner`` from ``app.finetune``.
    """
    import logging

    from app.database import async_session
    from app.finetune import FineTuner

    logger = logging.getLogger(__name__)

    async with async_session() as db:
        # Gather all corrections for this user with image/bbox info.
        corr_stmt = (
            select(Correction, OcrResult, Page.image_path)
            .join(OcrResult, Correction.ocr_result_id == OcrResult.id)
            .join(Page, OcrResult.page_id == Page.id)
            .join(Document, Page.document_id == Document.id)
            .where(Document.user_id == user_id)
        )
        result = await db.execute(corr_stmt)
        rows = result.all()

        total_corrections = len(rows)
        if total_corrections == 0:
            logger.warning("No corrections found for user %d, skipping training", user_id)
            return

        # Build the correction dicts expected by FineTuner.prepare_training_data.
        corrections_data: list[dict] = []
        for correction, ocr_result, image_path in rows:
            corrections_data.append({
                "page_image_path": image_path,
                "bbox": {
                    "x": ocr_result.bbox_x,
                    "y": ocr_result.bbox_y,
                    "w": ocr_result.bbox_w,
                    "h": ocr_result.bbox_h,
                },
                "corrected_text": correction.corrected_text,
            })

        # Run fine-tuning (CPU/GPU-bound, runs synchronously).
        try:
            finetuner = FineTuner(model_dir=settings.MODEL_DIR)
            lora_path = finetuner.train(
                user_id=user_id,
                corrections=corrections_data,
                epochs=epochs,
                lr=learning_rate,
            )
        except Exception:
            logger.exception("Fine-tuning failed for user %d", user_id)
            return

        # Record the trained model in the database.
        user_model = UserModel(
            user_id=user_id,
            version=new_version,
            lora_path=lora_path,
            num_corrections_trained=total_corrections,
        )
        db.add(user_model)
        await db.commit()

        logger.info(
            "Fine-tuning complete for user %d: v%d with %d corrections",
            user_id, new_version, total_corrections,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_latest_model(user_id: int, db: AsyncSession) -> UserModel | None:
    stmt = (
        select(UserModel)
        .where(UserModel.user_id == user_id)
        .order_by(UserModel.version.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def _count_user_corrections(user_id: int, db: AsyncSession) -> int:
    stmt = (
        select(func.count(Correction.id))
        .join(OcrResult, Correction.ocr_result_id == OcrResult.id)
        .join(Page, OcrResult.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(Document.user_id == user_id)
    )
    result = await db.execute(stmt)
    return result.scalar_one()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/status", response_model=ModelStatusOut)
async def model_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ModelStatusOut:
    """Return information about the user's fine-tuned model."""
    latest = await _get_latest_model(current_user.id, db)
    total_corrections = await _count_user_corrections(current_user.id, db)

    if latest is None:
        return ModelStatusOut(
            has_model=False,
            total_corrections=total_corrections,
            corrections_since_last_train=total_corrections,
        )

    return ModelStatusOut(
        has_model=True,
        version=latest.version,
        num_corrections_trained=latest.num_corrections_trained,
        total_corrections=total_corrections,
        corrections_since_last_train=total_corrections - latest.num_corrections_trained,
        created_at=latest.created_at,
    )


@router.post("/train", response_model=TrainResponse)
async def train_model(
    body: TrainRequest = TrainRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TrainResponse:
    """Trigger fine-tuning with accumulated corrections.

    Training runs as a background task.
    """
    total_corrections = await _count_user_corrections(current_user.id, db)

    if total_corrections == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No corrections available for training.",
        )

    latest = await _get_latest_model(current_user.id, db)
    new_version = (latest.version + 1) if latest else 1

    # Check if there are new corrections since last training.
    trained_so_far = latest.num_corrections_trained if latest else 0
    if total_corrections <= trained_so_far:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No new corrections since the last training run.",
        )

    background_tasks.add_task(
        _run_finetuning,
        current_user.id,
        new_version,
        body.epochs,
        body.learning_rate,
    )

    return TrainResponse(
        message=f"Training v{new_version} started in background",
        version=new_version,
        num_corrections=total_corrections,
    )


@router.get("/export")
async def export_model(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Download the user's latest LoRA weights as a ZIP archive."""
    latest = await _get_latest_model(current_user.id, db)

    if latest is None or not latest.lora_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No fine-tuned model available for export.",
        )

    lora_dir = latest.lora_path
    if not os.path.isdir(lora_dir):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model files not found on disk.",
        )

    # Build an in-memory zip of the LoRA directory.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(lora_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, lora_dir)
                zf.write(full, arcname)
    buf.seek(0)

    zip_filename = f"lora_v{latest.version}.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )
