"""Document management routes -- upload, camera capture, list, detail, delete."""

import base64
import os
import uuid
from typing import List

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Document, Page, User
from app.schemas import DocumentListItem, DocumentOut, MessageResponse

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}


def _user_upload_dir(user_id: int) -> str:
    """Return (and create if needed) the per-user upload directory."""
    path = os.path.join(settings.UPLOAD_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def _validate_image_extension(filename: str) -> str:
    """Return the lowered extension or raise 400."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return ext


async def _save_upload_file(upload: UploadFile, dest: str) -> None:
    """Stream an UploadFile to disk."""
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 256):  # 256 KiB
            await f.write(chunk)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("", response_model=List[DocumentListItem])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list:
    """List all documents belonging to the current user."""
    # Sub-query to count pages per document.
    page_count_sq = (
        select(Page.document_id, func.count(Page.id).label("page_count"))
        .group_by(Page.document_id)
        .subquery()
    )

    stmt = (
        select(
            Document.id,
            Document.user_id,
            Document.name,
            Document.source,
            Document.created_at,
            func.coalesce(page_count_sq.c.page_count, 0).label("page_count"),
        )
        .outerjoin(page_count_sq, Document.id == page_count_sq.c.document_id)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    return [
        DocumentListItem(
            id=row.id,
            user_id=row.user_id,
            name=row.name,
            source=row.source,
            created_at=row.created_at,
            page_count=row.page_count,
        )
        for row in rows
    ]


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    files: List[UploadFile] = File(...),
    name: str = Form(default=""),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    """Upload one or more image files and create a Document with Pages."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    # Default name = first filename without extension.
    doc_name = name.strip() or os.path.splitext(files[0].filename or "untitled")[0]

    document = Document(user_id=current_user.id, name=doc_name, source="upload")
    db.add(document)
    await db.flush()  # get document.id

    upload_dir = _user_upload_dir(current_user.id)

    for idx, upload_file in enumerate(files):
        ext = _validate_image_extension(upload_file.filename or "image.png")
        unique_name = f"{uuid.uuid4().hex}{ext}"
        dest = os.path.join(upload_dir, unique_name)
        await _save_upload_file(upload_file, dest)

        page = Page(
            document_id=document.id,
            image_path=dest,
            page_number=idx + 1,
        )
        db.add(page)

    await db.flush()

    # Re-fetch with pages eager-loaded.
    stmt = (
        select(Document)
        .options(selectinload(Document.pages))
        .where(Document.id == document.id)
    )
    result = await db.execute(stmt)
    return result.scalar_one()


@router.post("/camera", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def camera_capture(
    name: str = Form(default="Camera capture"),
    image_data: str = Form(..., description="Base64-encoded image (data URI or raw base64)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    """Accept a base64-encoded image captured from the device camera."""
    # Strip optional data-URI prefix: "data:image/png;base64,..."
    raw = image_data
    if "," in raw:
        header, raw = raw.split(",", 1)
    else:
        header = ""

    try:
        img_bytes = base64.b64decode(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data",
        ) from exc

    # Guess extension from data-URI header, default to .png.
    ext = ".png"
    if "jpeg" in header or "jpg" in header:
        ext = ".jpg"
    elif "webp" in header:
        ext = ".webp"

    upload_dir = _user_upload_dir(current_user.id)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(upload_dir, unique_name)

    async with aiofiles.open(dest, "wb") as f:
        await f.write(img_bytes)

    document = Document(user_id=current_user.id, name=name.strip() or "Camera capture", source="camera")
    db.add(document)
    await db.flush()

    page = Page(document_id=document.id, image_path=dest, page_number=1)
    db.add(page)
    await db.flush()

    stmt = (
        select(Document)
        .options(selectinload(Document.pages))
        .where(Document.id == document.id)
    )
    result = await db.execute(stmt)
    return result.scalar_one()


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Document:
    """Return a document with its pages and their OCR results."""
    from app.models import OcrResult  # local to avoid circular at module level

    stmt = (
        select(Document)
        .options(
            selectinload(Document.pages).selectinload(Page.ocr_results),
        )
        .where(Document.id == document_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return document


@router.delete("/{document_id}", response_model=MessageResponse)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Delete a document and its associated pages / OCR results.

    Image files are also removed from disk.
    """
    stmt = (
        select(Document)
        .options(selectinload(Document.pages))
        .where(Document.id == document_id, Document.user_id == current_user.id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Clean up image files.
    for page in document.pages:
        try:
            if os.path.isfile(page.image_path):
                os.remove(page.image_path)
        except OSError:
            pass  # best-effort deletion

    await db.delete(document)
    return MessageResponse(message="Document deleted")
