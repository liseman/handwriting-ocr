"""Google Photos integration -- list albums, list items, import into documents."""

import os
import uuid
from typing import List, Optional

import aiofiles
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Document, Page, User
from app.schemas import (
    DocumentOut,
    GoogleAlbum,
    GoogleMediaItem,
    MessageResponse,
    PhotoImportRequest,
    PhotoImportResponse,
)

router = APIRouter(prefix="/photos", tags=["photos"])

GOOGLE_PHOTOS_API = "https://photoslibrary.googleapis.com/v1"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_google_token(request: Request) -> str:
    """Extract the Google access token from the session.

    The token is stored during the OAuth callback so we can call Google
    APIs on behalf of the user without re-authenticating.
    """
    token: Optional[str] = request.session.get("google_access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google Photos access token not available. Please re-authenticate.",
        )
    return token


def _user_upload_dir(user_id: int) -> str:
    path = os.path.join(settings.UPLOAD_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


async def _download_image(url: str, dest: str) -> None:
    """Download an image from *url* and save it to *dest*."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        async with aiofiles.open(dest, "wb") as f:
            await f.write(resp.content)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/albums", response_model=List[GoogleAlbum])
async def list_albums(
    request: Request,
    page_size: int = Query(default=50, ge=1, le=50),
    page_token: Optional[str] = Query(default=None),
    _current_user: User = Depends(get_current_user),
) -> list:
    """List the authenticated user's Google Photos albums."""
    token = _get_google_token(request)

    params: dict = {"pageSize": page_size}
    if page_token:
        params["pageToken"] = page_token

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{GOOGLE_PHOTOS_API}/albums",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )

    if resp.status_code == 401:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google access token expired. Please re-authenticate.",
        )
    resp.raise_for_status()

    data = resp.json()
    albums_raw = data.get("albums", [])

    return [
        GoogleAlbum(
            id=a["id"],
            title=a.get("title", "Untitled"),
            media_items_count=int(a.get("mediaItemsCount", 0)) if a.get("mediaItemsCount") else None,
            cover_photo_url=a.get("coverPhotoBaseUrl"),
        )
        for a in albums_raw
    ]


@router.get("/albums/{album_id}/items", response_model=List[GoogleMediaItem])
async def list_album_items(
    album_id: str,
    request: Request,
    page_size: int = Query(default=50, ge=1, le=100),
    page_token: Optional[str] = Query(default=None),
    _current_user: User = Depends(get_current_user),
) -> list:
    """List media items in a specific Google Photos album."""
    token = _get_google_token(request)

    body: dict = {
        "albumId": album_id,
        "pageSize": page_size,
    }
    if page_token:
        body["pageToken"] = page_token

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GOOGLE_PHOTOS_API}/mediaItems:search",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
        )

    if resp.status_code == 401:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google access token expired. Please re-authenticate.",
        )
    resp.raise_for_status()

    data = resp.json()
    items_raw = data.get("mediaItems", [])

    return [
        GoogleMediaItem(
            id=item["id"],
            filename=item.get("filename", "unknown"),
            mime_type=item.get("mimeType", "image/jpeg"),
            base_url=item.get("baseUrl", ""),
            width=int(item["mediaMetadata"]["width"]) if "mediaMetadata" in item and "width" in item["mediaMetadata"] else None,
            height=int(item["mediaMetadata"]["height"]) if "mediaMetadata" in item and "height" in item["mediaMetadata"] else None,
        )
        for item in items_raw
        if item.get("mimeType", "").startswith("image/")
    ]


@router.post("/import", response_model=PhotoImportResponse)
async def import_photos(
    body: PhotoImportRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PhotoImportResponse:
    """Import selected photos (or an entire album) as a new Document.

    Either ``media_item_ids`` or ``album_id`` must be provided.  If
    ``album_id`` is given without ``media_item_ids``, all images in the
    album are imported.
    """
    token = _get_google_token(request)

    if not body.media_item_ids and not body.album_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either media_item_ids or album_id.",
        )

    items_to_download: list[dict] = []

    if body.media_item_ids:
        # Fetch metadata for specific items via batchGet.
        async with httpx.AsyncClient(timeout=30) as client:
            # The API accepts up to 50 item IDs per call.
            for i in range(0, len(body.media_item_ids), 50):
                chunk = body.media_item_ids[i : i + 50]
                params = [("mediaItemIds", mid) for mid in chunk]
                resp = await client.get(
                    f"{GOOGLE_PHOTOS_API}/mediaItems:batchGet",
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                )
                if resp.status_code == 401:
                    raise HTTPException(status_code=401, detail="Google token expired.")
                resp.raise_for_status()
                for result in resp.json().get("mediaItemResults", []):
                    item = result.get("mediaItem")
                    if item and item.get("mimeType", "").startswith("image/"):
                        items_to_download.append(item)

    elif body.album_id:
        # Iterate through the entire album.
        next_token: Optional[str] = None
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                req_body: dict = {"albumId": body.album_id, "pageSize": 100}
                if next_token:
                    req_body["pageToken"] = next_token
                resp = await client.post(
                    f"{GOOGLE_PHOTOS_API}/mediaItems:search",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json=req_body,
                )
                if resp.status_code == 401:
                    raise HTTPException(status_code=401, detail="Google token expired.")
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("mediaItems", []):
                    if item.get("mimeType", "").startswith("image/"):
                        items_to_download.append(item)
                next_token = data.get("nextPageToken")
                if not next_token:
                    break

    if not items_to_download:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No images found to import.",
        )

    # Create document.
    document = Document(
        user_id=current_user.id,
        name=body.document_name,
        source="google_photos",
    )
    db.add(document)
    await db.flush()

    upload_dir = _user_upload_dir(current_user.id)

    for idx, item in enumerate(items_to_download):
        base_url = item.get("baseUrl", "")
        # Append =d to get the full-resolution download.
        download_url = f"{base_url}=d"

        filename = item.get("filename", f"photo_{idx}.jpg")
        ext = os.path.splitext(filename)[1].lower() or ".jpg"
        unique_name = f"{uuid.uuid4().hex}{ext}"
        dest = os.path.join(upload_dir, unique_name)

        try:
            await _download_image(download_url, dest)
        except httpx.HTTPError:
            # Skip items that fail to download.
            continue

        page = Page(
            document_id=document.id,
            image_path=dest,
            page_number=idx + 1,
        )
        db.add(page)

    await db.flush()

    # Reload with pages.
    stmt = (
        select(Document)
        .options(selectinload(Document.pages))
        .where(Document.id == document.id)
    )
    result = await db.execute(stmt)
    doc = result.scalar_one()

    return PhotoImportResponse(
        document=DocumentOut.model_validate(doc),
        imported_count=len(doc.pages),
    )
