"""Authentication routes -- Google OAuth2 login / callback / me / logout."""

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    create_access_token,
    get_current_user,
    get_or_create_user,
    oauth,
)
from app.config import settings
from app.database import get_db
from app.models import User
from app.schemas import MessageResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    """Redirect the browser to Google's OAuth2 consent screen."""
    redirect_uri = str(request.url_for("auth_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback")
async def callback(request: Request, db: AsyncSession = Depends(get_db)) -> RedirectResponse:
    """Handle the OAuth2 callback from Google.

    Exchanges the authorisation code for tokens, upserts the user row,
    mints a JWT, and redirects back to the frontend with the token as a
    query parameter so the SPA can store it.
    """
    token_data = await oauth.google.authorize_access_token(request)

    # The id_token is already verified by authlib; extract user info.
    userinfo: dict = token_data.get("userinfo", {})
    if not userinfo:
        userinfo = await oauth.google.userinfo(token=token_data)

    google_id: str = userinfo["sub"]
    email: str = userinfo["email"]
    name: str = userinfo.get("name", email)
    picture: str | None = userinfo.get("picture")

    user = await get_or_create_user(db, google_id=google_id, email=email, name=name, picture_url=picture)
    await db.commit()

    # Store the Google access token in the session so we can call
    # Google Photos API on behalf of the user later.
    request.session["google_access_token"] = token_data.get("access_token")

    jwt_token = create_access_token(data={"sub": str(user.id)})

    # Redirect to frontend with the JWT in query params.
    params = urlencode({"token": jwt_token})
    return RedirectResponse(url=f"{settings.FRONTEND_URL}/auth/callback?{params}")


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> User:
    """Return the currently authenticated user's profile."""
    return current_user


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> MessageResponse:
    """Log out the current user.

    Since JWTs are stateless the client is responsible for discarding its
    token.  This endpoint clears any server-side session data (e.g. the
    cached Google access token) and returns a confirmation message.
    """
    request.session.clear()
    return MessageResponse(message="Logged out successfully")
