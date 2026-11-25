"""Opt-in Google OIDC sign-in for the Gradio web application.

The ordinary local email/password mode remains in ``app.py``. This module is
loaded only with ``--google-auth`` and deliberately keeps provider identity
separate from local accounts. Authlib validates the OIDC ID token (issuer,
audience, signature, time claims and nonce); the application additionally
requires a verified email and keys history by an HMAC of Google's ``sub``.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

LOG = logging.getLogger(__name__)
SESSION_COOKIE = "kt_google_session"
OAUTH_STATE_COOKIE = "kt_oauth_state"
SESSION_SECONDS = 12 * 60 * 60
STATE_SECONDS = 10 * 60
DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"


@dataclass(frozen=True)
class GoogleSettings:
    client_id: str
    client_secret: str
    public_url: str
    session_secret: str

    @property
    def secure_cookie(self) -> bool:
        return urlsplit(self.public_url).scheme == "https"

    @property
    def callback_url(self) -> str:
        return self.public_url.rstrip("/") + "/auth/callback"

    @classmethod
    def from_env(cls) -> "GoogleSettings":
        names = ("KT_GOOGLE_CLIENT_ID", "KT_GOOGLE_CLIENT_SECRET", "KT_PUBLIC_URL", "KT_SESSION_SECRET")
        values = {name: os.environ.get(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError("Google girişinde gerekli ortam değişkenleri eksik: " + ", ".join(missing))
        if len(values["KT_SESSION_SECRET"].encode("utf-8")) < 32:
            raise ValueError("KT_SESSION_SECRET en az 32 bayt olmalı (ör. openssl rand -hex 32).")
        parts = urlsplit(values["KT_PUBLIC_URL"])
        if (parts.scheme not in ("https", "http") or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment or parts.path not in ("", "/")):
            raise ValueError("KT_PUBLIC_URL kök URL olmalı; sorgu, kullanıcı bilgisi veya alt yol içeremez.")
        if parts.scheme != "https" and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Google girişinde herkese açık KT_PUBLIC_URL HTTPS olmalı.")
        return cls(**{ "client_id": values["KT_GOOGLE_CLIENT_ID"],
                       "client_secret": values["KT_GOOGLE_CLIENT_SECRET"],
                       "public_url": values["KT_PUBLIC_URL"].rstrip("/"),
                       "session_secret": values["KT_SESSION_SECRET"] })


class GoogleSessionStore:
    """Opaque, process-local sessions and stable pseudonymous Google user IDs."""

    def __init__(self, secret: str, *, clock=time.time):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("session secret must contain at least 32 bytes")
        self._clock = clock
        self._lock = threading.RLock()
        self._users_key = hmac.new(secret.encode("utf-8"), b"kt-google-user-id-v1", hashlib.sha256).digest()
        self._cookies: dict[str, dict] = {}
        self._sessions: dict[str, dict] = {}

    def create(self, claims: dict) -> str:
        sub = claims.get("sub")
        email = claims.get("email")
        name = claims.get("name") or email
        if not isinstance(sub, str) or not sub or not isinstance(email, str) or not email:
            raise ValueError("verified Google identity is incomplete")
        if claims.get("email_verified") is not True:
            raise ValueError("Google account email is not verified")
        user_id = hmac.new(self._users_key, sub.encode("utf-8"), hashlib.sha256).hexdigest()
        sid = secrets.token_urlsafe(32)
        with self._lock:
            self._expire_locked()
            self._cookies[sid] = {"id": user_id, "ad": str(name)[:120], "eposta": email[:320],
                                  "bitis": self._clock() + SESSION_SECONDS, "app_sessions": set()}
        return sid

    def oturum_ac(self, cookie_sid: str, request_cookie: str | None = None) -> str | None:
        """Create a separate Gradio token after a cookie-authenticated page load."""
        user = self._cookie_user(cookie_sid)
        if user is None or request_cookie != cookie_sid:
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            cookie = self._cookies.get(cookie_sid)
            if not cookie:
                return None
            cookie["app_sessions"].add(token)
            self._sessions[token] = {"cookie_sid": cookie_sid, "bitis": cookie["bitis"]}
        return token

    def cookie_user(self, cookie_sid) -> dict | None:
        """Resolve an opaque HttpOnly cookie for Gradio's auth dependency."""
        user = self._cookie_user(cookie_sid)
        if user is None:
            return None
        return {key: user[key] for key in ("id", "ad", "eposta")}

    def kullanici(self, token) -> dict | None:
        """Resolve a Gradio token only when this request also has its secure OIDC cookie."""
        if not isinstance(token, str) or not token:
            return None
        cookie_sid = self._request_cookie()
        with self._lock:
            session = self._sessions.get(token)
            if session is None or session["bitis"] <= self._clock() or session["cookie_sid"] != cookie_sid:
                self._sessions.pop(token, None)
                return None
            user = self._cookies.get(cookie_sid)
            if user is None or user["bitis"] <= self._clock():
                self._sessions.pop(token, None)
                return None
            return {key: user[key] for key in ("id", "ad", "eposta")}

    def cikis(self, token) -> None:
        with self._lock:
            session = self._sessions.pop(token, None)
            if session:
                user = self._cookies.get(session["cookie_sid"])
                if user:
                    user["app_sessions"].discard(token)

    def cikis_cookie(self, sid) -> None:
        with self._lock:
            user = self._cookies.pop(sid, None)
            if user:
                for token in user["app_sessions"]:
                    self._sessions.pop(token, None)

    def _cookie_user(self, sid) -> dict | None:
        if not isinstance(sid, str) or not sid:
            return None
        with self._lock:
            user = self._cookies.get(sid)
            if user is None or user["bitis"] <= self._clock():
                self.cikis_cookie(sid)
                return None
            return user

    @staticmethod
    def _request_cookie() -> str | None:
        """Read the cookie from Gradio's server-side request context, never from State."""
        try:
            from gradio.context import LocalContext

            request = LocalContext.request.get(None)
            return request.cookies.get(SESSION_COOKIE) if request is not None else None
        except (ImportError, AttributeError):
            return None

    def _expire_locked(self) -> None:
        now = self._clock()
        for sid in [sid for sid, user in self._cookies.items() if user["bitis"] <= now]:
            self.cikis_cookie(sid)
        for token in [token for token, session in self._sessions.items() if session["bitis"] <= now]:
            self.cikis(token)


def _google_client(settings: GoogleSettings):
    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        server_metadata_url=DISCOVERY_URL,
        client_kwargs={"scope": "openid email profile", "code_challenge_method": "S256"},
    )
    return oauth.google


def create_google_asgi(demo, settings: GoogleSettings, sessions: GoogleSessionStore | None = None,
                       oidc_client=None, theme=None, css: str | None = None):
    """Build OIDC routes and mount the existing Gradio UI behind the Google session.

    ``oidc_client`` is an injection point for deterministic provider tests; normal
    startup always creates Authlib's discovery-backed Google client.
    """
    import gradio as gr

    sessions = sessions or GoogleSessionStore(settings.session_secret)
    client = oidc_client or _google_client(settings)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        session_cookie=OAUTH_STATE_COOKIE,
        max_age=STATE_SECONDS,
        same_site="lax",
        https_only=settings.secure_cookie,
    )

    @app.get("/auth/login")
    async def google_login(request: Request):
        try:
            return await client.authorize_redirect(request, settings.callback_url)
        except Exception:
            LOG.warning("Google sign-in could not be started")
            return PlainTextResponse("Google girişini başlatamadık. Yeniden deneyin.", status_code=503)

    @app.get("/auth/callback")
    async def google_callback(request: Request):
        try:
            token = await client.authorize_access_token(request)
            claims = token.get("userinfo") if isinstance(token, dict) else None
            if not isinstance(claims, dict) or claims.get("email_verified") is not True:
                raise ValueError("unverified OIDC identity")
            sid = sessions.create(claims)
        except Exception:
            request.session.clear()
            LOG.warning("Google sign-in callback rejected")
            return PlainTextResponse("Google girişi doğrulanamadı. Giriş akışını yeniden başlatın.", status_code=401)
        request.session.clear()
        response = RedirectResponse("/app", status_code=303)
        response.headers["Cache-Control"] = "no-store"
        response.set_cookie(
            SESSION_COOKIE, sid, max_age=SESSION_SECONDS, httponly=True,
            secure=settings.secure_cookie, samesite="lax", path="/",
        )
        return response

    @app.post("/auth/logout")
    async def google_logout(request: Request):
        origin = request.headers.get("origin", "")
        expected_origin = urlsplit(settings.public_url).scheme + "://" + urlsplit(settings.public_url).netloc
        if origin != expected_origin:
            return PlainTextResponse("Çıkış isteği reddedildi.", status_code=403)
        sessions.cikis_cookie(request.cookies.get(SESSION_COOKIE))
        response = RedirectResponse("/", status_code=303)
        response.headers["Cache-Control"] = "no-store"
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True,
                              secure=settings.secure_cookie, samesite="lax")
        return response

    @app.get("/", response_class=HTMLResponse)
    async def login_landing(request: Request):
        if sessions.cookie_user(request.cookies.get(SESSION_COOKIE)):
            return RedirectResponse("/app", status_code=303)
        return HTMLResponse(
            "<!doctype html><html lang='tr'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Konuşmacı Değişimi Tespiti</title>"
            "<body style='font:16px system-ui;max-width:38rem;margin:12vh auto;padding:1.5rem'>"
            "<h1>Konuşmacı Değişimi Tespiti</h1><p>Uygulamaya Google hesabınızla giriş yapın.</p>"
            + google_login_html()
            + "</body></html>",
            headers={"Cache-Control": "no-store"},
        )

    def google_auth_dependency(request: Request) -> str | None:
        user = sessions.cookie_user(request.cookies.get(SESSION_COOKIE))
        return user["id"] if user else None

    app = gr.mount_gradio_app(app, demo, path="/app", auth_dependency=google_auth_dependency,
                              theme=theme, css=css, footer_links=["gradio", "settings"], mcp_server=False)
    return app, sessions


def google_login_html() -> str:
    return ('<a class="kt-google-login" href="/auth/login" rel="nofollow">'
            '<span aria-hidden="true">G</span> Google ile devam et</a>')


def google_logout_html() -> str:
    return ('<form class="kt-google-logout" action="/auth/logout" method="post">'
            '<button type="submit">Çıkış</button></form>')
