import asyncio
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient
from gradio.context import LocalContext

from src import gecmis
from src.arayuz import CSS, Uygulama, google_arayuz_olustur, tema
from src.google_auth import (
    SESSION_COOKIE,
    GoogleSessionStore,
    GoogleSettings,
    _google_client,
    create_google_asgi,
)


class MockGoogleOIDC:
    """Provider stand-in that enforces the state, nonce and PKCE round trip."""

    def __init__(self, claims=None):
        self.claims = claims or {
            "sub": "google-sub-1",
            "email": "same@example.org",
            "email_verified": True,
            "name": "Google User",
        }
        self.state = "mock-state-123"

    async def authorize_redirect(self, request, redirect_uri):
        request.session["mock_state"] = self.state
        request.session["mock_nonce"] = "mock-nonce-123"
        request.session["mock_code_verifier"] = "mock-pkce-verifier"
        self.redirect_uri = redirect_uri
        return RedirectResponse(
            "https://accounts.google.test/authorize?" +
            "state=mock-state-123&nonce=mock-nonce-123&code_challenge=challenge&code_challenge_method=S256",
            status_code=302,
        )

    async def authorize_access_token(self, request):
        assert request.query_params.get("state") == request.session.get("mock_state")
        assert request.session.get("mock_nonce") == "mock-nonce-123"
        assert request.session.get("mock_code_verifier") == "mock-pkce-verifier"
        if request.query_params.get("code") != "accepted-code":
            raise ValueError("invalid code")
        return {"userinfo": self.claims}


@pytest.fixture
def settings():
    return GoogleSettings(
        client_id="test-client-id",
        client_secret="test-client-secret",
        public_url="http://127.0.0.1:8765",
        session_secret="s" * 48,
    )


def _test_client(settings, claims=None):
    store = GoogleSessionStore(settings.session_secret)
    provider = MockGoogleOIDC(claims)
    demo = google_arayuz_olustur(Uygulama(store, gecmis_kok="/tmp/kt-google-auth-test-history"))
    app, mounted_store = create_google_asgi(
        demo, settings, sessions=store, oidc_client=provider,
        theme=tema(), css=CSS + "\n.week8-mount-proof { color: #123456; }",
    )
    assert mounted_store is store
    return TestClient(app, base_url=settings.public_url), store, provider


def _login(client, provider):
    start = client.get("/auth/login", follow_redirects=False)
    assert start.status_code == 302
    assert urlsplit(start.headers["location"]).netloc == "accounts.google.test"
    query = parse_qs(urlsplit(start.headers["location"]).query)
    assert query["code_challenge_method"] == ["S256"]
    assert "kt_oauth_state" in client.cookies
    return client.get(
        f"/auth/callback?code=accepted-code&state={provider.state}", follow_redirects=False
    )


def test_oidc_login_mounts_ui_and_uses_http_only_cookie(settings):
    client, store, provider = _test_client(settings)

    landing = client.get("/")
    assert landing.status_code == 200
    assert "/auth/login" in landing.text
    assert client.get("/app/", follow_redirects=False).status_code == 401
    assert client.post("/app/gradio_api/upload", files={"files": ("private.wav", b"audio", "audio/wav")}).status_code == 401
    assert client.get("/app/gradio_api/file=private.wav").status_code == 401
    assert client.get("/app/config").status_code == 401
    assert client.post("/app/gradio_api/run/0", json={}).status_code == 401

    callback = _login(client, provider)
    assert provider.redirect_uri == settings.callback_url
    assert callback.status_code == 303
    assert callback.headers["location"] == "/app"
    cookie_header = callback.headers["set-cookie"].lower()
    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header
    assert "secure" not in cookie_header  # loopback-only test URL is plain HTTP
    sid = client.cookies.get(SESSION_COOKIE)
    assert sid and len(sid) >= 40
    assert store.cookie_user(sid)
    assert sid not in callback.text
    assert client.get("/", follow_redirects=False).status_code == 303
    app_page = client.get("/app/")
    assert app_page.status_code == 200
    assert ".week8-mount-proof" in app_page.text
    assert ".kt-gizli" in app_page.text

    app_token = store.oturum_ac(sid, sid)
    assert app_token and app_token != sid
    request = SimpleNamespace(cookies={SESSION_COOKIE: sid})
    context_token = LocalContext.request.set(request)
    try:
        assert store.kullanici(app_token)["eposta"] == "same@example.org"
    finally:
        LocalContext.request.reset(context_token)


def test_callback_rejects_state_failure_and_unverified_email(settings):
    client, store, provider = _test_client(settings)
    client.get("/auth/login", follow_redirects=False)
    bad_state = client.get("/auth/callback?code=accepted-code&state=attacker", follow_redirects=False)
    assert bad_state.status_code == 401
    assert client.cookies.get(SESSION_COOKIE) is None

    client, store, provider = _test_client(settings, {
        "sub": "google-sub-unverified", "email": "same@example.org", "email_verified": False
    })
    denied = _login(client, provider)
    assert denied.status_code == 401
    assert client.cookies.get(SESSION_COOKIE) is None
    assert store.cookie_user("invalid") is None


def test_logout_revokes_cookie_and_rejects_cross_origin(settings):
    client, store, provider = _test_client(settings)
    callback = _login(client, provider)
    assert callback.status_code == 303
    sid = client.cookies.get(SESSION_COOKIE)
    app_token = store.oturum_ac(sid, sid)

    rejected = client.post("/auth/logout", headers={"Origin": "https://evil.example"}, follow_redirects=False)
    assert rejected.status_code == 403
    assert store.cookie_user(sid)

    logged_out = client.post("/auth/logout", headers={"Origin": settings.public_url}, follow_redirects=False)
    assert logged_out.status_code == 303
    assert SESSION_COOKIE not in client.cookies
    assert store.cookie_user(sid) is None
    context_token = LocalContext.request.set(SimpleNamespace(cookies={SESSION_COOKIE: sid}))
    try:
        assert store.kullanici(app_token) is None
    finally:
        LocalContext.request.reset(context_token)
    assert client.get("/app/").status_code == 401


def test_google_sub_isolation_and_history_isolation_even_with_same_email(tmp_path, settings):
    store = GoogleSessionStore(settings.session_secret)
    sid_a = store.create({"sub": "google-a", "email": "same@example.org", "email_verified": True})
    sid_b = store.create({"sub": "google-b", "email": "same@example.org", "email_verified": True})
    token_a = store.oturum_ac(sid_a, sid_a)
    token_b = store.oturum_ac(sid_b, sid_b)
    app = Uygulama(store, gecmis_kok=str(tmp_path))

    context_a = LocalContext.request.set(SimpleNamespace(cookies={SESSION_COOKIE: sid_a}))
    try:
        user_a = store.kullanici(token_a)
        assert user_a and user_a["id"] != store.cookie_user(sid_b)["id"]
        history_a = app.gecmis_koku(user_a)
        record = gecmis.kaydet(history_a, {
            "dosya_id": "demo",
            "dosya_adi": "demo.wav",
            "segmentler": [[0.0, 1.0, "spk00"]],
            "ses_sn": 1.0,
        })
        assert len(app.gecmis_listesi(token_a)) == 1
    finally:
        LocalContext.request.reset(context_a)

    context_b = LocalContext.request.set(SimpleNamespace(cookies={SESSION_COOKIE: sid_b}))
    try:
        user_b = store.kullanici(token_b)
        assert user_b and app.gecmis_koku(user_b) != history_a
        assert app.gecmis_listesi(token_b) == []
        data, _, _ = app.gecmis_ac(token_b, record)
        assert data is None
    finally:
        LocalContext.request.reset(context_b)

    wrong_cookie = LocalContext.request.set(SimpleNamespace(cookies={SESSION_COOKIE: sid_b}))
    try:
        assert store.kullanici(token_a) is None
    finally:
        LocalContext.request.reset(wrong_cookie)


def test_authlib_client_config_enables_oidc_nonce_and_pkce(settings):
    client = _google_client(settings)
    assert client._server_metadata_url.endswith("openid-configuration")
    assert client.client_kwargs["scope"] == "openid email profile"
    assert client.client_kwargs["code_challenge_method"] == "S256"
    client.server_metadata = {
        "issuer": "https://accounts.google.com",
        "authorization_endpoint": "https://accounts.google.test/auth",
        "token_endpoint": "https://accounts.google.test/token",
        "jwks_uri": "https://accounts.google.test/keys",
    }
    authorization = asyncio.run(client.create_authorization_url(settings.callback_url))
    params = parse_qs(urlsplit(authorization["url"]).query)
    assert params["response_type"] == ["code"]
    assert params["redirect_uri"] == [settings.callback_url]
    assert params["scope"] == ["openid email profile"]
    assert params["state"] == [authorization["state"]]
    assert params["nonce"] == [authorization["nonce"]]
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"] and authorization["code_verifier"]
