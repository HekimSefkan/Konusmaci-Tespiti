"""Web arayüzü: python app.py [--share] [--google-auth] [--port 7860]

Yerelde http://127.0.0.1:7860 adresinde açılır. --share, Gradio'nun genel geçici bağlantısını
da üretir (Colab için). Mantık: src/arayuz.py; sözleşme: docs/arayuz.md (3.15).

Uygulama içi hesaplar vardır (giriş / kayıt; src/hesap.py). Ortam değişkenleri:
  KT_KULLANICI_DB : kullanıcı dosyası (JSON; yoksa ~/.konusmaci-tespiti/kullanicilar.json).
  KT_KAYIT_ACIK   : "0" ise yeni kayıt kapalı (yalnızca giriş).
  KT_IZINLI_DIZIN : "Sunucudaki ses yolu" yalnızca bu klasörün altındaki dosyaları açar.
  KT_GECMIS_DIZIN : analiz geçmişi (kullanıcı başına alt klasör; yoksa geçmiş kapalı).
API sayfası ve istemci erişimi kapalıdır (tüm olaylar api_visibility="private").
"""
import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python app.py", description="Konuşmacı değişimi tespiti web arayüzü")
    ap.add_argument("--share", action="store_true", help="genel geçici bağlantı oluştur (Colab)")
    ap.add_argument("--google-auth", action="store_true",
                    help="isteğe bağlı Google OIDC girişi (KT_GOOGLE_* ortam değişkenleri gerekir)")
    ap.add_argument("--host", default="127.0.0.1", help="dinlenecek adres (varsayılan 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7860, help="port (varsayılan 7860)")
    a = ap.parse_args(argv)

    if a.share and a.google_auth:
        ap.error("--google-auth ile --share birlikte kullanılamaz; Google girişinde sabit HTTPS URL gerekir.")

    from src.oznitelik import cpu_yapilandir
    cpu_yapilandir()

    from src import hesap
    from src.arayuz import CSS, arayuz_olustur, tema

    if a.google_auth:
        from src import gecmis
        from src.arayuz import Uygulama, google_arayuz_olustur
        from src.google_auth import GoogleSessionStore, GoogleSettings, create_google_asgi
        import uvicorn

        settings = GoogleSettings.from_env()
        history_root = gecmis.gecmis_dizini()
        if not history_root:
            ap.error("Google modunda kullanıcı başına geçmiş için KT_GECMIS_DIZIN tanımlanmalı.")
        depo = GoogleSessionStore(settings.session_secret)
        uygulama = Uygulama(depo, gecmis_kok=history_root)
        demo = google_arayuz_olustur(uygulama)
        asgi_app, _ = create_google_asgi(demo.queue(), settings, sessions=depo, theme=tema(), css=CSS)
        uvicorn.run(asgi_app, host=a.host, port=a.port, access_log=False)
        return 0

    if a.share and hesap.kayit_acik():
        print("UYARI: --share bağlantısı herkese açıktır ve yeni kayıt açık (KT_KAYIT_ACIK). "
              "Bağlantıyı yalnızca güvendiğin kişilerle paylaş; gerekirse KT_KAYIT_ACIK=0 ile kaydı kapat.",
              file=sys.stderr)
    arayuz_olustur().queue().launch(server_name=a.host, server_port=a.port, share=a.share, theme=tema(), css=CSS,
                                    footer_links=["gradio", "settings"], mcp_server=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
