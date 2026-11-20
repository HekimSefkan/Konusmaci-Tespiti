"""Web arayüzü: python app.py [--share] [--port 7860]

Yerelde http://127.0.0.1:7860 adresinde açılır. --share, Gradio'nun genel geçici bağlantısını
da üretir (Colab için). Mantık: src/arayuz.py; sözleşme: docs/arayuz.md (3.15).

Ortam değişkenleri:
  KT_KULLANICI, KT_SIFRE : ikisi de varsa arayüz şifreyle korunur (launch(auth=...)).
  KT_IZINLI_DIZIN        : "Sunucudaki ses yolu" yalnızca bu klasörün altındaki dosyaları açar.
  KT_GECMIS_DIZIN        : analiz geçmişinin klasörü (yoksa geçmiş kapalı).
"""
import argparse
import os
import sys


def kimlik() -> tuple[str, str] | None:
    """(kullanıcı, şifre): KT_KULLANICI ve KT_SIFRE ikisi de doluysa, değilse None."""
    k, s = os.environ.get("KT_KULLANICI", "").strip(), os.environ.get("KT_SIFRE", "")
    return (k, s) if k and s else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python app.py", description="Konuşmacı değişimi tespiti web arayüzü")
    ap.add_argument("--share", action="store_true", help="genel geçici bağlantı oluştur (Colab)")
    ap.add_argument("--host", default="127.0.0.1", help="dinlenecek adres (varsayılan 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7860, help="port (varsayılan 7860)")
    a = ap.parse_args(argv)

    from src.arayuz import CSS, arayuz_olustur, izinli_yollar, tema

    auth = kimlik()
    if a.share and auth is None:
        print("UYARI: --share bağlantısı herkese açık ve şifre yok. KT_KULLANICI ve KT_SIFRE ortam "
              "değişkenleriyle şifre koruması açın.", file=sys.stderr)
    arayuz_olustur().queue().launch(server_name=a.host, server_port=a.port, share=a.share, auth=auth,
                                    theme=tema(), css=CSS, allowed_paths=izinli_yollar())
    return 0


if __name__ == "__main__":
    sys.exit(main())
