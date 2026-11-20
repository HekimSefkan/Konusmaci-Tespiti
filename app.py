"""Web arayüzü: python app.py [--share] [--port 7860]

Yerelde http://127.0.0.1:7860 adresinde açılır. --share, Gradio'nun genel geçici bağlantısını
da üretir (Colab için). Mantık: src/arayuz.py; sözleşme: docs/arayuz.md (3.15).
"""
import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python app.py", description="Konuşmacı değişimi tespiti web arayüzü")
    ap.add_argument("--share", action="store_true", help="genel geçici bağlantı oluştur (Colab)")
    ap.add_argument("--host", default="127.0.0.1", help="dinlenecek adres (varsayılan 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7860, help="port (varsayılan 7860)")
    a = ap.parse_args(argv)

    from src.arayuz import CSS, arayuz_olustur, tema

    arayuz_olustur().queue().launch(server_name=a.host, server_port=a.port, share=a.share, theme=tema(), css=CSS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
