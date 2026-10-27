"""Arayüz sözleşmesi kontrolü: modüller import edilir, fonksiyonlar var mı bakılır.

Çalıştırma: python tests/test_arayuz.py
"""
import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# modül -> {fonksiyon: parametre adları}  (docs/arayuz.md ile aynı olmalı)
SOZLESME = {
    "src.ses_girdi": {"ses_yukle": ["yol"]},
    "src.vad": {"konusma_bolgeleri": ["ses", "sr"]},
    "src.oznitelik": {
        "pencerele": ["bolgeler", "pencere_sn", "kaydirma_sn"],
        "vektorler": ["ses", "sr", "pencereler", "yontem"],
    },
    "src.kumeleme": {"kumele": ["X", "konusmaci_sayisi", "esik"]},
    "src.cikti": {
        "segmentleri_birlestir": ["pencereler", "etiketler", "ada_esik_sn", "bosluk_sn"],
        "degisim_noktalari": ["segmentler"],
        "rttm_yaz": ["segmentler", "dosya_id", "yol"],
        "rttm_oku": ["yol"],
        "tablo_yazdir": ["segmentler"],
    },
    "src.degerlendirme": {"degerlendir": ["ref_rttm", "hyp_rttm", "collar", "tolerans_sn"]},
    "src.main": {"main": ["argv"]},
}


def test_sozlesme():
    for modul, fonksiyonlar in SOZLESME.items():
        m = importlib.import_module(modul)
        for ad, parametreler in fonksiyonlar.items():
            assert hasattr(m, ad), f"{modul}.{ad} yok"
            gercek = list(inspect.signature(getattr(m, ad)).parameters)
            assert gercek == parametreler, f"{modul}.{ad} imzası uyuşmuyor: {gercek} != {parametreler}"
    from src import ses_girdi
    assert ses_girdi.SR == 16000


if __name__ == "__main__":
    test_sozlesme()
    print("test_arayuz: OK")
