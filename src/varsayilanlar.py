"""Ayarlanmış varsayılanlar: src/varsayilanlar.json. Sözleşme: docs/arayuz.md (3.12)

Eğitim yarısında seçilen AHC eşikleri (yöntem başına) ve genel MFCC ölçeği tek
dosyada durur; tek dosyalık komut (src.main), kumeleme.VARSAYILAN_ESIK ve toplu
çalıştırmanın --esikler verilmediği durum bunları kullanır.
Yalnızca standart kütüphane.
"""
import json
import os

DOSYA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "varsayilanlar.json")
YONTEMLER = ("mfcc", "ecapa")
MFCC_BOYUT = 26

_veri: dict | None = None


def yukle() -> dict:
    """varsayilanlar.json'u bir kez okur, denetler ve döndürür.

    Hata: ValueError (eksik yöntem eşiği, pozitif olmayan eşik, mfcc_olcek biçimi bozuk).
    """
    global _veri
    if _veri is None:
        with open(DOSYA, encoding="utf-8") as f:
            veri = json.load(f)
        esikler = veri.get("esik") or {}
        for y in YONTEMLER:
            if not isinstance(esikler.get(y), (int, float)) or esikler[y] <= 0:
                raise ValueError(f"{DOSYA}: '{y}' için geçerli eşik yok")
        olcek = veri.get("mfcc_olcek")
        if olcek is not None:
            if (len(olcek.get("ortalama", [])) != MFCC_BOYUT or len(olcek.get("std", [])) != MFCC_BOYUT
                    or min(olcek["std"]) <= 0):
                raise ValueError(f"{DOSYA}: mfcc_olcek {MFCC_BOYUT} ortalama ve {MFCC_BOYUT} pozitif std içermeli")
        _veri = veri
    return _veri


def esik(yontem: str) -> float:
    """Yöntemin ayarlanmış AHC eşiği. Hata: ValueError (bilinmeyen yöntem)."""
    if yontem not in YONTEMLER:
        raise ValueError(f"bilinmeyen yöntem {yontem!r} ({', '.join(YONTEMLER)})")
    return float(yukle()["esik"][yontem])


def mfcc_olcek() -> dict | None:
    """Genel MFCC ölçeği ({"ortalama", "std", ...}); henüz doldurulmadıysa None."""
    return yukle().get("mfcc_olcek")
