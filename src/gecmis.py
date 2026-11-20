"""Analiz geçmişi: KT_GECMIS_DIZIN altına kaydet, listele, aç, sil. Sözleşme: docs/arayuz.md (3.17)

Her başarılı analiz <YYYYAAGG-SSDDss>_<dosya adı>/ klasörüne yazılır: sonuc.json (özet, ayarlar,
segmentler, metrikler ve görünüm için ara bilgiler), <dosya_id>.rttm ve ses dosyasının kopyası
(yalnızca MAKS_SES_BAYT altıysa). Ortam değişkeni yoksa geçmiş kapalıdır.
Yalnızca standart kütüphane.
"""
import json
import os
import re
import shutil
from datetime import datetime

ORTAM = "KT_GECMIS_DIZIN"
SONUC = "sonuc.json"
MAKS_SES_BAYT = 50 * 1024 * 1024
YEREL_ANAHTARLAR = ("rttm_yolu", "ses_yolu", "gecmis_adi", "oynat_yolu")  # sonuc.json'a yazılmaz (yerel yollar)


def gecmis_dizini() -> str | None:
    """KT_GECMIS_DIZIN'in gerçek yolu; tanımlı değilse None (geçmiş kapalı)."""
    deger = os.environ.get(ORTAM, "").strip()
    return os.path.realpath(os.path.expanduser(deger)) if deger else None


def _guvenli_ad(ad: str) -> str:
    return re.sub(r"[^\w.-]+", "_", ad).strip("._")[:60] or "kayit"


def kaydet(kok: str, veri: dict, ses_yolu: str | None = None, rttm_yolu: str | None = None,
           zaman: datetime | None = None) -> str:
    """Analizi yeni bir klasöre kaydeder; klasör adını döndürür.

    veri: arayüzün sonuç sözlüğü (JSON'a yazılabilir olmalı). rttm_yolu kopyalanır; ses_yolu
    MAKS_SES_BAYT altındaysa kopyalanır (değilse "ses_notu" yazılır). Hata: OSError.
    """
    zaman = zaman or datetime.now()
    os.makedirs(kok, exist_ok=True)
    temel = f"{zaman:%Y%m%d-%H%M%S}_{_guvenli_ad(veri.get('dosya_id') or 'kayit')}"
    ad, n = temel, 2
    while os.path.exists(os.path.join(kok, ad)):
        ad, n = f"{temel}-{n}", n + 1
    klasor = os.path.join(kok, ad)
    os.makedirs(klasor)
    kayit = {k: v for k, v in veri.items() if k not in YEREL_ANAHTARLAR}
    kayit["kayit_tarihi"] = zaman.isoformat(timespec="seconds")
    kayit["rttm"] = kayit["ses"] = None
    if rttm_yolu and os.path.isfile(rttm_yolu):
        kayit["rttm"] = f"{_guvenli_ad(veri.get('dosya_id') or 'kayit')}.rttm"
        shutil.copyfile(rttm_yolu, os.path.join(klasor, kayit["rttm"]))
    if ses_yolu and os.path.isfile(ses_yolu):
        if os.path.getsize(ses_yolu) < MAKS_SES_BAYT:
            kayit["ses"] = _guvenli_ad(os.path.basename(ses_yolu))
            shutil.copyfile(ses_yolu, os.path.join(klasor, kayit["ses"]))
        else:
            kayit["ses_notu"] = f"ses {MAKS_SES_BAYT // (1024 * 1024)} MB'tan büyük; kopyalanmadı"
    with open(os.path.join(klasor, SONUC), "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=1)
    return ad


def _klasor(kok: str, ad: str) -> str:
    """Kayıt klasörünün yolu; ad kökün doğrudan alt klasörü değilse veya sonuc.json yoksa ValueError."""
    kok_g = os.path.realpath(kok)
    if not ad or ad != os.path.basename(ad) or ad in (".", ".."):
        raise ValueError("geçmiş kaydı bulunamadı")
    yol = os.path.realpath(os.path.join(kok_g, ad))
    if os.path.dirname(yol) != kok_g or not os.path.isfile(os.path.join(yol, SONUC)):
        raise ValueError(f"geçmiş kaydı bulunamadı ({ad})")
    return yol


def listele(kok: str) -> list[dict]:
    """Kayıtlar, yeniden eskiye: [{"ad", "tarih", "dosya", "konusmaci", "ses_sn"}]. Bozuk kayıt atlanır."""
    if not kok or not os.path.isdir(kok):
        return []
    sonuc = []
    for ad in sorted(os.listdir(kok), reverse=True):
        try:
            with open(os.path.join(_klasor(kok, ad), SONUC), encoding="utf-8") as f:
                k = json.load(f)
        except (ValueError, OSError):
            continue
        sonuc.append({"ad": ad, "tarih": k.get("kayit_tarihi", ""), "dosya": k.get("dosya_adi", ""),
                      "konusmaci": len({s[2] for s in k.get("segmentler", [])}), "ses_sn": k.get("ses_sn", 0.0)})
    return sonuc


def ac(kok: str, ad: str) -> dict:
    """Kaydı okur; yerel yolları ekler ("rttm_yolu", "ses_yolu": dosya yoksa None, "gecmis_adi").

    Hata: ValueError (kayıt yok veya bozuk).
    """
    klasor = _klasor(kok, ad)
    try:
        with open(os.path.join(klasor, SONUC), encoding="utf-8") as f:
            veri = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"geçmiş kaydı okunamadı ({ad}): {e}") from e
    for anahtar, alan in (("rttm_yolu", "rttm"), ("ses_yolu", "ses")):
        yol = os.path.join(klasor, veri[alan]) if veri.get(alan) else None
        veri[anahtar] = yol if yol and os.path.isfile(yol) else None
    veri["gecmis_adi"] = ad
    return veri


def guncelle(kok: str, ad: str, veri: dict) -> None:
    """Kaydın sonuc.json'unu veri ile günceller (ör. konuşmacı adları); dosya adları ve kayıt tarihi korunur.

    Yazma atomiktir (geçici dosya + os.replace). Hata: ValueError (kayıt yok), OSError.
    """
    klasor = _klasor(kok, ad)
    yol = os.path.join(klasor, SONUC)
    with open(yol, encoding="utf-8") as f:
        eski = json.load(f)
    yeni = {k: v for k, v in veri.items() if k not in YEREL_ANAHTARLAR}
    for k in ("kayit_tarihi", "rttm", "ses", "ses_notu"):
        if k in eski:
            yeni[k] = eski[k]
    gecici = yol + ".tmp"
    with open(gecici, "w", encoding="utf-8") as f:
        json.dump(yeni, f, ensure_ascii=False, indent=1)
    os.replace(gecici, yol)


def sil(kok: str, ad: str) -> None:
    """Kaydın klasörünü siler. Hata: ValueError (kayıt yok)."""
    shutil.rmtree(_klasor(kok, ad))
