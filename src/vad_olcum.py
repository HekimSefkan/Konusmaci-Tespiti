"""VAD ölçümü: Silero ve WebRTC VAD'ın referans RTTM'e göre karşılaştırılması. Sözleşme: docs/arayuz.md (3.14)

Komut satırı:
  python -m src.vad_olcum --zip Z --rttm-dizin R --liste L [L ...] --cikti-dizin C [--limit N]

Ses zip'ten okunur (toplu.SesKaynagi; süre sınırı uygulanmaz). Üç VAD çalışır:
  silero   : vad.konusma_bolgeleri (min konuşma 250 ms, min sessizlik 300 ms, 30 ms kenar payı)
  webrtc2/3: WebRTC VAD, 30 ms çerçeve, agresiflik 2 ve 3; aynı yumuşatma ve 30 ms kenar payı
Referans konuşma = RTTM segmentlerinin birleşimi. Ölçüler 10 ms çözünürlükte, tüm kayıtların
çerçeveleri toplanarak hesaplanır: kaçırılan, yanlış alarm ve toplam hata, referans konuşma
süresinin yüzdesi olarak; ayrıca yanlış alarm, referansta konuşma olmayan sürenin yüzdesi olarak
ve kayıt saati başına işlem süresi (sn).
webrtcvad / silero-vad fonksiyon içinde import edilir; ölçüm fonksiyonları yalnızca numpy kullanır.
"""
import argparse
import csv
import math
import os
import sys
import time
import traceback

import numpy as np

from . import cikti, vad
from .toplu import SesKaynagi, _hata_yaz, _sure, listeleri_oku

COZUNURLUK_SN = 0.01          # ölçüm çerçevesi: 10 ms
WEBRTC_CERCEVE_MS = 30
KENAR_PAYI_MS = 30            # silero get_speech_timestamps varsayılanı (speech_pad_ms); WebRTC'ye de uygulanır
VADLAR = ("silero", "webrtc2", "webrtc3")
KAYIT_SUTUNLARI = ["vad", "kayit", "ses_sn", "ref_konusma_sn", "ref_sessiz_sn", "kacirilan_sn",
                   "yanlis_alarm_sn", "sure_sn"]
OZET_SUTUNLARI = ["vad", "kayit_sayisi", "ses_saat", "kacirilan_yuzde", "yanlis_alarm_yuzde",
                  "toplam_hata_yuzde", "yanlis_alarm_sessiz_yuzde", "sn_per_saat"]
KAYIT_CSV = "vad_kayitlar.csv"
OZET_CSV = "vad_karsilastirma.csv"


# ---------------------------------------------------------------------------
# Saf numpy yardımcıları (yerelde test edilir)
# ---------------------------------------------------------------------------

def cercevelerden_bolgeler(kararlar, cerceve_sn: float) -> list[tuple[float, float]]:
    """Çerçeve başına konuşma kararlarını (bool dizisi) bitişik (bas, bit) bölgelerine çevirir."""
    k = np.asarray(kararlar, dtype=bool)
    if k.size == 0:
        return []
    fark = np.diff(np.concatenate([[0], k.astype(np.int8), [0]]))
    baslar, bitisler = np.where(fark == 1)[0], np.where(fark == -1)[0]
    return [(round(float(a) * cerceve_sn, 3), round(float(b) * cerceve_sn, 3)) for a, b in zip(baslar, bitisler)]


def yumusat(bolgeler, sure_sn: float, min_konusma_ms: int = vad.MIN_KONUSMA_MS,
            min_sessizlik_ms: int = vad.MIN_SESSIZLIK_MS, kenar_payi_ms: int = KENAR_PAYI_MS):
    """Silero ile aynı yumuşatma: kısa sessizlikleri doldur, kısa konuşmayı at, kenar payı ekle.

    1) min_sessizlik_ms'den kısa boşluklar birleştirilir; 2) min_konusma_ms'den kısa bölgeler atılır;
    3) her bölge iki yandan kenar_payi_ms genişletilir, [0, sure_sn] ile kırpılır, örtüşenler birleşir.
    """
    sirali = sorted((float(a), float(b)) for a, b in bolgeler if b > a)
    ms = lambda x: int(round(x * 1000))  # noqa: E731  (kayan nokta: 2,3 - 2,0 = 0,2999...)
    birlesik: list[list[float]] = []
    for a, b in sirali:
        if birlesik and ms(a) - ms(birlesik[-1][1]) < min_sessizlik_ms:
            birlesik[-1][1] = max(birlesik[-1][1], b)
        else:
            birlesik.append([a, b])
    uzun = [(a, b) for a, b in birlesik if ms(b) - ms(a) >= min_konusma_ms]
    pay = kenar_payi_ms / 1000
    sonuc: list[list[float]] = []
    for a, b in uzun:
        a, b = max(0.0, a - pay), min(sure_sn, b + pay)
        if sonuc and a <= sonuc[-1][1]:
            sonuc[-1][1] = max(sonuc[-1][1], b)
        else:
            sonuc.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in sonuc]


def maske(bolgeler, sure_sn: float, cozunurluk_sn: float = COZUNURLUK_SN) -> np.ndarray:
    """[0, sure_sn) aralığını cozunurluk_sn'lik çerçevelere böler; bölge içindekiler True."""
    n = int(math.ceil(sure_sn / cozunurluk_sn - 1e-9))
    m = np.zeros(max(n, 0), dtype=bool)
    for a, b in bolgeler:
        i0 = max(0, int(round(a / cozunurluk_sn)))
        i1 = min(n, int(round(b / cozunurluk_sn)))
        if i1 > i0:
            m[i0:i1] = True
    return m


def hata_olc(ref_maske: np.ndarray, hyp_maske: np.ndarray, cozunurluk_sn: float = COZUNURLUK_SN) -> dict:
    """Saniye cinsinden referans konuşma/sessizlik, kaçırılan konuşma ve yanlış alarm."""
    if ref_maske.shape != hyp_maske.shape:
        raise ValueError(f"maske boyları farklı: {ref_maske.shape} != {hyp_maske.shape}")
    return {
        "ref_konusma_sn": float(ref_maske.sum()) * cozunurluk_sn,
        "ref_sessiz_sn": float((~ref_maske).sum()) * cozunurluk_sn,
        "kacirilan_sn": float((ref_maske & ~hyp_maske).sum()) * cozunurluk_sn,
        "yanlis_alarm_sn": float((~ref_maske & hyp_maske).sum()) * cozunurluk_sn,
    }


def ozet_hesapla(satirlar: list[dict]) -> list[dict]:
    """VAD başına toplam (mikro) yüzdeler; satırlar KAYIT_SUTUNLARI biçiminde (str veya sayı).

    kacirilan/yanlis_alarm/toplam_hata yüzdeleri referans konuşma süresine,
    yanlis_alarm_sessiz_yuzde referansta konuşma olmayan süreye göredir.
    sn_per_saat: toplam işlem süresi / toplam ses süresi (saat).
    """
    sonuc = []
    for v in sorted({r["vad"] for r in satirlar}, key=lambda x: (VADLAR.index(x) if x in VADLAR else 99, x)):
        rr = [r for r in satirlar if r["vad"] == v]
        t = {k: sum(float(r[k]) for r in rr) for k in
             ("ses_sn", "ref_konusma_sn", "ref_sessiz_sn", "kacirilan_sn", "yanlis_alarm_sn", "sure_sn")}
        konusma, sessiz, saat = t["ref_konusma_sn"], t["ref_sessiz_sn"], t["ses_sn"] / 3600
        yuzde = (lambda x, payda: 100 * x / payda if payda > 0 else math.nan)
        sonuc.append({
            "vad": v, "kayit_sayisi": len(rr), "ses_saat": round(saat, 3),
            "kacirilan_yuzde": round(yuzde(t["kacirilan_sn"], konusma), 3),
            "yanlis_alarm_yuzde": round(yuzde(t["yanlis_alarm_sn"], konusma), 3),
            "toplam_hata_yuzde": round(yuzde(t["kacirilan_sn"] + t["yanlis_alarm_sn"], konusma), 3),
            "yanlis_alarm_sessiz_yuzde": round(yuzde(t["yanlis_alarm_sn"], sessiz), 3),
            "sn_per_saat": round(t["sure_sn"] / saat, 2) if saat > 0 else math.nan,
        })
    return sonuc


def ozet_tablosu(ozet: list[dict]) -> str:
    baslik = (f"{'vad':<9}{'kayıt':>6}{'saat':>8}{'kaçırılan%':>12}{'y.alarm%':>10}{'toplam%':>9}"
              f"{'y.alarm/sessiz%':>17}{'sn/saat':>9}")
    satirlar = [baslik, "-" * len(baslik)]
    for o in ozet:
        satirlar.append(f"{o['vad']:<9}{o['kayit_sayisi']:>6}{o['ses_saat']:>8.2f}{o['kacirilan_yuzde']:>12.2f}"
                        f"{o['yanlis_alarm_yuzde']:>10.2f}{o['toplam_hata_yuzde']:>9.2f}"
                        f"{o['yanlis_alarm_sessiz_yuzde']:>17.2f}{o['sn_per_saat']:>9.1f}")
    return "\n".join(satirlar)


# ---------------------------------------------------------------------------
# VAD'lar (ağır kütüphaneler fonksiyon içinde)
# ---------------------------------------------------------------------------

def webrtc_bolgeleri(ses: np.ndarray, sr: int, agresiflik: int) -> list[tuple[float, float]]:
    """WebRTC VAD: 30 ms çerçeveler (16 bit PCM), ardından yumusat (Silero ile aynı)."""
    import webrtcvad

    v = webrtcvad.Vad(agresiflik)
    n = int(sr * WEBRTC_CERCEVE_MS / 1000)
    pcm = (np.clip(np.asarray(ses, dtype=np.float32), -1, 1) * 32767).astype("<i2").tobytes()
    adet = len(ses) // n  # son eksik çerçeve atılır
    kararlar = [v.is_speech(pcm[2 * i * n: 2 * (i + 1) * n], sr) for i in range(adet)]
    return yumusat(cercevelerden_bolgeler(kararlar, WEBRTC_CERCEVE_MS / 1000), len(ses) / sr)


def _vad_calistir(ad: str, ses: np.ndarray, sr: int) -> list[tuple[float, float]]:
    if ad == "silero":
        return vad.konusma_bolgeleri(ses, sr)
    if ad.startswith("webrtc"):
        return webrtc_bolgeleri(ses, sr, int(ad[len("webrtc"):]))
    raise ValueError(f"bilinmeyen VAD {ad!r}")


# ---------------------------------------------------------------------------
# Ölçüm
# ---------------------------------------------------------------------------

def _csv_oku(yol: str) -> list[dict]:
    if not os.path.isfile(yol):
        return []
    with open(yol, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _csv_ekle(yol: str, satir: dict, sutunlar: list[str]) -> None:
    yeni = not os.path.isfile(yol) or os.path.getsize(yol) == 0
    with open(yol, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sutunlar, extrasaction="ignore")
        if yeni:
            w.writeheader()
        w.writerow(satir)


def olc(zip_yol: str, rttm_dizin: str, kayitlar: list[str], cikti_dizin: str, limit: int | None = None,
        log=print) -> list[dict]:
    """Kayıtlar üzerinde üç VAD'ı ölçer; vad_kayitlar.csv'ye kayıt satırları ekler (devam edebilir),
    vad_karsilastirma.csv'yi yazar ve özeti döndürür. limit: ilk N kayıt."""
    os.makedirs(cikti_dizin, exist_ok=True)
    if limit:
        kayitlar = kayitlar[:limit]
    kayit_csv = os.path.join(cikti_dizin, KAYIT_CSV)
    yapilan = {(r["vad"], r["kayit"]) for r in _csv_oku(kayit_csv)}
    bekleyen = [k for k in kayitlar if any((v, k) not in yapilan for v in VADLAR)]
    log(f"VAD ölçümü: {len(kayitlar)} kayıt, {len(kayitlar) - len(bekleyen)} önceden yapılmış, {len(bekleyen)} kaldı")
    if bekleyen and "silero" in VADLAR:
        vad._silero_modeli()  # model yükleme süresi ölçüme girmesin
    t_bas = time.perf_counter()
    with SesKaynagi(zip_yol) as kaynak:
        for i, kayit in enumerate(bekleyen, 1):
            try:
                ref = cikti.rttm_oku(os.path.join(rttm_dizin, f"{kayit}.rttm"))
                ses, sr = kaynak.oku(kayit)
                sure = max(len(ses) / sr, max((b for _, b, _ in ref), default=0.0))
                ref_m = maske([(a, b) for a, b, _ in ref], sure)
                bilgi = []
                for ad in VADLAR:
                    if (ad, kayit) in yapilan:
                        continue
                    t0 = time.perf_counter()
                    bolgeler = _vad_calistir(ad, ses, sr)
                    gecen = time.perf_counter() - t0
                    olcu = hata_olc(ref_m, maske(bolgeler, sure))
                    _csv_ekle(kayit_csv, {"vad": ad, "kayit": kayit, "ses_sn": round(len(ses) / sr, 3),
                                          **{k: round(v, 3) for k, v in olcu.items()}, "sure_sn": round(gecen, 4)},
                              KAYIT_SUTUNLARI)
                    bilgi.append(f"{ad} hata %{100 * (olcu['kacirilan_sn'] + olcu['yanlis_alarm_sn']) / max(olcu['ref_konusma_sn'], 1e-9):.1f}")
                bilgi = "  ".join(bilgi)
            except Exception as e:
                _hata_yaz(cikti_dizin, "vad-olcum", kayit, e)
                bilgi = f"HATA {type(e).__name__}: {e}"
            gecen = time.perf_counter() - t_bas
            log(f"[{i}/{len(bekleyen)}] {kayit}  {bilgi}  geçen {_sure(gecen)}  "
                f"kalan ~{_sure(gecen / i * (len(bekleyen) - i))}")
    secili = set(kayitlar)
    ozet = ozet_hesapla([r for r in _csv_oku(kayit_csv) if r["kayit"] in secili])
    with open(os.path.join(cikti_dizin, OZET_CSV), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OZET_SUTUNLARI)
        w.writeheader()
        w.writerows(ozet)
    log("")
    log(ozet_tablosu(ozet))
    return ozet


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.vad_olcum", description="Silero ve WebRTC VAD ölçümü")
    ap.add_argument("--zip", required=True, help="voxconverse_dev_wav.zip")
    ap.add_argument("--rttm-dizin", required=True, help="referans RTTM klasörü (voxconverse/dev)")
    ap.add_argument("--liste", nargs="+", required=True, help="kayıt listesi dosya(lar)ı")
    ap.add_argument("--cikti-dizin", required=True, help="vad_kayitlar.csv ve vad_karsilastirma.csv klasörü")
    ap.add_argument("--limit", type=int, help="yalnızca ilk N kayıt (deneme)")
    a = ap.parse_args(argv)
    try:
        olc(a.zip, a.rttm_dizin, listeleri_oku(a.liste), a.cikti_dizin, a.limit)
    except (FileNotFoundError, ValueError) as e:
        print(f"Girdi hatası: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"İç hata: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
