"""VAD ölçümü: Silero ve WebRTC VAD'ın referans RTTM'e göre karşılaştırılması. Sözleşme: docs/arayuz.md (3.14)

Komut satırı:
  python -m src.vad_olcum --zip Z --rttm-dizin R --liste L [L ...] --cikti-dizin C [--limit N]
                          [--dinleme-dizin D] [--dinleme-sayisi 5] [--parca-sn 10]

Ses zip'ten okunur (toplu.SesKaynagi; süre sınırı uygulanmaz). Üç VAD çalışır:
  silero   : vad.konusma_bolgeleri (min konuşma 250 ms, min sessizlik 300 ms, 30 ms kenar payı)
  webrtc2/3: WebRTC VAD, 30 ms çerçeve, agresiflik 2 ve 3; aynı yumuşatma ve 30 ms kenar payı
Referans konuşma = RTTM segmentlerinin birleşimi. Ölçüler 10 ms çözünürlükte, tüm kayıtların
çerçeveleri toplanarak hesaplanır: kaçırılan, yanlış alarm ve toplam hata, referans konuşma
süresinin yüzdesi olarak; ayrıca yanlış alarm, referansta konuşma olmayan sürenin yüzdesi olarak
ve kayıt saati başına işlem süresi (sn).
Aynı geçişte her kayıt için KABA bir SNR tahmini yapılır (Silero'nun konuşma dediği bölgelerin
ortalama gücü / konuşma olmayan bölgelerin ortalama gücü, dB); en düşük tahmini SNR'lı kayıtlardan
dinlemek için 10 sn'lik parçalar yazılabilir (--dinleme-dizin). Son seçim dinleyerek yapılır.
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
                   "yanlis_alarm_sn", "sure_sn", "snr_db", "konusmasiz_sn"]  # SNR yalnızca silero satırında
OZET_SUTUNLARI = ["vad", "kayit_sayisi", "ses_saat", "kacirilan_yuzde", "yanlis_alarm_yuzde",
                  "toplam_hata_yuzde", "yanlis_alarm_sessiz_yuzde", "sn_per_saat"]
KAYIT_CSV = "vad_kayitlar.csv"
OZET_CSV = "vad_karsilastirma.csv"
DINLEME_CSV = "dinleme_listesi.csv"
MIN_KONUSMASIZ_SN = 1.0       # SNR için en az konuşma olmayan süre; altında NaN
DINLEME_SAYISI = 5
PARCA_SN = 10.0


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


def snr_tahmini(ses: np.ndarray, sr: int, bolgeler, min_konusmasiz_sn: float = MIN_KONUSMASIZ_SN):
    """KABA SNR (dB): konuşma bölgelerinin ortalama gücü / konuşma olmayan bölgelerin ortalama gücü.

    Konuşma gücü gürültüyü de içerdiği için yalnızca kayıtları sıralamak (gürültülü aday bulmak)
    içindir. Dönüş: (snr_db, konusmasiz_sn). Konuşma olmayan süre min_konusmasiz_sn'den kısaysa,
    konuşma yoksa veya güçlerden biri 0 ise snr_db = NaN.
    """
    x = np.asarray(ses, dtype=np.float64)
    m = np.zeros(len(x), dtype=bool)
    for a, b in bolgeler:
        m[max(0, int(round(a * sr))):max(0, int(round(b * sr)))] = True
    konusmasiz_sn = float((~m).sum()) / sr
    if konusmasiz_sn < min_konusmasiz_sn or not m.any():
        return math.nan, konusmasiz_sn
    guc_k, guc_g = float(np.mean(x[m] ** 2)), float(np.mean(x[~m] ** 2))
    if guc_k <= 0 or guc_g <= 0:
        return math.nan, konusmasiz_sn
    return 10 * math.log10(guc_k / guc_g), konusmasiz_sn


def dinleme_parcasi(bolgeler, sure_sn: float, uzunluk_sn: float = PARCA_SN) -> tuple[float, float]:
    """Hem konuşma olmayan hem konuşma içeren bir parça: önünde en uzun sessizlik olan konuşma
    başlangıcı ortalanır (ilk yarı sessizlik, ikinci yarı konuşma). Konuşma yoksa dosyanın başı.
    Parça [0, sure_sn] içinde kalır. Dönüş: (bas_sn, bit_sn)."""
    L = min(uzunluk_sn, sure_sn)
    merkez, en_uzun, onceki_bit = 0.0, -1.0, 0.0
    for a, b in sorted(bolgeler):
        if a - onceki_bit > en_uzun:
            merkez, en_uzun = a, a - onceki_bit
        onceki_bit = max(onceki_bit, b)
    bas = min(max(0.0, merkez - L / 2), max(0.0, sure_sn - L))
    return round(bas, 3), round(bas + L, 3)


def _wav_yaz(yol: str, ses: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(yol, ses, sr)


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
        dinleme_dizin: str | None = None, dinleme_sayisi: int = DINLEME_SAYISI, parca_sn: float = PARCA_SN,
        log=print) -> list[dict]:
    """Kayıtlar üzerinde üç VAD'ı ölçer; vad_kayitlar.csv'ye kayıt satırları ekler (devam edebilir),
    vad_karsilastirma.csv'yi yazar ve özeti döndürür. limit: ilk N kayıt.
    Silero satırına kaba SNR (snr_tahmini) yazılır. dinleme_dizin verilirse en düşük SNR'lı
    dinleme_sayisi kayıttan parca_sn'lik birer WAV parçası ve dinleme_listesi.csv yazılır."""
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
                    snr = {}
                    if ad == "silero":
                        snr_db, konusmasiz = snr_tahmini(ses, sr, bolgeler)
                        snr = {"snr_db": round(snr_db, 2), "konusmasiz_sn": round(konusmasiz, 2)}
                    _csv_ekle(kayit_csv, {"vad": ad, "kayit": kayit, "ses_sn": round(len(ses) / sr, 3),
                                          **{k: round(v, 3) for k, v in olcu.items()}, "sure_sn": round(gecen, 4),
                                          **snr}, KAYIT_SUTUNLARI)
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
    if dinleme_dizin:
        dinleme_yaz(zip_yol, [r for r in _csv_oku(kayit_csv) if r["kayit"] in secili], dinleme_dizin,
                    cikti_dizin, dinleme_sayisi, parca_sn, log)
    return ozet


def dinleme_yaz(zip_yol: str, satirlar: list[dict], dinleme_dizin: str, cikti_dizin: str,
                sayi: int = DINLEME_SAYISI, parca_sn: float = PARCA_SN, log=print) -> list[dict]:
    """En düşük tahmini SNR'lı `sayi` kaydı seçer (silero satırları, NaN hariç), her birinden
    dinleme_parcasi ile seçilen parçayı WAV olarak yazar; listeyi dinleme_listesi.csv'ye yazar."""
    adaylar = []
    for r in satirlar:
        if r["vad"] == "silero" and r.get("snr_db") not in (None, "", "nan"):
            snr = float(r["snr_db"])
            if not math.isnan(snr):
                adaylar.append((snr, r["kayit"], float(r.get("konusmasiz_sn") or "nan")))
    adaylar.sort()
    secilen = []
    os.makedirs(dinleme_dizin, exist_ok=True)
    with SesKaynagi(zip_yol) as kaynak:
        for snr, kayit, konusmasiz in adaylar[:sayi]:
            try:
                ses, sr = kaynak.oku(kayit)
                bolgeler = vad.konusma_bolgeleri(ses, sr)
                bas, bit = dinleme_parcasi(bolgeler, len(ses) / sr, parca_sn)
                ad = f"{kayit}_{bas:.0f}s_snr{snr:.1f}dB.wav"
                _wav_yaz(os.path.join(dinleme_dizin, ad), ses[int(bas * sr):int(bit * sr)], sr)
                secilen.append({"kayit": kayit, "snr_db": snr, "konusmasiz_sn": konusmasiz,
                                "bas_sn": bas, "bit_sn": bit, "dosya": ad})
            except Exception as e:
                _hata_yaz(cikti_dizin, "dinleme", kayit, e)
                log(f"  HATA {kayit}: {type(e).__name__}: {e}")
    with open(os.path.join(cikti_dizin, DINLEME_CSV), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kayit", "snr_db", "konusmasiz_sn", "bas_sn", "bit_sn", "dosya"])
        w.writeheader()
        w.writerows(secilen)
    log("")
    log(f"En düşük tahmini SNR'lı {len(secilen)} kayıt (KABA tahmin; son seçim dinleyerek):")
    for s in secilen:
        log(f"  {s['kayit']}  SNR {s['snr_db']:6.1f} dB  konuşmasız {s['konusmasiz_sn']:6.1f} sn  "
            f"parça {s['bas_sn']:.1f}-{s['bit_sn']:.1f} sn -> {s['dosya']}")
    return secilen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.vad_olcum", description="Silero ve WebRTC VAD ölçümü")
    ap.add_argument("--zip", required=True, help="voxconverse_dev_wav.zip")
    ap.add_argument("--rttm-dizin", required=True, help="referans RTTM klasörü (voxconverse/dev)")
    ap.add_argument("--liste", nargs="+", required=True, help="kayıt listesi dosya(lar)ı")
    ap.add_argument("--cikti-dizin", required=True, help="vad_kayitlar.csv ve vad_karsilastirma.csv klasörü")
    ap.add_argument("--limit", type=int, help="yalnızca ilk N kayıt (deneme)")
    ap.add_argument("--dinleme-dizin", help="en düşük SNR'lı kayıtların dinleme parçaları (WAV) klasörü")
    ap.add_argument("--dinleme-sayisi", type=int, default=DINLEME_SAYISI)
    ap.add_argument("--parca-sn", type=float, default=PARCA_SN)
    a = ap.parse_args(argv)
    try:
        olc(a.zip, a.rttm_dizin, listeleri_oku(a.liste), a.cikti_dizin, a.limit,
            a.dinleme_dizin, a.dinleme_sayisi, a.parca_sn)
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
