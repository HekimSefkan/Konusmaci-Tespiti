"""Toplu çalıştırma: öznitelik önbelleği, E1-E5 deneyleri, eşik taraması, değişim modeli eğitimi.
Sözleşme: docs/arayuz.md

Komut satırı (bir iş seçilir):
  python -m src.toplu --deney E1|E2|E3|E4|E5 --zip Z --rttm-dizin R --liste L [L ...]
                      --cikti-dizin C --onbellek O [--esik X | --esikler J] [--model M]
  python -m src.toplu --onbellek-doldur --zip Z --liste L [L ...] --onbellek O [--yontemler mfcc ecapa]
  python -m src.toplu --esik-tara --zip Z --rttm-dizin R --liste L --onbellek O --esikler J
  python -m src.toplu --model-egit --zip Z --rttm-dizin R --liste L --onbellek O --model M

Sesler zip'ten tek tek belleğe okunur (zip açılmaz). Her kaydın VAD bölgeleri,
pencereleri ve vektörleri önbelleğe yazılır; iş yarıda kalırsa kaldığı yerden sürer.
"""
import argparse
import csv
import datetime
import io
import json
import math
import os
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np

from . import bolme, cikti, degerlendirme, degisim_modeli, kumeleme, oznitelik, ses_girdi, vad, varsayilanlar
from . import main as ana

DENEYLER = {
    "E1": ("mfcc", "sayi"),    # MFCC + AHC, konuşmacı sayısı referanstan
    "E2": ("mfcc", "esik"),    # MFCC + AHC, eşikle
    "E3": ("ecapa", "sayi"),   # ECAPA + AHC, konuşmacı sayısı referanstan
    "E4": ("ecapa", "esik"),   # ECAPA + AHC, eşikle
    "E5": ("ecapa", "model"),  # ECAPA + eğitilmiş değişim modeli + AHC (eşikle)
}
CSV_SUTUNLARI = [
    "deney", "kayit", "der", "jer", "f1",  # ilk beş sütun sabit (grafik betiği okur)
    "precision", "recall", "tp", "fp", "fn",
    "yanlis_alarm_sn", "kacirilan_sn", "karisiklik_sn", "referans_sn",
    "ses_sn", "sure_sn",
]
OZET_SUTUNLARI = ["deney", "kayit_sayisi", "der", "jer_ort", "mikro_f1",
                  "toplam_sure_sn", "toplam_ses_sn", "atlanan"]
COLLAR = 0.5
TOLERANS_SN = 0.5
PENCERE_SN = 1.5
KAYDIRMA_SN = 0.75
YUZDELIK_SAYISI = 40     # veriden eşik ızgarası: birleşme yüksekliklerinin yüzdelikleri
GENISLETME_TURU = 3      # en iyi eşik kenara düşerse en çok bu kadar genişletme
HATA_GUNLUGU = "hata_gunlugu.txt"


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _yaz(*args):
    print(*args, flush=True)


def _sure(sn: float) -> str:
    sn = int(round(sn))
    s, sn = divmod(sn, 3600)
    d, sn = divmod(sn, 60)
    return f"{s}:{d:02d}:{sn:02d}"


def _hata_yaz(klasor: str, is_adi: str, kayit: str, hata: BaseException) -> None:
    os.makedirs(klasor, exist_ok=True)
    iz = "".join(traceback.format_exception(type(hata), hata, hata.__traceback__)).strip().splitlines()
    with open(os.path.join(klasor, HATA_GUNLUGU), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {is_adi} {kayit}: "
                f"{type(hata).__name__}: {hata}\n")
        f.writelines(f"    {s}\n" for s in iz[-8:])


def _referans_yolu(rttm_dizin: str, kayit: str) -> str:
    yol = os.path.join(rttm_dizin, f"{kayit}.rttm")
    if not os.path.isfile(yol):
        raise FileNotFoundError(f"{kayit}: referans RTTM yok")
    return yol


def listeleri_oku(yollar: list[str]) -> list[str]:
    """Bir veya daha çok liste dosyasını okur; tekrarları atar, sırayı korur."""
    gorulen, sonuc = set(), []
    for yol in yollar:
        for k in bolme.liste_oku(yol):
            if k not in gorulen:
                gorulen.add(k)
                sonuc.append(k)
    return sonuc


# ---------------------------------------------------------------------------
# Ses kaynağı (zip)
# ---------------------------------------------------------------------------

class SesKaynagi:
    """Zip içindeki WAV'ları kayıt kimliğiyle (dosya adı, uzantısız) tek tek okur."""

    def __init__(self, zip_yol: str | None):
        self.zip_yol = zip_yol
        self._zf = None
        self._adlar: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.kapat()

    def _ac(self):
        if self._zf is None:
            if not self.zip_yol or not os.path.isfile(self.zip_yol):
                raise FileNotFoundError(f"zip bulunamadı: {self.zip_yol}")
            self._zf = zipfile.ZipFile(self.zip_yol)
            for ad in self._zf.namelist():
                if ad.lower().endswith(".wav"):
                    self._adlar[os.path.splitext(os.path.basename(ad))[0]] = ad

    def oku(self, kayit: str) -> tuple[np.ndarray, int]:
        """(ses, 16000); ses_yukle ile aynı ön işleme (on_isle)."""
        self._ac()
        if kayit not in self._adlar:
            raise ValueError(f"{kayit}: zip içinde WAV yok")
        veri = self._zf.read(self._adlar[kayit])
        ses_girdi.wav_denetle(veri, f"{kayit}.wav")  # kesik WAV sessizce kısaltılıp okunmasın
        import soundfile as sf

        try:
            x, sr = sf.read(io.BytesIO(veri), dtype="float32", always_2d=True)
        except Exception:
            # Bellekten okunamazsa: yerel geçici dosyaya yaz, oku, sil.
            with tempfile.TemporaryDirectory(prefix="kt_") as d:
                yol = os.path.join(d, f"{kayit}.wav")
                with open(yol, "wb") as f:
                    f.write(veri)
                return ses_girdi.ses_yukle(yol)
        try:
            return ses_girdi.on_isle(x, int(sr)), ses_girdi.SR
        except ValueError as e:
            raise ValueError(f"{kayit}: {e}") from e

    def kapat(self):
        if self._zf is not None:
            self._zf.close()
            self._zf = None


# ---------------------------------------------------------------------------
# Önbellek
# ---------------------------------------------------------------------------

def onbellek_anahtarlari() -> dict[str, str]:
    """Önbellek alt klasör adları; hesaplamayı etkileyen tüm parametreleri içerir."""
    v = f"v{vad.MIN_KONUSMA_MS}_s{vad.MIN_SESSIZLIK_MS}"
    p = f"p{PENCERE_SN:g}_k{KAYDIRMA_SN:g}_{v}"
    return {
        "vad": v,
        "mfcc": f"{p}_mfcc{oznitelik.N_MFCC}c1_f{oznitelik.N_FFT}_w{oznitelik.WIN_LENGTH}_h{oznitelik.HOP_LENGTH}",
        "ecapa": f"{p}_ecapa-voxceleb",
    }


def onbellek_yolu(onbellek: str, tur: str, kayit: str) -> str:
    """{onbellek}/<tur>/<anahtar>/<kayit>.npz; tur: "vad", "mfcc" veya "ecapa"."""
    return os.path.join(onbellek, tur, onbellek_anahtarlari()[tur], f"{kayit}.npz")


def _parametre_yaz(klasor: str, tur: str) -> None:
    yol = os.path.join(klasor, "parametreler.json")
    if os.path.exists(yol):
        return
    p = {"tur": tur, "anahtar": onbellek_anahtarlari()[tur], "sr": ses_girdi.SR,
         "vad": {"min_konusma_ms": vad.MIN_KONUSMA_MS, "min_sessizlik_ms": vad.MIN_SESSIZLIK_MS}}
    if tur != "vad":
        p["pencere"] = {"pencere_sn": PENCERE_SN, "kaydirma_sn": KAYDIRMA_SN}
    if tur == "mfcc":
        p["mfcc"] = {"n_mfcc": oznitelik.N_MFCC, "c0_atilir": True, "n_fft": oznitelik.N_FFT,
                     "win_length": oznitelik.WIN_LENGTH, "hop_length": oznitelik.HOP_LENGTH}
    if tur == "ecapa":
        p["ecapa"] = {"kaynak": oznitelik.ECAPA_KAYNAK}
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(p, f, ensure_ascii=False, indent=2)


def _npz_yaz(yol: str, tur: str, **diziler) -> None:
    klasor = os.path.dirname(yol)
    os.makedirs(klasor, exist_ok=True)
    _parametre_yaz(klasor, tur)
    gecici = yol + ".yaziliyor"
    with open(gecici, "wb") as f:
        np.savez(f, **diziler)
    os.replace(gecici, yol)  # yarım dosya geçerli sayılmasın


def _npz_oku(yol: str) -> dict | None:
    if not os.path.isfile(yol):
        return None
    try:
        with np.load(yol) as d:
            return {k: d[k] for k in d.files}
    except Exception:
        return None  # bozuk önbellek: yeniden hesaplanır


def ozellik_al(kaynak: SesKaynagi, kayit: str, onbellek: str, yontemler=("mfcc", "ecapa")) -> dict:
    """Kaydın VAD bölgeleri ve istenen yöntemlerin pencere/vektörleri (önbellekten veya hesaplayarak).

    Dönüş: {"bolgeler", "ses_sn", "sure_sn", <yontem>: {"pencereler", "X"}}.
    sure_sn: bu verilerin ilk hesaplanma süresi (ses okuma + VAD + öznitelik).
    """
    ses_bellek: dict = {}

    def ses():
        if "x" not in ses_bellek:
            ses_bellek["x"] = kaynak.oku(kayit)
        return ses_bellek["x"]

    yol = onbellek_yolu(onbellek, "vad", kayit)
    d = _npz_oku(yol)
    if d is None:
        t0 = time.perf_counter()
        x, sr = ses()
        bolgeler = vad.konusma_bolgeleri(x, sr)
        d = {"bolgeler": np.array(bolgeler, dtype=np.float64).reshape(-1, 2),
             "ses_sn": np.float64(len(x) / sr), "sure_sn": np.float64(time.perf_counter() - t0)}
        _npz_yaz(yol, "vad", **d)
    bolgeler = [(float(a), float(b)) for a, b in d["bolgeler"]]
    sonuc = {"bolgeler": bolgeler, "ses_sn": float(d["ses_sn"]), "sure_sn": float(d["sure_sn"])}

    for yontem in yontemler:
        yol = onbellek_yolu(onbellek, yontem, kayit)
        e = _npz_oku(yol)
        if e is None:
            t0 = time.perf_counter()
            pencereler = oznitelik.pencerele(bolgeler, PENCERE_SN, KAYDIRMA_SN)
            if pencereler:
                x, sr = ses()
                X = oznitelik.vektorler(x, sr, pencereler, yontem)
            else:
                X = np.zeros((0, oznitelik.BOYUT[yontem]), dtype=np.float32)
            e = {"pencereler": np.array(pencereler, dtype=np.float64).reshape(-1, 2),
                 "X": X.astype(np.float32), "sure_sn": np.float64(time.perf_counter() - t0)}
            _npz_yaz(yol, yontem, **e)
        sonuc[yontem] = {"pencereler": [(float(a), float(b)) for a, b in e["pencereler"]],
                         "X": np.asarray(e["X"], dtype=np.float32)}
        sonuc["sure_sn"] += float(e["sure_sn"])
    return sonuc


def onbellek_doldur(zip_yol: str, kayitlar: list[str], onbellek: str,
                    yontemler=("mfcc", "ecapa"), gunluk_dizin: str | None = None, log=_yaz) -> dict:
    """Tüm kayıtlar için VAD + öznitelik önbelleğini doldurur. Dönüş: {"tamam", "hatali"}."""
    gunluk_dizin = gunluk_dizin or onbellek
    t_bas, tamam, hatali = time.perf_counter(), 0, 0
    with SesKaynagi(zip_yol) as kaynak:
        for i, kayit in enumerate(kayitlar, 1):
            try:
                oz = ozellik_al(kaynak, kayit, onbellek, tuple(yontemler))
                tamam += 1
                bilgi = f"{oz['ses_sn']:.0f} sn ses, {len(oz['bolgeler'])} bölge"
            except Exception as e:
                hatali += 1
                _hata_yaz(gunluk_dizin, "onbellek", kayit, e)
                bilgi = f"HATA {type(e).__name__}: {e}"
            gecen = time.perf_counter() - t_bas
            log(f"[{i}/{len(kayitlar)}] {kayit}  {bilgi}  geçen {_sure(gecen)}  "
                f"kalan ~{_sure(gecen / i * (len(kayitlar) - i))}")
    log(f"Önbellek: {tamam} tamam, {hatali} hatalı")
    return {"tamam": tamam, "hatali": hatali}


# ---------------------------------------------------------------------------
# CSV ve özet
# ---------------------------------------------------------------------------

def _csv_oku(yol: str) -> list[dict]:
    if not os.path.isfile(yol):
        return []
    with open(yol, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _csv_ekle(yol: str, satir: dict) -> None:
    yeni = not os.path.isfile(yol) or os.path.getsize(yol) == 0
    with open(yol, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_SUTUNLARI, extrasaction="ignore")
        if yeni:
            w.writeheader()
        w.writerow(satir)
        f.flush()


def _sayi(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def ozet_hesapla(satirlar: list[dict]) -> dict:
    """Kayıt satırlarından toplam özet.

    der = Σ(yanlis_alarm + kacirilan + karisiklik) / Σ referans_sn; jer_ort = NaN'sız ortalama;
    mikro_f1 = Σtp, Σfp, Σfn'den F1 (boş tahminde kesinlik 1, boş referansta duyarlık 1).
    """
    hata = sum(_sayi(r["yanlis_alarm_sn"]) + _sayi(r["kacirilan_sn"]) + _sayi(r["karisiklik_sn"])
               for r in satirlar)
    ref = sum(_sayi(r["referans_sn"]) for r in satirlar)
    jer = [_sayi(r["jer"]) for r in satirlar if not math.isnan(_sayi(r["jer"]))]
    tp = sum(int(_sayi(r["tp"])) for r in satirlar)
    fp = sum(int(_sayi(r["fp"])) for r in satirlar)
    fn = sum(int(_sayi(r["fn"])) for r in satirlar)
    return {
        "kayit_sayisi": len(satirlar),
        "der": hata / ref if ref > 0 else math.nan,
        "jer_ort": float(np.mean(jer)) if jer else math.nan,
        "mikro_f1": degisim_modeli._mikro_f1(tp, fp, fn),
        "toplam_sure_sn": sum(_sayi(r["sure_sn"]) for r in satirlar),
        "toplam_ses_sn": sum(_sayi(r["ses_sn"]) for r in satirlar),
    }


def ozet_yaz(yol: str, ozet: dict) -> None:
    """ozet.csv'de aynı deneyin satırını değiştirir (yoksa ekler)."""
    satirlar = [r for r in _csv_oku(yol) if r.get("deney") != ozet["deney"]]
    satirlar.append({k: ozet.get(k, "") for k in OZET_SUTUNLARI})
    satirlar.sort(key=lambda r: r["deney"])
    os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
    gecici = yol + ".yaziliyor"
    with open(gecici, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OZET_SUTUNLARI, extrasaction="ignore")
        w.writeheader()
        w.writerows(satirlar)
    os.replace(gecici, yol)


def ozet_tablosu(yol: str) -> str:
    """ozet.csv'yi okunabilir tabloya çevirir."""
    satirlar = _csv_oku(yol)
    if not satirlar:
        return "(özet yok)"
    baslik = f"{'deney':<6}{'kayıt':>7}{'DER':>9}{'JER ort':>9}{'mikro F1':>10}{'süre':>10}{'ses':>10}{'atlanan':>9}"
    cizgi = "-" * len(baslik)
    govde = [
        f"{r['deney']:<6}{r['kayit_sayisi']:>7}{_sayi(r['der']):>9.4f}{_sayi(r['jer_ort']):>9.4f}"
        f"{_sayi(r['mikro_f1']):>10.4f}{_sure(_sayi(r['toplam_sure_sn'])):>10}"
        f"{_sure(_sayi(r['toplam_ses_sn'])):>10}{r['atlanan']:>9}"
        for r in satirlar
    ]
    return "\n".join([baslik, cizgi, *govde])


# ---------------------------------------------------------------------------
# Deneyler
# ---------------------------------------------------------------------------

def deney_calistir(zip_yol: str, rttm_dizin: str, kayitlar: list[str], deney: str, cikti_dizin: str,
                   onbellek: str, esik: float | None = None, model_yolu: str | None = None,
                   mfcc_olcek: dict | None = None, log=_yaz) -> dict:
    """Bir deneyi kayıtlar üzerinde çalıştırır; RTTM, kayıt CSV'si ve ozet.csv yazar.

    Önceden CSV'de olan kayıtlar atlanır. esik: E2/E4/E5 için AHC eşiği (None ->
    yöntemin ayarlanmış eşiği, varsayilanlar.esik; E5'te ecapa). mfcc_olcek: E1/E2'de kümelemeden önce uygulanan genel MFCC
    ölçeği (esikler.json "mfcc_olcek"; None ise ham MFCC, uyarı yazılır). E5 ölçeği
    model paketinden alır. Dönüş: özet sözlüğü (+ "hatali": bu çalıştırmada hata veren kayıt sayısı).
    Hata: ValueError (bilinmeyen deney, E5'te model yok).
    """
    if deney not in DENEYLER:
        raise ValueError(f"bilinmeyen deney {deney!r} ({', '.join(DENEYLER)})")
    yontem, kip = DENEYLER[deney]
    yontemler = ("ecapa", "mfcc") if kip == "model" else (yontem,)
    paket = None
    if kip == "model":
        if not model_yolu:
            raise ValueError("E5 için --model gerekir")
        paket = degisim_modeli.yukle(model_yolu)

    if kip != "sayi" and esik is None:
        esik = varsayilanlar.esik(yontem)  # yöntemin ayarlanmış eşiği (E5: ecapa)
    rttm_klasor = os.path.join(cikti_dizin, deney)
    os.makedirs(rttm_klasor, exist_ok=True)
    csv_yol = os.path.join(cikti_dizin, f"{deney}.csv")
    yapilan = {r["kayit"] for r in _csv_oku(csv_yol)}
    bekleyen = [k for k in kayitlar if k not in yapilan]
    esik_metni = "sayı referanstan" if kip == "sayi" else f"eşik {esik}"
    if yontem == "mfcc" and kip != "model":
        esik_metni += ", MFCC genel ölçekli" if mfcc_olcek else ", ham MFCC"
        if not mfcc_olcek:
            log("UYARI: MFCC ölçeği verilmedi (esikler.json 'mfcc_olcek'); ham MFCC kümelenecek")
    log(f"{deney} ({yontem}, {esik_metni}): {len(kayitlar)} kayıt, "
        f"{len(kayitlar) - len(bekleyen)} önceden yapılmış, {len(bekleyen)} kaldı")

    t_bas, hatali = time.perf_counter(), 0
    with SesKaynagi(zip_yol) as kaynak:
        for i, kayit in enumerate(bekleyen, 1):
            try:
                ref_yol = _referans_yolu(rttm_dizin, kayit)
                oz = ozellik_al(kaynak, kayit, onbellek, yontemler)
                t1 = time.perf_counter()
                if kip == "model":
                    segler = degisim_modeli.e5_segmentleri(
                        paket, oz["bolgeler"], oz["ecapa"]["pencereler"], oz["ecapa"]["X"],
                        oz["mfcc"]["X"], esik)
                else:
                    sayi = None
                    if kip == "sayi":
                        sayi = len({e for _, _, e in cikti.rttm_oku(ref_yol)}) or None
                    X = oz[yontem]["X"]
                    if yontem == "mfcc":
                        X = oznitelik.olcekle(X, mfcc_olcek)
                    segler = ana.hat_segmentleri(oz["bolgeler"], oz[yontem]["pencereler"], X,
                                                 sayi, esik if kip == "esik" else None)
                hyp_yol = os.path.join(rttm_klasor, f"{kayit}.rttm")
                cikti.rttm_yaz(segler, kayit, hyp_yol)
                s = degerlendirme.degerlendir(ref_yol, hyp_yol, COLLAR, TOLERANS_SN)
                s = {k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()}
                satir = {"deney": deney, "kayit": kayit, **s, "ses_sn": round(oz["ses_sn"], 3),
                         "sure_sn": round(oz["sure_sn"] + time.perf_counter() - t1, 3)}
                _csv_ekle(csv_yol, satir)
                bilgi = f"DER={s['der']:.3f} F1={s['f1']:.3f}"
            except Exception as e:
                hatali += 1
                _hata_yaz(cikti_dizin, deney, kayit, e)
                bilgi = f"HATA {type(e).__name__}: {e}"
            gecen = time.perf_counter() - t_bas
            log(f"[{i}/{len(bekleyen)}] {kayit}  {bilgi}  geçen {_sure(gecen)}  "
                f"kalan ~{_sure(gecen / i * (len(bekleyen) - i))}")

    secili = set(kayitlar)
    satirlar = [r for r in _csv_oku(csv_yol) if r["kayit"] in secili]
    ozet = {"deney": deney, **ozet_hesapla(satirlar)}
    ozet["atlanan"] = len(kayitlar) - ozet["kayit_sayisi"]
    ozet_yolu = os.path.join(cikti_dizin, "ozet.csv")
    ozet_yaz(ozet_yolu, ozet)
    log("")
    log(ozet_tablosu(ozet_yolu))
    ozet["hatali"] = hatali
    return ozet


# ---------------------------------------------------------------------------
# Eşik taraması
# ---------------------------------------------------------------------------

def yuzdelik_izgarasi(yukseklikler, n: int = YUZDELIK_SAYISI) -> list[float]:
    """AHC birleşme yüksekliklerinden aday eşikler.

    %1 ile %99 arasında n yüzdelik; üst uca doğru sıklaşır (q = 100 - geomspace(99, 1, n)),
    çünkü konuşmacılar arası birleşmeler her kaydın en yüksek birkaç birleşmesidir.
    """
    h = np.asarray(yukseklikler, dtype=np.float64)
    if h.size == 0:
        return []
    q = 100 - np.geomspace(99, 1, n)
    return sorted({round(float(v), 5) for v in np.percentile(h, q)})


def _genislet(izgara: list[float], yukseklikler, kenar: str, tur: int) -> list[float]:
    """Kenara düşen ızgarayı o yönde genişletir (yalnızca yeni değerler döner)."""
    h = np.asarray(yukseklikler, dtype=np.float64)
    if kenar == "alt":
        uc = izgara[0]
        if tur == 0:
            yeni = list(np.percentile(h, np.geomspace(1, 0.01, 8))) + [float(h.min()) * 0.9]
        else:
            yeni = [uc * f for f in (0.8, 0.6, 0.4)]
        yeni = [v for v in yeni if 0 < v < uc - 1e-9]
    else:
        uc = izgara[-1]
        if tur == 0:
            yeni = list(np.percentile(h, 100 - np.geomspace(1, 0.01, 8))) + [float(h.max()) * 1.01]
        else:
            yeni = [min(2.0, uc * f) for f in (1.1, 1.25, 1.5)]
        yeni = [v for v in yeni if v > uc + 1e-9]
    return sorted({round(float(v), 5) for v in yeni})


def _tara(veriler: list[dict], izgara, topla: dict, log, etiket: str) -> None:
    """Her kayıt ve her eşik için DER bileşenlerini topla[t]'ye ekler (kayıt başı önbellekli)."""
    from scipy.cluster.hierarchy import fcluster

    t_bas = time.perf_counter()
    for i, v in enumerate(veriler, 1):
        for t in izgara:
            if v["sabit"] is not None:
                anahtar, segler = b"sabit", v["sabit"]
            else:
                et = kumeleme._yeniden_numarala(fcluster(v["Z"], t=t, criterion="distance"))
                anahtar, segler = et.tobytes(), None
            if anahtar not in v["bellek"]:  # aynı etiketler -> aynı sonuç
                if segler is None:
                    segler = cikti.segmentleri_birlestir(v["P"], et)
                v["bellek"][anahtar] = degerlendirme._segmentleri_degerlendir(
                    v["ref"], segler, COLLAR, TOLERANS_SN)
            s = v["bellek"][anahtar]
            a = topla.setdefault(t, {"hata": 0.0, "ref": 0.0, "tp": 0, "fp": 0, "fn": 0, "jer": []})
            a["hata"] += s["yanlis_alarm_sn"] + s["kacirilan_sn"] + s["karisiklik_sn"]
            a["ref"] += s["referans_sn"]
            a["tp"] += s["tp"]
            a["fp"] += s["fp"]
            a["fn"] += s["fn"]
            if not math.isnan(s["jer"]):
                a["jer"].append(s["jer"])
        if i % 10 == 0 or i == len(veriler):
            gecen = time.perf_counter() - t_bas
            log(f"[{etiket} {i}/{len(veriler)}] {len(izgara)} eşik  geçen {_sure(gecen)}  "
                f"kalan ~{_sure(gecen / i * (len(veriler) - i))}")


def _satir(yontem: str, t: float, a: dict) -> dict:
    return {"yontem": yontem, "esik": t,
            "der": a["hata"] / a["ref"] if a["ref"] > 0 else math.nan,
            "jer_ort": float(np.mean(a["jer"])) if a["jer"] else math.nan,
            "mikro_f1": degisim_modeli._mikro_f1(a["tp"], a["fp"], a["fn"])}


def _en_iyi(satirlar: list[dict]) -> dict:
    """En düşük toplam DER; eşitlikte büyük eşik (daha az küme)."""
    return min(satirlar, key=lambda r: (np.nan_to_num(r["der"], nan=np.inf), -r["esik"]))


def esik_tara(zip_yol: str, rttm_dizin: str, kayitlar: list[str], onbellek: str,
              yontemler=("mfcc", "ecapa"), izgara=None, gunluk_dizin: str | None = None,
              log=_yaz) -> dict:
    """AHC eşiğini tarar; her yöntem için toplam DER'i en düşük eşiği seçer (eğitim yarısı).

    izgara None ise aday eşikler veriden üretilir: yöntem başına tüm kayıtların AHC
    birleşme yüksekliklerinin YUZDELIK_SAYISI yüzdeliği (%1-%99, üst uca doğru sık).
    En iyi eşik ızgaranın kenarına düşerse o yönde genişletilip yeniden taranır (en çok
    GENISLETME_TURU kez) ve uyarı yazılır. izgara verilirse yalnızca o değerler taranır.
    MFCC, eğitim kayıtlarının tüm pencerelerinden hesaplanan genel ortalama/std ile
    standartlaştırılır (oznitelik.olcek_hesapla); istatistik sonuçta "mfcc_olcek" alanındadır.
    Bağlantı ağacı kayıt başına bir kez kurulur, her eşikte yalnızca kesilir.
    Dönüş (esikler.json içeriği): {"mfcc", "ecapa", "mfcc_olcek", "tarama": [...],
    "izgara": {yontem: [...]}, "genisletme": {yontem: n}, "kayit_sayisi", "hatali", "uyarilar",
    "collar", "tarih"}.
    Hata: ValueError (bir yöntem için hiç kayıt işlenemedi).
    """
    gunluk_dizin = gunluk_dizin or onbellek
    sonuc = {"kayit_sayisi": len(kayitlar), "mfcc_olcek": None, "tarama": [], "izgara": {},
             "genisletme": {}, "hatali": {}, "uyarilar": [], "collar": COLLAR,
             "tarih": datetime.datetime.now().isoformat(timespec="seconds")}
    for yontem in yontemler:
        # 1) Önbellekten kayıtları topla
        veriler, hatali = [], 0
        with SesKaynagi(zip_yol) as kaynak:
            for i, kayit in enumerate(kayitlar, 1):
                try:
                    ref = cikti.rttm_oku(_referans_yolu(rttm_dizin, kayit))
                    oz = ozellik_al(kaynak, kayit, onbellek, (yontem,))
                    veriler.append({"kayit": kayit, "ref": ref, "bolgeler": oz["bolgeler"],
                                    "P": oz[yontem]["pencereler"], "X": oz[yontem]["X"], "bellek": {}})
                except Exception as e:
                    hatali += 1
                    _hata_yaz(gunluk_dizin, f"esik-tara-{yontem}", kayit, e)
                    log(f"  HATA {kayit}: {type(e).__name__}: {e}")
        sonuc["hatali"][yontem] = hatali
        if not veriler:
            raise ValueError(f"{yontem}: eşik taraması için işlenebilen kayıt yok")
        log(f"{yontem}: {len(veriler)} kayıt yüklendi ({hatali} hatalı)")

        # 2) MFCC genel standartlaştırma (eğitim yarısının tüm pencereleri)
        olcek = None
        if yontem == "mfcc":
            olcek = oznitelik.olcek_hesapla([v["X"] for v in veriler])
            sonuc["mfcc_olcek"] = olcek
            log(f"mfcc: genel ölçek {olcek['pencere_sayisi']} pencereden hesaplandı")

        # 3) Kayıt başına bağlantı ağacı
        yukseklikler = []
        for v in veriler:
            X = oznitelik.olcekle(v["X"], olcek)
            v["sabit"], v["Z"] = None, None
            if not v["bolgeler"] or ana.kisa_konusma(v["bolgeler"]) or len(v["P"]) < 2:
                v["sabit"] = ana.hat_segmentleri(v["bolgeler"], v["P"], X, None, None)  # eşikten bağımsız
            else:
                v["Z"] = kumeleme._baglanti(X)
                yukseklikler.append(v["Z"][:, 2])
            v["X"] = None  # bellek
        yukseklikler = np.concatenate(yukseklikler) if yukseklikler else np.zeros(0)

        # 4) Tara; kenara düşerse genişlet
        aday = sorted(float(t) for t in izgara) if izgara is not None else yuzdelik_izgarasi(yukseklikler)
        if not aday:
            aday = [kumeleme.VARSAYILAN_ESIK]
        topla: dict = {}
        _tara(veriler, aday, topla, log, yontem)
        genisletme = 0
        while izgara is None and yukseklikler.size and genisletme < GENISLETME_TURU:
            sirali = sorted(topla)
            en_iyi = _en_iyi([_satir(yontem, t, topla[t]) for t in sirali])["esik"]
            kenar = "alt" if en_iyi == sirali[0] else "ust" if en_iyi == sirali[-1] else None
            if kenar is None:
                break
            yeni = _genislet(sirali, yukseklikler, kenar, genisletme)
            if not yeni:
                break
            genisletme += 1
            uyari = (f"{yontem}: en iyi eşik ({en_iyi}) ızgaranın {'alt' if kenar == 'alt' else 'üst'} "
                     f"kenarındaydı; {len(yeni)} yeni değerle genişletildi (tur {genisletme})")
            sonuc["uyarilar"].append(uyari)
            log(f"UYARI: {uyari}")
            _tara(veriler, yeni, topla, log, f"{yontem} genişletme {genisletme}")
        sirali = sorted(topla)
        satirlar = [_satir(yontem, t, topla[t]) for t in sirali]
        en_iyi = _en_iyi(satirlar)
        if len(sirali) > 1 and en_iyi["esik"] in (sirali[0], sirali[-1]):
            sonuc["uyarilar"].append(f"{yontem}: en iyi eşik hâlâ ızgaranın kenarında ({en_iyi['esik']})")
        sonuc[yontem] = en_iyi["esik"]
        sonuc["izgara"][yontem] = sirali
        sonuc["genisletme"][yontem] = genisletme
        sonuc["tarama"] += satirlar
        log(f"{yontem}: en iyi eşik {en_iyi['esik']:.5f}  DER {en_iyi['der']:.4f}  "
            f"JER {en_iyi['jer_ort']:.4f}  mikro F1 {en_iyi['mikro_f1']:.4f}  "
            f"({len(sirali)} aday, aralık {sirali[0]:.4f}-{sirali[-1]:.4f})")
    for u in sonuc["uyarilar"]:
        log(f"UYARI: {u}")
    return sonuc


def _json_temiz(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _json_temiz(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_temiz(v) for v in x]
    return x


def esikleri_yaz(yol: str, sonuc: dict) -> None:
    """esikler.json yazar (NaN -> null)."""
    os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
    gecici = yol + ".yaziliyor"
    with open(gecici, "w", encoding="utf-8") as f:
        json.dump(_json_temiz(sonuc), f, ensure_ascii=False, indent=2)
    os.replace(gecici, yol)


def esikleri_oku(yol: str) -> dict:
    with open(yol, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Değişim modeli eğitimi
# ---------------------------------------------------------------------------

def model_egit(zip_yol: str, rttm_dizin: str, kayitlar: list[str], onbellek: str, model_yolu: str,
               gunluk_dizin: str | None = None, log=_yaz) -> dict:
    """Eğitim yarısının önbelleğinden değişim modelini eğitir ve model_yolu'na kaydeder.

    MFCC özelliği (mfcc_cos), eğitim kayıtlarının tüm MFCC pencerelerinden hesaplanan
    genel ölçekle standartlaştırılır; ölçek pakete "mfcc_olcek" olarak yazılır ve tahminde
    (E5) aynısı kullanılır. Aynı eğitim listesiyle esik_tara'nın ölçeğiyle aynıdır.
    """
    gunluk_dizin = gunluk_dizin or onbellek
    hamlar = []
    with SesKaynagi(zip_yol) as kaynak:
        for i, kayit in enumerate(kayitlar, 1):
            try:
                ref = cikti.rttm_oku(_referans_yolu(rttm_dizin, kayit))
                oz = ozellik_al(kaynak, kayit, onbellek, ("ecapa", "mfcc"))
                hamlar.append((kayit, ref, oz))
            except Exception as e:
                _hata_yaz(gunluk_dizin, "model-egit", kayit, e)
                log(f"  HATA {kayit}: {type(e).__name__}: {e}")
            if i % 20 == 0 or i == len(kayitlar):
                log(f"  veri hazırlığı {i}/{len(kayitlar)}")
    olcek = oznitelik.olcek_hesapla([oz["mfcc"]["X"] for _, _, oz in hamlar])
    veriler = []
    for kayit, ref, oz in hamlar:
        P = oz["ecapa"]["pencereler"]
        if len(P) >= 2:
            veriler.append(degisim_modeli.kayit_verisi(
                kayit, P, oz["ecapa"]["X"], oznitelik.olcekle(oz["mfcc"]["X"], olcek), oz["bolgeler"], ref))
    paket = degisim_modeli.egit(veriler, log=log)
    paket["mfcc_olcek"] = olcek
    degisim_modeli.kaydet(paket, model_yolu)
    log(f"Model kaydedildi: {model_yolu}")
    return paket


# ---------------------------------------------------------------------------
# Komut satırı
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.toplu", description="Toplu çalıştırma ve deneyler")
    is_ = ap.add_mutually_exclusive_group(required=True)
    is_.add_argument("--deney", choices=sorted(DENEYLER), help="çalıştırılacak deney")
    is_.add_argument("--onbellek-doldur", action="store_true", help="yalnızca VAD + öznitelik önbelleği")
    is_.add_argument("--esik-tara", action="store_true", help="AHC eşik taraması -> --esikler dosyası")
    is_.add_argument("--model-egit", action="store_true", help="değişim modelini eğit -> --model dosyası")
    ap.add_argument("--zip", required=True, help="voxconverse_dev_wav.zip")
    ap.add_argument("--rttm-dizin", help="referans RTTM klasörü (voxconverse/dev)")
    ap.add_argument("--liste", nargs="+", required=True, help="kayıt listesi dosya(lar)ı")
    ap.add_argument("--cikti-dizin", help="RTTM, CSV, ozet.csv ve hata günlüğü klasörü")
    ap.add_argument("--onbellek", required=True, help="öznitelik önbelleği klasörü")
    ap.add_argument("--yontemler", nargs="+", default=["mfcc", "ecapa"], choices=["mfcc", "ecapa"])
    ap.add_argument("--esik", type=float, help="AHC eşiği (E2/E4/E5); --esikler'den önceliklidir")
    ap.add_argument("--esikler", help="esikler.json (E2/E4/E5 okur, --esik-tara yazar)")
    ap.add_argument("--model", help="degisim_modeli.joblib (E5 okur, --model-egit yazar)")
    a = ap.parse_args(argv)

    try:
        kayitlar = listeleri_oku(a.liste)
        if a.onbellek_doldur:
            s = onbellek_doldur(a.zip, kayitlar, a.onbellek, a.yontemler, a.cikti_dizin)
            return 1 if s["hatali"] else 0
        if not a.rttm_dizin:
            ap.error("--rttm-dizin gerekli")
        if a.esik_tara:
            if not a.esikler:
                ap.error("--esik-tara için --esikler (çıktı yolu) gerekli")
            s = esik_tara(a.zip, a.rttm_dizin, kayitlar, a.onbellek, a.yontemler, gunluk_dizin=a.cikti_dizin)
            esikleri_yaz(a.esikler, s)
            _yaz(f"Yazıldı: {a.esikler}  " + json.dumps({y: s.get(y) for y in a.yontemler}))
            return 1 if any(s["hatali"].values()) else 0
        if a.model_egit:
            if not a.model:
                ap.error("--model-egit için --model (çıktı yolu) gerekli")
            model_egit(a.zip, a.rttm_dizin, kayitlar, a.onbellek, a.model, a.cikti_dizin)
            return 0
        if not a.cikti_dizin:
            ap.error("--deney için --cikti-dizin gerekli")
        yontem, kip = DENEYLER[a.deney]
        esikler = esikleri_oku(a.esikler) if a.esikler else {}
        esik = a.esik
        if esik is None and kip != "sayi":
            esik = esikler.get(yontem)
        olcek = (esikler.get("mfcc_olcek") or varsayilanlar.mfcc_olcek()) if yontem == "mfcc" else None
        s = deney_calistir(a.zip, a.rttm_dizin, kayitlar, a.deney, a.cikti_dizin, a.onbellek, esik, a.model,
                           mfcc_olcek=olcek)
        return 1 if s["hatali"] else 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Girdi hatası: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"İç hata: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
