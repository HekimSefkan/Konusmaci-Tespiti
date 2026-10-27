"""Ses okuma ve ön işleme. Sahibi: Abdullah. Sözleşme: docs/arayuz.md

soundfile / librosa fonksiyon içinde import edilir; ön işleme (on_isle)
yalnızca numpy + scipy kullanır.
"""
import io
import os
import struct
from math import gcd

import numpy as np

SR = 16000  # örnekleme hızı (Hz), tüm hat boyunca sabit
HEDEF_RMS_DBFS = -20.0
TEPE_SINIRI = 0.99
DESTEKLENEN = (".wav", ".flac", ".mp3")


def wav_denetle(kaynak, ad: str | None = None) -> float:
    """WAV (RIFF) başlığını standart kütüphaneyle denetler; başlıktaki süreyi (sn) döndürür.

    kaynak: dosya yolu (str) veya dosyanın baytları (bytes; ör. zip içinden okunmuş).
    Kesik/bozuk sayılır ve ValueError verilir:
      - RIFF/WAVE imzası, fmt veya data bölümü yok; fmt alanları geçersiz (kanal,
        örnekleme hızı, blok hizası 0; blok hizası kanal x örnek genişliğiyle uyuşmuyor);
        data'dan önceki bir bölüm dosyanın sonunu aşıyor;
      - data bölümünün bildirdiği boyut, dosyada gerçekten kalan bayttan bir örnek
        çerçevesinden (blok hizası) fazla büyük (dosya yarıda kesilmiş).
    Mesajda dosya adı, başlıktaki süre ve gerçek süre geçer.
    """
    if isinstance(kaynak, (bytes, bytearray)):
        f, toplam, ad = io.BytesIO(kaynak), len(kaynak), ad or "<bellek>"
        return _wav_denetle(f, toplam, ad)
    ad = ad or os.path.basename(kaynak)
    with open(kaynak, "rb") as f:
        return _wav_denetle(f, os.path.getsize(kaynak), ad)


def _wav_denetle(f, toplam: int, ad: str) -> float:
    def bozuk(neden):
        return ValueError(f"{ad}: bozuk WAV ({neden})")

    bas = f.read(12)
    if len(bas) < 12 or bas[:4] != b"RIFF" or bas[8:12] != b"WAVE":
        raise bozuk("RIFF/WAVE imzası yok")
    fmt = None
    konum = 12
    while konum + 8 <= toplam:
        f.seek(konum)
        kimlik, boyut = struct.unpack("<4sI", f.read(8))
        govde = konum + 8
        if kimlik == b"fmt ":
            if boyut < 16 or govde + 16 > toplam:
                raise bozuk("fmt bölümü eksik")
            bicim, kanal, hiz, bayt_hizi, blok, bit = struct.unpack("<HHIIHH", f.read(16))
            if kanal == 0 or hiz == 0 or blok == 0 or bayt_hizi == 0:
                raise bozuk(f"fmt alanları geçersiz (kanal={kanal}, hız={hiz}, blok={blok})")
            if bit and bit % 8 == 0 and blok != kanal * (bit // 8):
                raise bozuk(f"blok hizası {blok}, kanal x örnek genişliği {kanal * bit // 8} değil")
            fmt = (hiz, bayt_hizi, blok)
        elif kimlik == b"data":
            if fmt is None:
                raise bozuk("data bölümü fmt bölümünden önce")
            hiz, bayt_hizi, blok = fmt
            kalan = toplam - govde
            basliktaki, gercek = boyut / bayt_hizi, max(kalan, 0) / bayt_hizi
            if boyut - kalan > blok:
                raise ValueError(f"{ad}: bozuk/kesik WAV: başlıkta {basliktaki:.2f} sn veri yazıyor, "
                                 f"dosyada {gercek:.2f} sn var")
            return basliktaki
        if govde + boyut > toplam:
            raise bozuk(f"'{kimlik.decode('latin-1')}' bölümü dosya sonunu aşıyor")
        konum = govde + boyut + (boyut & 1)  # bölümler çift bayta hizalıdır
    raise bozuk("fmt veya data bölümü yok")


def on_isle(veri: np.ndarray, sr_girdi: int) -> np.ndarray:
    """Ham sesi hat biçimine getirir: mono, 16 kHz, RMS ~ -20 dBFS, tepe <= 0,99.

    Girdi: veri (n_ornek,) veya (n_ornek, kanal); sr_girdi (Hz).
    Çıktı: float32, (n_ornek,). Sessiz (sıfır) ses değiştirilmeden döner.
    Hata: ValueError (boş ses, sonsuz/NaN değer, geçersiz örnekleme hızı).
    """
    x = np.asarray(veri, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)  # kanal ortalaması
    elif x.ndim != 1:
        raise ValueError(f"beklenmeyen ses şekli {x.shape}")
    if x.size == 0:
        raise ValueError("ses boş (0 örnek)")
    if not np.all(np.isfinite(x)):
        raise ValueError("seste NaN/sonsuz değer var")
    if sr_girdi <= 0:
        raise ValueError(f"geçersiz örnekleme hızı {sr_girdi}")

    if sr_girdi != SR:
        from scipy.signal import resample_poly

        g = gcd(int(sr_girdi), SR)
        x = resample_poly(x, SR // g, int(sr_girdi) // g)

    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms > 1e-8:
        kazanc = 10 ** (HEDEF_RMS_DBFS / 20) / rms
        tepe = float(np.max(np.abs(x))) * kazanc
        if tepe > TEPE_SINIRI:
            kazanc *= TEPE_SINIRI / tepe
        x = x * kazanc
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def ses_yukle(yol: str) -> tuple[np.ndarray, int]:
    """Ses dosyasını okur; mono, 16 kHz, RMS ~ -20 dBFS'e normalize eder.

    Girdi: yol (wav/mp3/flac dosya yolu).
    Çıktı: (ses, sr). ses: float32, şekil (n_ornek,), değerler [-1, 1];
           sr: her zaman 16000 (Hz). Tepe değer 0,99'u geçmez (gerekirse
           RMS -20 dBFS'in altında kalır).
    Hata: FileNotFoundError (dosya yok), ValueError (bozuk/desteklenmeyen
          biçim; mesajda dosya adı geçer).
    """
    ad = os.path.basename(yol)
    if not os.path.isfile(yol):
        raise FileNotFoundError(f"{ad}: dosya bulunamadı")
    uzanti = os.path.splitext(yol)[1].lower()
    if uzanti not in DESTEKLENEN:
        raise ValueError(f"{ad}: desteklenmeyen biçim '{uzanti}' (wav/flac/mp3)")

    if uzanti == ".wav":
        wav_denetle(yol, ad)  # soundfile kesik dosyayı sessizce kısaltıp okur; önce başlığı denetle
    if uzanti in (".wav", ".flac"):
        import soundfile as sf

        try:
            veri, sr = sf.read(yol, dtype="float32", always_2d=True)  # (n, kanal)
        except Exception as e:
            raise ValueError(f"{ad}: okunamadı ({e})") from e
    else:
        import librosa

        try:
            veri, sr = librosa.load(yol, sr=None, mono=False)  # (kanal, n) veya (n,)
        except Exception as e:
            raise ValueError(f"{ad}: okunamadı ({e})") from e
        if veri.ndim == 2:
            veri = veri.T

    try:
        return on_isle(veri, int(sr)), SR
    except ValueError as e:
        raise ValueError(f"{ad}: {e}") from e
