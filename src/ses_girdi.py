"""Ses okuma ve ön işleme. Sahibi: Abdullah. Sözleşme: docs/arayuz.md

soundfile / librosa fonksiyon içinde import edilir; ön işleme (on_isle)
yalnızca numpy + scipy kullanır.
"""
import os
from math import gcd

import numpy as np

SR = 16000  # örnekleme hızı (Hz), tüm hat boyunca sabit
HEDEF_RMS_DBFS = -20.0
TEPE_SINIRI = 0.99
DESTEKLENEN = (".wav", ".flac", ".mp3")


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
