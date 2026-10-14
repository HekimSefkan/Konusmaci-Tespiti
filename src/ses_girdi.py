"""Ses okuma ve ön işleme. Sahibi: Abdullah. Sözleşme: docs/arayuz.md"""
import numpy as np

SR = 16000  # örnekleme hızı (Hz), tüm hat boyunca sabit


def ses_yukle(yol: str) -> tuple[np.ndarray, int]:
    """Ses dosyasını okur; mono, 16 kHz, RMS ~ -20 dBFS'e normalize eder.

    Girdi: yol (wav/mp3/flac dosya yolu).
    Çıktı: (ses, sr). ses: float32, şekil (n_ornek,), değerler [-1, 1];
           sr: her zaman 16000 (Hz).
    Hata: FileNotFoundError (dosya yok), ValueError (bozuk/desteklenmeyen
          biçim; mesajda dosya adı geçer).
    """
    raise NotImplementedError
