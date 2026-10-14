"""Konuşma etkinliği tespiti (VAD). Sahibi: Abdullah. Sözleşme: docs/arayuz.md"""
import numpy as np


def konusma_bolgeleri(ses: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Konuşma olan bölgeleri bulur (Silero VAD; min. konuşma 250 ms, min. sessizlik 300 ms).

    Girdi: ses (float32, (n_ornek,), [-1, 1]); sr (Hz, 16000).
    Çıktı: [(baslangic_sn, bitis_sn), ...] saniye cinsinden, zamana göre sıralı,
           birbiriyle örtüşmeyen. Konuşma yoksa boş liste.
    """
    raise NotImplementedError
