"""Konuşmacı kümeleme. Sahibi: Cevat. Sözleşme: docs/arayuz.md"""
import numpy as np


def kumele(X: np.ndarray, konusmaci_sayisi: int | None = None) -> np.ndarray:
    """Pencere vektörlerini konuşmacılara kümeler.

    Girdi: X, şekil (N, D), float32; konusmaci_sayisi: bilinen konuşmacı
           sayısı, None ise otomatik kestirilir.
    Çıktı: etiketler, şekil (N,), int, değerler 0..K-1. N == 0 ise boş dizi,
           N == 1 ise [0].
    Hata: ValueError (X iki boyutlu değil veya konusmaci_sayisi < 1).
    """
    raise NotImplementedError
