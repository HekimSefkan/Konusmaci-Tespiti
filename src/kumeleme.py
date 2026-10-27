"""Konuşmacı kümeleme. Sahibi: Cevat. Sözleşme: docs/arayuz.md"""
import numpy as np


VARSAYILAN_ESIK = None  # kosinüs mesafesi eşiği; VoxConverse dev üzerinde ayarlanacak (KG-3)


def kumele(X: np.ndarray, konusmaci_sayisi: int | None = None, esik: float | None = None) -> np.ndarray:
    """Pencere vektörlerini konuşmacılara kümeler (ortalama bağlantılı AHC, kosinüs mesafesi).

    Girdi: X, şekil (N, D), float32; konusmaci_sayisi: bilinen konuşmacı
           sayısı, None ise mesafe eşiğiyle otomatik kestirilir (tek konuşmacıda
           1 küme çıkabilir, TG-6); esik: AHC kosinüs mesafesi eşiği, None ise
           VARSAYILAN_ESIK.
    Çıktı: etiketler, şekil (N,), int, değerler 0..K-1. N == 0 ise boş dizi,
           N == 1 ise [0].
    Hata: ValueError (X iki boyutlu değil veya konusmaci_sayisi < 1).
    """
    raise NotImplementedError
