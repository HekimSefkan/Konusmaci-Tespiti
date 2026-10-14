"""Segmentleme ve öznitelik/gömme çıkarımı. Sahibi: Abdullah. Sözleşme: docs/arayuz.md"""
import numpy as np


def pencerele(
    bolgeler: list[tuple[float, float]],
    pencere_sn: float = 1.5,
    kaydirma_sn: float = 0.75,
) -> list[tuple[float, float]]:
    """Konuşma bölgelerini örtüşen sabit uzunlukta pencerelere böler.

    Girdi: bolgeler [(baslangic_sn, bitis_sn)]; pencere_sn ve kaydirma_sn saniye.
    Çıktı: [(baslangic_sn, bitis_sn)], zamana göre sıralı. Pencereler bölge
           sınırını aşmaz; pencere_sn'den kısa bölge tek pencere olur.
           Boş girdi için boş liste.
    Hata: ValueError (pencere_sn <= 0 veya kaydirma_sn <= 0).
    """
    raise NotImplementedError


def vektorler(
    ses: np.ndarray,
    sr: int,
    pencereler: list[tuple[float, float]],
    yontem: str = "mfcc",
) -> np.ndarray:
    """Her pencere için bir vektör üretir.

    Girdi: ses (float32, (n_ornek,)); sr (16000); pencereler [(bas_sn, bit_sn)];
           yontem "mfcc" (13 katsayı ortalama+std, D=26) veya "ecapa" (D=192).
    Çıktı: np.ndarray, şekil (N, D), float32; N = len(pencereler).
           Boş pencere listesinde şekil (0, D).
    Hata: ValueError (bilinmeyen yontem).
    """
    raise NotImplementedError
