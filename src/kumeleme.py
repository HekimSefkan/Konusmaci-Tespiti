"""Konuşmacı kümeleme. Sahibi: Cevat. Sözleşme: docs/arayuz.md"""
import numpy as np

from . import varsayilanlar

# Kosinüs mesafesi eşiği (0-2), Hafta 4'te VoxConverse dev eğitim yarısında yöntem
# başına ayarlandı (src/varsayilanlar.json). kumele yöntemi bilmediği için esik=None'da
# varsayılan yöntemin (ECAPA) eşiğini kullanır; MFCC için varsayilanlar.esik("mfcc").
VARSAYILAN_ESIK = varsayilanlar.esik("ecapa")

_EPS = 1e-12


def _yeniden_numarala(etiketler: np.ndarray) -> np.ndarray:
    """Etiketleri ilk görünme sırasına göre 0..K-1 yapar."""
    eslesme: dict[int, int] = {}
    sonuc = np.empty(len(etiketler), dtype=int)
    for i, e in enumerate(etiketler):
        sonuc[i] = eslesme.setdefault(int(e), len(eslesme))
    return sonuc


def kumele(X: np.ndarray, konusmaci_sayisi: int | None = None, esik: float | None = None) -> np.ndarray:
    """Legacy public entry point: AHC clustering without support cleanup."""
    return _kumele(X, konusmaci_sayisi, esik, min_destek_pencere=0)


def kumele_destekli(
    X: np.ndarray,
    konusmaci_sayisi: int | None = None,
    esik: float | None = None,
    min_destek_pencere: int = 1,
) -> np.ndarray:
    """Opt-in AHC entry point with cleanup for low-support automatic clusters."""
    return _kumele(X, konusmaci_sayisi, esik, min_destek_pencere)


def _kumele(
    X: np.ndarray,
    konusmaci_sayisi: int | None,
    esik: float | None,
    min_destek_pencere: int,
) -> np.ndarray:
    """Pencere vektörlerini konuşmacılara kümeler (ortalama bağlantılı AHC, kosinüs mesafesi).

    Girdi: X, şekil (N, D), float32; konusmaci_sayisi: bilinen konuşmacı
           sayısı, None ise mesafe eşiğiyle otomatik kestirilir (tek konuşmacıda
           1 küme çıkabilir, TG-6); esik: AHC kosinüs mesafesi eşiği, None ise
           VARSAYILAN_ESIK. min_destek_pencere > 0 ise otomatik küme sayısı
           kipinde bu sayıda veya daha az pencereye dayanan kümeler, yalnızca
           en yakın ana küme merkezi mevcut AHC eşiği içindeyse ona katılır.
           Varsayılan 0 mevcut davranışı korur.
    Çıktı: etiketler, şekil (N,), int, değerler 0..K-1. N == 0 ise boş dizi,
           N == 1 ise [0].
    Hata: ValueError (X iki boyutlu değil veya konusmaci_sayisi < 1).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X iki boyutlu olmalı, şekil {X.shape}")
    if konusmaci_sayisi is not None and konusmaci_sayisi < 1:
        raise ValueError(f"konusmaci_sayisi >= 1 olmalı, verilen {konusmaci_sayisi}")
    if not isinstance(min_destek_pencere, (int, np.integer)) or min_destek_pencere < 0:
        raise ValueError(f"min_destek_pencere >= 0 tamsayı olmalı, verilen {min_destek_pencere}")
    if min_destek_pencere and konusmaci_sayisi is not None:
        raise ValueError("min_destek_pencere yalnızca otomatik konuşmacı sayısı kipinde kullanılabilir")
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    from scipy.cluster.hierarchy import fcluster

    Z = _baglanti(X)
    if konusmaci_sayisi is not None:
        etiketler = fcluster(Z, t=konusmaci_sayisi, criterion="maxclust")
    else:
        t = VARSAYILAN_ESIK if esik is None else esik
        etiketler = fcluster(Z, t=t, criterion="distance")
        if min_destek_pencere:
            etiketler = _destek_kumelerini_birlestir(X, etiketler, min_destek_pencere, t)
    return _yeniden_numarala(etiketler)


def _destek_kumelerini_birlestir(
    X: np.ndarray, etiketler: np.ndarray, en_cok_pencere: int, esik: float
) -> np.ndarray:
    """Küçük destekli kümeleri en yakın uygun ana küme merkezine bağlar.

    Yalnızca başlangıçta en çok ``en_cok_pencere`` örneği olan kümeler ele
    alınır. Hedefler daha büyük kümelerdir ve her birinin L2-normalize edilmiş
    vektör merkezi kullanılır. En yakın merkezle kosinüs mesafesi ``esik``
    değerini aşarsa küme korunur. En az iki küme ve bir hedef bulunmadıkça
    etiketler değişmez.
    """
    etiketler = np.asarray(etiketler, dtype=int).copy()
    X = np.asarray(X, dtype=np.float64)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), _EPS)
    kume, sayi = np.unique(etiketler, return_counts=True)
    sayilar = dict(zip(kume.tolist(), sayi.tolist()))
    kucukler = [int(k) for k in kume if sayilar[int(k)] <= en_cok_pencere]
    ana_kumeler = [int(k) for k in kume if sayilar[int(k)] > en_cok_pencere]
    if not kucukler or not ana_kumeler:
        return etiketler

    merkezler = {}
    for k in kume:
        m = Xn[etiketler == k].mean(axis=0)
        merkezler[int(k)] = m / max(float(np.linalg.norm(m)), _EPS)

    for k in kucukler:
        hedef = min(
            ana_kumeler,
            key=lambda h: float(np.clip(1.0 - merkezler[k] @ merkezler[h], 0.0, 2.0)),
        )
        uzaklik = float(np.clip(1.0 - merkezler[k] @ merkezler[hedef], 0.0, 2.0))
        if uzaklik <= esik:
            etiketler[etiketler == k] = hedef
    return etiketler


def _baglanti(X: np.ndarray) -> np.ndarray:
    """L2 normalizasyonu + kosinüs mesafesi + ortalama bağlantı ağacı (N >= 2).

    Eşik taramasında ağaç kayıt başına bir kez kurulup her eşikte yalnızca
    kesilsin diye ayrı tutulur.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    X = np.asarray(X, dtype=np.float64)
    # Sıfır vektör sıfır kalır (her şeye mesafesi 1 olur).
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), _EPS)
    D = np.clip(1.0 - Xn @ Xn.T, 0.0, 2.0)  # kosinüs mesafesi
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return linkage(squareform(D, checks=False), method="average")
