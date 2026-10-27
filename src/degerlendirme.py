"""Değerlendirme (DER/JER, değişim noktası F1). Sahibi: Sefkan. Sözleşme: docs/arayuz.md

pyannote.metrics fonksiyon içinde import edilir; modül yalnızca standart
kütüphane ile import edilebilir.
"""
import math
from bisect import bisect_left, bisect_right

from . import cikti


def _degisim_eslestir(ref: list[float], hyp: list[float], tolerans_sn: float) -> tuple[int, int, int]:
    """Değişim noktalarını eşleştirir ve (tp, fp, fn) döndürür.

    Her referans noktası en fazla bir tahminle, her tahmin en fazla bir
    referansla eşleşir. |r - h| <= tolerans_sn olan çiftler mesafeye göre
    artan sırada (eşitlikte önce erken referans) açgözlü biçimde eşleşir.
    """
    hyp_sirali = sorted(enumerate(hyp), key=lambda x: x[1])
    zamanlar = [h for _, h in hyp_sirali]
    pay = tolerans_sn + 1e-9
    ciftler = []
    for i, r in enumerate(ref):  # yalnızca tolerans penceresindeki tahminlere bak
        for k in range(bisect_left(zamanlar, r - pay), bisect_right(zamanlar, r + pay)):
            j, h = hyp_sirali[k]
            d = abs(r - h)
            if d <= pay:
                ciftler.append((d, i, j))
    ciftler.sort()
    ref_kullanildi, hyp_kullanildi = set(), set()
    for _, i, j in ciftler:
        if i in ref_kullanildi or j in hyp_kullanildi:
            continue
        ref_kullanildi.add(i)
        hyp_kullanildi.add(j)
    tp = len(ref_kullanildi)
    return tp, len(hyp) - tp, len(ref) - tp


def _annotation(segmentler):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for i, (a, b, etiket) in enumerate(segmentler):
        ann[Segment(a, b), i] = etiket
    return ann


def degerlendir(
    ref_rttm: str,
    hyp_rttm: str,
    collar: float = 0.5,
    tolerans_sn: float = 0.5,
) -> dict:
    """Referans ve tahmin RTTM'ini karşılaştırır (pyannote.metrics).

    Girdi: ref_rttm, hyp_rttm (dosya yolları); collar (sn, pyannote'da tam
           genişlik: 0.5 = ±0,25 sn); tolerans_sn (değişim noktası eşleşme toleransı, sn).
    Çıktı: sözlük (ayrıntı: docs/arayuz.md 3.6):
           der, jer       : oran (0.25 = %25); referansta konuşma yoksa NaN.
           yanlis_alarm_sn, kacirilan_sn, karisiklik_sn, referans_sn :
                            DER bileşenleri (sn); referans_sn = pyannote 'total'.
           f1, precision, recall, tp, fp, fn : değişim noktası eşleşmesi.
           Örtüşen konuşma dahildir (skip_overlap=False).
    Hata: FileNotFoundError (dosya yok), ValueError (RTTM biçimi bozuk).
    """
    if collar < 0 or tolerans_sn < 0:
        raise ValueError("collar ve tolerans_sn negatif olamaz")
    return _segmentleri_degerlendir(cikti.rttm_oku(ref_rttm), cikti.rttm_oku(hyp_rttm), collar, tolerans_sn)


def _segmentleri_degerlendir(ref, hyp, collar: float = 0.5, tolerans_sn: float = 0.5) -> dict:
    """degerlendir'in bellek içi hâli: ref ve hyp [(bas_sn, bit_sn, etiket)] listeleri.

    Dönüş ve tanımlar degerlendir ile aynıdır. Toplu eşik taramasında her eşik
    için RTTM dosyası yazıp okumamak için kullanılır.
    """

    # --- DER / JER (pyannote.metrics) ---
    sonuc = {
        "der": math.nan,
        "jer": math.nan,
        "yanlis_alarm_sn": 0.0,
        "kacirilan_sn": 0.0,
        "karisiklik_sn": 0.0,
        "referans_sn": 0.0,
    }
    bitis = max((b for _, b, _ in ref + hyp), default=0.0)
    if bitis > 0:
        from pyannote.core import Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

        ref_ann, hyp_ann = _annotation(ref), _annotation(hyp)
        uem = Timeline([Segment(0.0, bitis)])  # dosya başından en geç bitişe kadar
        ayrinti = DiarizationErrorRate(collar=collar, skip_overlap=False)(
            ref_ann, hyp_ann, uem=uem, detailed=True
        )
        sonuc["yanlis_alarm_sn"] = float(ayrinti["false alarm"])
        sonuc["kacirilan_sn"] = float(ayrinti["missed detection"])
        sonuc["karisiklik_sn"] = float(ayrinti["confusion"])
        sonuc["referans_sn"] = float(ayrinti["total"])
        if ayrinti["total"] > 0:
            # Referans boşken pyannote DER'i 0/1 verir, JER sıfıra böler: NaN bırakılır.
            sonuc["der"] = float(ayrinti["diarization error rate"])
            sonuc["jer"] = float(
                JaccardErrorRate(collar=collar, skip_overlap=False)(ref_ann, hyp_ann, uem=uem)
            )

    # --- Değişim noktası F1 (tanım: cikti.tek_konusmaci_segmentleri) ---
    ref_d = cikti.degisim_noktalari(ref)
    hyp_d = cikti.degisim_noktalari(hyp)
    tp, fp, fn = _degisim_eslestir(ref_d, hyp_d, tolerans_sn)
    precision = tp / len(hyp_d) if hyp_d else 1.0
    recall = tp / len(ref_d) if ref_d else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    sonuc.update({"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn})
    return sonuc
