"""Değerlendirme (DER/JER, değişim noktası F1). Sahibi: Sefkan. Sözleşme: docs/arayuz.md"""


def degerlendir(
    ref_rttm: str,
    hyp_rttm: str,
    collar: float = 0.5,
    tolerans_sn: float = 0.5,
) -> dict:
    """Referans ve tahmin RTTM'ini karşılaştırır (pyannote.metrics).

    Girdi: ref_rttm, hyp_rttm (dosya yolları); collar (sn, pyannote'da tam
           genişlik: 0.5 = ±0,25 sn); tolerans_sn (değişim noktası eşleşme toleransı, sn).
    Çıktı: {"der": float, "jer": float, "f1": float}; der/jer oran (0.25 = %25),
           f1 değişim noktası F1'i (0-1).
    Hata: FileNotFoundError (dosya yok), ValueError (RTTM biçimi bozuk).
    """
    raise NotImplementedError
