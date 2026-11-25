"""Komut satırı: python -m src.main <girdi_ses> -o <cikti.rttm> [--yontem ecapa|mfcc]

Varsayılan yöntem ECAPA (Hafta 4'te en iyi sonuç, E4). Eşik verilmezse yöntemin
ayarlanmış eşiği (src/varsayilanlar.json) kullanılır; MFCC'de genel ölçek uygulanır.

Çıkış kodları: 0 başarılı, 1 girdi hatası (bulunamadı/bozuk/desteklenmeyen), 2 iç hata.
"""
import argparse
import os
import sys
import time
import warnings

from . import cikti, kumeleme, oznitelik, ses_girdi, vad, varsayilanlar

MIN_KONUSMA_SN = 1.5  # bundan kısa toplam konuşmada uyarı verilir
VARSAYILAN_YONTEM = "ecapa"


def kisa_konusma(bolgeler: list[tuple[float, float]]) -> bool:
    """Toplam konuşma MIN_KONUSMA_SN'den kısa mı (kümeleme atlanır, tek konuşmacı)."""
    return sum(b - a for a, b in bolgeler) < MIN_KONUSMA_SN


def hat_segmentleri(bolgeler, pencereler, X, konusmaci_sayisi=None, esik=None):
    """Sözleşmedeki varsayılan kümeleme hattı; imza ve çağrılar sabit kalır."""
    return _hat_segmentleri(bolgeler, pencereler, X, konusmaci_sayisi, esik, 0)


def _hat_segmentleri(bolgeler, pencereler, X, konusmaci_sayisi=None, esik=None,
                     min_destek_pencere: int = 0):
    """VAD bölgeleri, pencereler ve vektörlerden konuşmacı segmentlerini üretir.

    Konuşma yoksa []. Toplam konuşma MIN_KONUSMA_SN'den kısaysa kümeleme
    atlanır, her bölge "spk00" olur (pencereler ve X kullanılmaz). Aksi halde
    kumele + segmentleri_birlestir. main ve toplu çalıştırma aynı kuralı
    kullansın diye tek yerde durur.
    """
    if not bolgeler:
        return []
    if kisa_konusma(bolgeler):
        return [(a, b, "spk00") for a, b in bolgeler]
    if min_destek_pencere:
        etiketler = kumeleme.kumele_destekli(
            X, konusmaci_sayisi, esik, min_destek_pencere
        )
    else:
        etiketler = kumeleme.kumele(X, konusmaci_sayisi, esik)
    return cikti.segmentleri_birlestir(pencereler, etiketler)


def _hat_ayrintili(girdi: str, yontem: str = VARSAYILAN_YONTEM, konusmaci_sayisi: int | None = None,
                   esik: float | None = None, maks_sure_sn: float | None = ses_girdi.MAKS_SURE_SN,
                   min_destek_pencere: int = 0) -> dict:
    """Hattı çalıştırır; segmentlerle birlikte ara bilgileri de döndürür (dosya yazmaz).

    esik None ve konuşmacı sayısı verilmemişse yöntemin ayarlanmış eşiği
    (varsayilanlar.esik) kullanılır. MFCC vektörleri kümelemeden önce genel ölçekle
    standartlaştırılır (varsayilanlar.mfcc_olcek; yoksa ham MFCC ve uyarı).
    Dönüş: {"segmentler", "ses_sn", "ses", "sr", "bolgeler" (VAD), "pencereler", "X" (kümelenen
    vektörler; kısa/sessiz kayıtta None), "esik" (kullanılan; konuşmacı sayısı verildiyse None),
    "sureler": {"okuma", "vad", "oznitelik", "kumeleme"} (sn)}.
    Hata: FileNotFoundError, ValueError (ses_girdi.ses_yukle ile aynı).
    """
    if esik is None and konusmaci_sayisi is None:
        esik = varsayilanlar.esik(yontem)
    t0 = time.perf_counter()
    ses, sr = ses_girdi.ses_yukle(girdi, maks_sure_sn)
    t1 = time.perf_counter()
    bolgeler = vad.konusma_bolgeleri(ses, sr)
    t2 = time.perf_counter()
    pencereler, X = [], None
    if bolgeler and kisa_konusma(bolgeler):
        warnings.warn(f"{girdi}: konuşma {MIN_KONUSMA_SN} sn'den kısa; tek konuşmacı varsayıldı")
    elif bolgeler:
        pencereler = oznitelik.pencerele(bolgeler)
        X = oznitelik.vektorler(ses, sr, pencereler, yontem)
        if yontem == "mfcc":
            olcek = varsayilanlar.mfcc_olcek()
            if olcek is None:
                warnings.warn("MFCC ölçeği src/varsayilanlar.json'da yok; ham MFCC kümeleniyor")
            X = oznitelik.olcekle(X, olcek)
    t3 = time.perf_counter()
    segmentler = _hat_segmentleri(
        bolgeler, pencereler, X, konusmaci_sayisi, esik,
        min_destek_pencere,
    )
    t4 = time.perf_counter()
    sureler = {"okuma": t1 - t0, "vad": t2 - t1, "oznitelik": t3 - t2, "kumeleme": t4 - t3}
    return {"segmentler": segmentler, "ses_sn": len(ses) / sr, "ses": ses, "sr": sr, "bolgeler": bolgeler,
            "pencereler": pencereler, "X": X, "esik": None if konusmaci_sayisi is not None else esik,
            "sureler": sureler}


def hat_ayrintili(girdi: str, yontem: str = VARSAYILAN_YONTEM, konusmaci_sayisi: int | None = None,
                  esik: float | None = None, maks_sure_sn: float | None = ses_girdi.MAKS_SURE_SN) -> dict:
    """Public default pipeline; the established function signature is unchanged."""
    return _hat_ayrintili(girdi, yontem, konusmaci_sayisi, esik, maks_sure_sn, 0)


def segmentleri_hesapla(girdi: str, yontem: str = VARSAYILAN_YONTEM, konusmaci_sayisi: int | None = None,
                        esik: float | None = None, maks_sure_sn: float | None = ses_girdi.MAKS_SURE_SN
                        ) -> tuple[list[tuple[float, float, str]], float]:
    """Hattı çalıştırır (hat_ayrintili); dosya yazmadan (segmentler, ses süresi sn) döndürür.

    Hata: FileNotFoundError, ValueError (ses_girdi.ses_yukle ile aynı).
    """
    r = hat_ayrintili(girdi, yontem, konusmaci_sayisi, esik, maks_sure_sn)
    return r["segmentler"], r["ses_sn"]


def _segmentleri_hesapla_destekli(girdi, yontem, konusmaci_sayisi, esik, maks_sure_sn, min_destek_pencere):
    r = _hat_ayrintili(girdi, yontem, konusmaci_sayisi, esik, maks_sure_sn, min_destek_pencere)
    return r["segmentler"], r["ses_sn"]


def calistir(girdi: str, cikti_yolu: str, yontem: str, konusmaci_sayisi: int | None,
             esik: float | None = None, maks_sure_sn: float | None = ses_girdi.MAKS_SURE_SN) -> None:
    """Hattı sırayla çalıştırır (segmentleri_hesapla), RTTM yazar ve tabloyu basar."""
    dosya_id = os.path.splitext(os.path.basename(girdi))[0]
    segmentler, _ = segmentleri_hesapla(girdi, yontem, konusmaci_sayisi, esik, maks_sure_sn)
    _rttm_yaz_ve_goster(segmentler, girdi, cikti_yolu)


def _rttm_yaz_ve_goster(segmentler, girdi, cikti_yolu):
    dosya_id = os.path.splitext(os.path.basename(girdi))[0]
    cikti.rttm_yaz(segmentler, dosya_id, cikti_yolu)
    if not segmentler:  # sessiz dosya: boş RTTM
        return
    print(cikti.tablo_yazdir(segmentler))
    print("Değişim noktaları (sn):", cikti.degisim_noktalari(segmentler))


def _calistir_destekli(girdi, cikti_yolu, yontem, konusmaci_sayisi, esik, maks_sure_sn,
                       min_destek_pencere):
    segmentler, _ = _segmentleri_hesapla_destekli(
        girdi, yontem, konusmaci_sayisi, esik, maks_sure_sn, min_destek_pencere
    )
    _rttm_yaz_ve_goster(segmentler, girdi, cikti_yolu)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.main", description="Konuşmacı değişimi tespiti")
    ap.add_argument("girdi", help="girdi ses dosyası (wav/mp3/flac)")
    ap.add_argument("-o", "--cikti", required=True, help="çıktı .rttm yolu")
    ap.add_argument("--yontem", choices=["ecapa", "mfcc"], default=VARSAYILAN_YONTEM,
                    help="öznitelik yöntemi (varsayılan: ecapa)")
    ap.add_argument("--konusmaci-sayisi", type=int, default=None, help="biliniyorsa konuşmacı sayısı")
    ap.add_argument("--min-destek-pencere", type=int, default=0,
                    help="otomatik kümelemede en çok bu pencereli kümeleri yakın ana kümeye birleştir (varsayılan: 0, kapalı)")
    ap.add_argument("--esik", type=float, default=None, help="AHC kosinüs mesafesi eşiği (verilmezse yönteme göre src/varsayilanlar.json)")
    ap.add_argument("--maks-sure", type=float, default=ses_girdi.MAKS_SURE_SN,
                    help=f"en uzun girdi süresi, sn (varsayılan {ses_girdi.MAKS_SURE_SN}; 0 = sınırsız)")
    a = ap.parse_args(argv)
    maks = a.maks_sure if a.maks_sure and a.maks_sure > 0 else None
    if a.min_destek_pencere < 0:
        ap.error("--min-destek-pencere sıfır veya pozitif olmalı")
    if a.min_destek_pencere and a.konusmaci_sayisi is not None:
        ap.error("--min-destek-pencere yalnızca otomatik konuşmacı sayısı kipinde kullanılabilir")
    try:
        oznitelik.cpu_yapilandir()
        if a.min_destek_pencere:
            _calistir_destekli(
                a.girdi, a.cikti, a.yontem, a.konusmaci_sayisi, a.esik, maks,
                a.min_destek_pencere,
            )
        else:
            calistir(a.girdi, a.cikti, a.yontem, a.konusmaci_sayisi, a.esik, maks)
    except (FileNotFoundError, ValueError) as e:
        print(f"Girdi hatası ({a.girdi}): {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"İç hata: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
