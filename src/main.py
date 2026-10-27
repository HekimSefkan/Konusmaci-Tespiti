"""Komut satırı: python -m src.main <girdi_ses> -o <cikti.rttm> [--yontem ecapa|mfcc]

Varsayılan yöntem ECAPA (Hafta 4'te en iyi sonuç, E4). Eşik verilmezse yöntemin
ayarlanmış eşiği (src/varsayilanlar.json) kullanılır; MFCC'de genel ölçek uygulanır.

Çıkış kodları: 0 başarılı, 1 girdi hatası (bulunamadı/bozuk/desteklenmeyen), 2 iç hata.
"""
import argparse
import os
import sys
import warnings

from . import cikti, kumeleme, oznitelik, ses_girdi, vad, varsayilanlar

MIN_KONUSMA_SN = 1.5  # bundan kısa toplam konuşmada uyarı verilir
VARSAYILAN_YONTEM = "ecapa"


def kisa_konusma(bolgeler: list[tuple[float, float]]) -> bool:
    """Toplam konuşma MIN_KONUSMA_SN'den kısa mı (kümeleme atlanır, tek konuşmacı)."""
    return sum(b - a for a, b in bolgeler) < MIN_KONUSMA_SN


def hat_segmentleri(bolgeler, pencereler, X, konusmaci_sayisi=None, esik=None):
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
    etiketler = kumeleme.kumele(X, konusmaci_sayisi, esik)
    return cikti.segmentleri_birlestir(pencereler, etiketler)


def calistir(girdi: str, cikti_yolu: str, yontem: str, konusmaci_sayisi: int | None,
             esik: float | None = None) -> None:
    """Hattı sırayla çalıştırır ve RTTM yazar.

    esik None ve konuşmacı sayısı verilmemişse yöntemin ayarlanmış eşiği
    (varsayilanlar.esik) kullanılır. MFCC vektörleri kümelemeden önce genel ölçekle
    standartlaştırılır (varsayilanlar.mfcc_olcek; yoksa ham MFCC ve uyarı).
    """
    if esik is None and konusmaci_sayisi is None:
        esik = varsayilanlar.esik(yontem)
    dosya_id = os.path.splitext(os.path.basename(girdi))[0]
    ses, sr = ses_girdi.ses_yukle(girdi)
    bolgeler = vad.konusma_bolgeleri(ses, sr)
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
    segmentler = hat_segmentleri(bolgeler, pencereler, X, konusmaci_sayisi, esik)
    cikti.rttm_yaz(segmentler, dosya_id, cikti_yolu)
    if not segmentler:  # sessiz dosya: boş RTTM
        return
    print(cikti.tablo_yazdir(segmentler))
    print("Değişim noktaları (sn):", cikti.degisim_noktalari(segmentler))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.main", description="Konuşmacı değişimi tespiti")
    ap.add_argument("girdi", help="girdi ses dosyası (wav/mp3/flac)")
    ap.add_argument("-o", "--cikti", required=True, help="çıktı .rttm yolu")
    ap.add_argument("--yontem", choices=["ecapa", "mfcc"], default=VARSAYILAN_YONTEM,
                    help="öznitelik yöntemi (varsayılan: ecapa)")
    ap.add_argument("--konusmaci-sayisi", type=int, default=None, help="biliniyorsa konuşmacı sayısı")
    ap.add_argument("--esik", type=float, default=None, help="AHC kosinüs mesafesi eşiği (verilmezse yönteme göre src/varsayilanlar.json)")
    a = ap.parse_args(argv)
    try:
        calistir(a.girdi, a.cikti, a.yontem, a.konusmaci_sayisi, a.esik)
    except (FileNotFoundError, ValueError) as e:
        print(f"Girdi hatası ({a.girdi}): {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"İç hata: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
