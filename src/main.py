"""Komut satırı: python -m src.main <girdi_ses> -o <cikti.rttm>

Çıkış kodları: 0 başarılı, 1 girdi hatası (bulunamadı/bozuk/desteklenmeyen), 2 iç hata.
"""
import argparse
import os
import sys
import warnings

from . import cikti, kumeleme, oznitelik, ses_girdi, vad

MIN_KONUSMA_SN = 1.5  # bundan kısa toplam konuşmada uyarı verilir


def calistir(girdi: str, cikti_yolu: str, yontem: str, konusmaci_sayisi: int | None) -> None:
    """Hattı sırayla çalıştırır ve RTTM yazar."""
    dosya_id = os.path.splitext(os.path.basename(girdi))[0]
    ses, sr = ses_girdi.ses_yukle(girdi)
    bolgeler = vad.konusma_bolgeleri(ses, sr)
    if not bolgeler:  # sessiz dosya: boş RTTM
        cikti.rttm_yaz([], dosya_id, cikti_yolu)
        return
    if sum(b - a for a, b in bolgeler) < MIN_KONUSMA_SN:
        warnings.warn(f"{girdi}: konuşma {MIN_KONUSMA_SN} sn'den kısa; tek konuşmacı varsayıldı")
    pencereler = oznitelik.pencerele(bolgeler)
    X = oznitelik.vektorler(ses, sr, pencereler, yontem)
    etiketler = kumeleme.kumele(X, konusmaci_sayisi)
    segmentler = cikti.segmentleri_birlestir(pencereler, etiketler)
    cikti.rttm_yaz(segmentler, dosya_id, cikti_yolu)
    print(cikti.tablo_yazdir(segmentler))
    print("Değişim noktaları (sn):", cikti.degisim_noktalari(segmentler))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.main", description="Konuşmacı değişimi tespiti")
    ap.add_argument("girdi", help="girdi ses dosyası (wav/mp3/flac)")
    ap.add_argument("-o", "--cikti", required=True, help="çıktı .rttm yolu")
    ap.add_argument("--yontem", choices=["mfcc", "ecapa"], default="mfcc")
    ap.add_argument("--konusmaci-sayisi", type=int, default=None, help="biliniyorsa konuşmacı sayısı")
    a = ap.parse_args(argv)
    try:
        calistir(a.girdi, a.cikti, a.yontem, a.konusmaci_sayisi)
    except (FileNotFoundError, ValueError) as e:
        print(f"Girdi hatası ({a.girdi}): {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"İç hata: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
