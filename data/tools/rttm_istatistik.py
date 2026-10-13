# -*- coding: utf-8 -*-
"""
rttm_istatistik.py - RTTM dosyalarından kayıt istatistikleri çıkarır.

Grup_68 - "Ses Kayıtlarında Konuşmacı Değişimlerinin Tespiti" projesi.
Yalnızca Python standart kütüphanesini kullanır (Python 3.8+).

Her kayıt (RTTM'deki 2. alan, dosya adı) için hesaplananlar:
  - sure_sn          : Toplam süre = son segmentin bitişi (başlangıç + süre).
                       Not: Gerçek ses dosyası bundan biraz uzun olabilir;
                       RTTM yalnızca konuşma olan yerleri içerir.
  - konusmaci        : Farklı konuşmacı etiketi sayısı (8. alan).
  - konusma_sn       : En az bir konuşmacının konuştuğu toplam süre
                       (segmentlerin birleşimi; örtüşen yerler bir kez sayılır).
  - ortusme_sn       : Aynı anda en az iki konuşmacının konuştuğu süre.
  - ortusme_orani    : ortusme_sn / konusma_sn (0-1 arası).
  - degisim          : Konuşmacı değişim noktası sayısı (tanım aşağıda).

KONUŞMACI DEĞİŞİM NOKTASI TANIMI
  1) Zaman ekseni, segment sınırlarına göre küçük aralıklara bölünür.
  2) Her aralıkta etkin (konuşan) konuşmacı kümesine bakılır:
       - Tam olarak 1 konuşmacı varsa aralık "NON-OVERLAP" (tek konuşmacılı)
         aralıktır.
       - 2 veya daha fazla konuşmacı varsa aralık ÖRTÜŞEN bölgedir ve
         değişim noktası hesabında TAMAMEN ATILIR (ne segment sayılır ne de
         araya giren bir konuşmacı olarak değerlendirilir).
       - Hiç konuşmacı yoksa (sessizlik) aralık atılır.
  3) Kalan tek konuşmacılı aralıklar zaman sırasına dizilir; aynı
     konuşmacıya ait ardışık aralıklar (aralarında sessizlik veya örtüşme
     olsa bile) tek bir NON-OVERLAP segmentte birleştirilir.
  4) Zaman sırasına göre ardışık iki NON-OVERLAP segmentin konuşmacıları
     farklıysa, İKİNCİ segmentin başlangıcı bir değişim noktasıdır.
  Örnek: A konuşuyor -> A ile B örtüşüyor -> yalnız B konuşuyor ise değişim
  noktası örtüşmenin başı değil, B'nin tek başına konuştuğu kısmın başıdır.

İSTEĞE BAĞLI PENCERE ANALİZİ (--pencere SN)
  Örnek parça seçimine yardım eder: kayıttaki her segment başlangıcından
  başlayan SN saniyelik pencereler denenir; pencerede en az --min-konusma
  saniye konuşan konuşmacılar sayılır. En çok konuşmacı içeren ilk pencere
  raporlanır (pencere_bas_sn, pencere_konusmaci).

Kullanım örnekleri:
  python rttm_istatistik.py <rttm_dosyasi_veya_klasoru>
  python rttm_istatistik.py <klasor> --csv <cikti.csv> --pencere 90
  python rttm_istatistik.py <dosya.rttm> --degisim-listesi
"""

import argparse
import bisect
import csv
import os
import sys
from collections import defaultdict


def ms(saniye_metni):
    """Saniye cinsinden metni tamsayı milisaniyeye çevirir (kayan nokta
    hatalarını önlemek için tüm hesaplar milisaniye ile yapılır)."""
    return int(round(float(saniye_metni) * 1000))


def rttm_oku(yol):
    """RTTM dosyasını okur; {kayit_adi: [(bas_ms, bit_ms, konusmaci), ...]}
    döndürür. Yalnızca SPEAKER satırları dikkate alınır.
    RTTM alanları: 1 tür, 2 dosya, 3 kanal, 4 başlangıç (sn), 5 SÜRE (sn),
    6 <NA>, 7 <NA>, 8 konuşmacı, 9 <NA>, 10 <NA>."""
    kayitlar = defaultdict(list)
    with open(yol, encoding="utf-8") as f:
        for satir_no, satir in enumerate(f, 1):
            alanlar = satir.split()
            if not alanlar or alanlar[0] != "SPEAKER":
                continue
            if len(alanlar) < 8:
                raise ValueError(f"{yol}:{satir_no}: eksik alanlı RTTM satırı")
            bas = ms(alanlar[3])
            sure = ms(alanlar[4])
            if sure <= 0:
                continue
            kayitlar[alanlar[1]].append((bas, bas + sure, alanlar[7]))
    for segler in kayitlar.values():
        segler.sort()
    return kayitlar


def temel_araliklar(segler):
    """Zaman eksenini segment sınırlarına göre böler.
    [(bas_ms, bit_ms, etkin_konusmaci_kumesi), ...] döndürür (boş küme =
    sessizlik). Aynı konuşmacının üst üste binen segmentleri tek sayılır."""
    olaylar = defaultdict(list)
    for bas, bit, k in segler:
        olaylar[bas].append((k, +1))
        olaylar[bit].append((k, -1))
    zamanlar = sorted(olaylar)
    etkin = defaultdict(int)
    sonuc = []
    for i, t in enumerate(zamanlar):
        for k, d in olaylar[t]:
            etkin[k] += d
            if etkin[k] == 0:
                del etkin[k]
        if i + 1 < len(zamanlar):
            sonuc.append((t, zamanlar[i + 1], frozenset(etkin)))
    return sonuc


def non_overlap_segmentler(araliklar):
    """Tek konuşmacılı aralıklardan birleşik NON-OVERLAP segmentleri üretir.
    Örtüşen (>=2 konuşmacı) ve sessiz aralıklar atılır; aynı konuşmacının
    ardışık aralıkları tek segmentte birleştirilir.
    [(bas_ms, bit_ms, konusmaci), ...] döndürür."""
    segmentler = []
    for bas, bit, kume in araliklar:
        if len(kume) != 1:
            continue
        (k,) = kume
        if segmentler and segmentler[-1][2] == k:
            segmentler[-1] = (segmentler[-1][0], bit, k)
        else:
            segmentler.append((bas, bit, k))
    return segmentler


def degisim_noktalari(segler):
    """Değişim noktalarını [(zaman_ms, onceki_konusmaci, yeni_konusmaci)]
    olarak döndürür (tanım için modül açıklamasına bakın)."""
    nov = non_overlap_segmentler(temel_araliklar(segler))
    return [(nov[i][0], nov[i - 1][2], nov[i][2]) for i in range(1, len(nov))]


def en_iyi_pencere(segler, pencere_ms, min_konusma_ms):
    """Her segment başlangıcından başlayan pencereleri dener; pencerede en
    az min_konusma_ms konuşan konuşmacı sayısı en yüksek olan ilk pencereyi
    (bas_ms, konusmaci_sayisi) olarak döndürür."""
    if not segler:
        return 0, 0
    baslar = [s[0] for s in segler]
    en_uzun = max(b - a for a, b, _ in segler)
    en_iyi = (segler[0][0], -1)
    for pb in sorted(set(baslar)):
        pbit = pb + pencere_ms
        sure = defaultdict(int)
        j = bisect.bisect_left(baslar, pb - en_uzun)
        while j < len(segler) and segler[j][0] < pbit:
            a, b, k = segler[j]
            kesisim = min(b, pbit) - max(a, pb)
            if kesisim > 0:
                sure[k] += kesisim
            j += 1
        sayi = sum(1 for v in sure.values() if v >= min_konusma_ms)
        if sayi > en_iyi[1]:
            en_iyi = (pb, sayi)
    return en_iyi


def istatistik(segler, pencere_ms=None, min_konusma_ms=2000):
    """Tek kaydın istatistik sözlüğünü döndürür (süreler saniye)."""
    araliklar = temel_araliklar(segler)
    konusma = sum(b - a for a, b, k in araliklar if len(k) >= 1)
    ortusme = sum(b - a for a, b, k in araliklar if len(k) >= 2)
    sonuc = {
        "sure_sn": round(max(b for _, b, _ in segler) / 1000, 2),
        "konusmaci": len({k for _, _, k in segler}),
        "konusma_sn": round(konusma / 1000, 2),
        "ortusme_sn": round(ortusme / 1000, 2),
        "ortusme_orani": round(ortusme / konusma, 4) if konusma else 0.0,
        "degisim": len(degisim_noktalari(segler)),
    }
    if pencere_ms:
        pb, sayi = en_iyi_pencere(segler, pencere_ms, min_konusma_ms)
        sonuc["pencere_bas_sn"] = round(pb / 1000, 2)
        sonuc["pencere_konusmaci"] = sayi
    return sonuc


def rttm_dosyalari(yol):
    if os.path.isdir(yol):
        return sorted(
            os.path.join(yol, a) for a in os.listdir(yol) if a.lower().endswith(".rttm")
        )
    return [yol]


def tablo_yazdir(satirlar, sutunlar):
    genislik = {s: max(len(s), *(len(str(r[s])) for r in satirlar)) for s in sutunlar}
    print("  ".join(s.ljust(genislik[s]) for s in sutunlar))
    print("  ".join("-" * genislik[s] for s in sutunlar))
    for r in satirlar:
        print("  ".join(str(r[s]).ljust(genislik[s]) for s in sutunlar))


def main():
    ap = argparse.ArgumentParser(description="RTTM dosyalarından süre, konuşmacı "
                                 "sayısı, örtüşme ve değişim noktası istatistikleri.")
    ap.add_argument("girdi", help="RTTM dosyası veya RTTM'lerin bulunduğu klasör")
    ap.add_argument("--csv", help="Sonuçların yazılacağı CSV dosyası (isteğe bağlı)")
    ap.add_argument("--pencere", type=float,
                    help="Pencere analizi için pencere uzunluğu (sn), örn. 90")
    ap.add_argument("--min-konusma", type=float, default=2.0,
                    help="Pencerede konuşmacı sayılması için gereken en az "
                         "konuşma süresi (sn, varsayılan 2.0)")
    ap.add_argument("--sirala", default="kayit",
                    help="Sıralama sütunu (varsayılan: kayit)")
    ap.add_argument("--degisim-listesi", action="store_true",
                    help="Her kaydın değişim noktalarını tek tek yazdır")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pencere_ms = int(args.pencere * 1000) if args.pencere else None
    satirlar = []
    for dosya in rttm_dosyalari(args.girdi):
        for kayit, segler in rttm_oku(dosya).items():
            if not segler:
                continue
            satir = {"kayit": kayit}
            satir.update(istatistik(segler, pencere_ms, int(args.min_konusma * 1000)))
            satirlar.append(satir)
            if args.degisim_listesi:
                print(f"# {kayit} degisim noktalari (sn, onceki -> yeni)")
                for t, once, yeni in degisim_noktalari(segler):
                    print(f"{t / 1000:.3f}\t{once} -> {yeni}")

    if not satirlar:
        print("RTTM satırı bulunamadı.", file=sys.stderr)
        return 1
    satirlar.sort(key=lambda r: r.get(args.sirala, r["kayit"]))
    sutunlar = list(satirlar[0].keys())
    tablo_yazdir(satirlar, sutunlar)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sutunlar)
            w.writeheader()
            w.writerows(satirlar)
        print(f"\nCSV yazildi: {args.csv} ({len(satirlar)} kayit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
