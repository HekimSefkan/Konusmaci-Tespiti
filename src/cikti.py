"""Segment birleştirme, değişim noktaları, RTTM okuma/yazma. Sahibi: Cevat. Sözleşme: docs/arayuz.md

Bu modül yalnızca standart kütüphaneyi kullanır (numpy import etmez); böylece
data/tools/rttm_istatistik.py değişim noktası tanımını buradan alabilir.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # yalnızca tip ipucu için
    import numpy as np

# segment: (baslangic_sn, bitis_sn, etiket) ; etiket "spk00", "spk01", ...

_EPS = 1e-6  # saniye karşılaştırmalarında kayan nokta payı


# ---------------------------------------------------------------------------
# Değişim noktası tanımı (tek kaynak). Tüm zamanlar TAMSAYI milisaniyedir.
# ---------------------------------------------------------------------------

def temel_araliklar(segmentler_ms):
    """Zaman eksenini segment sınırlarına göre küçük aralıklara böler.

    Girdi: [(bas_ms, bit_ms, etiket)], tamsayı ms; sıra önemli değil.
    Çıktı: [(bas_ms, bit_ms, etkin_etiketler: frozenset)], zamana göre sıralı;
           boş küme sessizliktir. Aynı etiketin üst üste binen segmentleri tek sayılır.
    """
    olaylar = defaultdict(list)
    for bas, bit, etiket in segmentler_ms:
        if bit <= bas:
            continue
        olaylar[bas].append((etiket, +1))
        olaylar[bit].append((etiket, -1))
    zamanlar = sorted(olaylar)
    etkin = defaultdict(int)
    sonuc = []
    for i, t in enumerate(zamanlar):
        for etiket, d in olaylar[t]:
            etkin[etiket] += d
            if etkin[etiket] == 0:
                del etkin[etiket]
        if i + 1 < len(zamanlar):
            sonuc.append((t, zamanlar[i + 1], frozenset(etkin)))
    return sonuc


def tek_konusmaci_segmentleri(segmentler_ms):
    """Değişim noktası hesabında kullanılan tek konuşmacılı (NON-OVERLAP) segmentler.

    Girdi: [(bas_ms, bit_ms, etiket)], tamsayı ms.
    Çıktı: [(bas_ms, bit_ms, etiket)], zamana göre sıralı. Örtüşen (>= 2 konuşmacı)
           ve sessiz aralıklar atılır; aynı konuşmacının ardışık tek konuşmacılı
           aralıkları (arada sessizlik veya örtüşme olsa bile) tek segmentte birleşir.
           Böylece ardışık iki çıktı segmentinin etiketi her zaman farklıdır ve
           her segmentin (ilki hariç) başlangıcı bir değişim noktasıdır.
    """
    sonuc = []
    for bas, bit, kume in temel_araliklar(segmentler_ms):
        if len(kume) != 1:
            continue
        (etiket,) = kume
        if sonuc and sonuc[-1][2] == etiket:
            sonuc[-1] = (sonuc[-1][0], bit, etiket)
        else:
            sonuc.append((bas, bit, etiket))
    return sonuc


def _ms(saniye: float) -> int:
    return int(round(float(saniye) * 1000))


# ---------------------------------------------------------------------------
# Sözleşme fonksiyonları
# ---------------------------------------------------------------------------

def segmentleri_birlestir(
    pencereler: list[tuple[float, float]],
    etiketler: np.ndarray,
    ada_esik_sn: float = 1.0,
    bosluk_sn: float = 0.3,
) -> list[tuple[float, float, str]]:
    """Pencere etiketlerini konuşmacı segmentlerine çevirir.

    Girdi: pencereler [(bas_sn, bit_sn)] zamana göre sıralı;
           etiketler np.ndarray (N,), int (N == len(pencereler)).
    Çıktı: [(bas_sn, bit_sn, "spkNN")], zamana göre sıralı. Ardışık aynı etiketli
           pencereler birleşir; örtüşen farklı etiketli pencerelerde sınır
           örtüşmenin ortasıdır. Süresi ada_esik_sn'den kısa ve iki yanı aynı
           etiketli koşular komşu etikete katılır (KG-10). Aynı etiketli ve
           aralarındaki boşluk <= bosluk_sn olan segmentler birleşir (KG-11).
           Etiketler konuşma süresine göre azalan sırada
           spk00, spk01, ... diye yeniden adlandırılır.
    Hata: ValueError (uzunluklar eşit değil).
    """
    etiket_listesi = [int(e) for e in etiketler]
    if len(pencereler) != len(etiket_listesi):
        raise ValueError(
            f"pencere sayısı ({len(pencereler)}) ile etiket sayısı ({len(etiket_listesi)}) eşit değil"
        )
    if not pencereler:
        return []

    # 1) Pencereleri segmentlere çevir: aynı etiketli örtüşen/bitişik pencereler
    #    birleşir, farklı etiketli örtüşmede sınır örtüşmenin ortasıdır.
    segler: list[list] = []
    for (s, e), etiket in zip(pencereler, etiket_listesi):
        s, e = float(s), float(e)
        if not segler:
            segler.append([s, e, etiket])
            continue
        onceki = segler[-1]
        if etiket == onceki[2] and s <= onceki[1] + _EPS:
            onceki[1] = max(onceki[1], e)
        elif etiket != onceki[2] and s < onceki[1]:
            orta = (s + onceki[1]) / 2
            onceki[1] = orta
            segler.append([orta, max(e, orta), etiket])
        else:
            segler.append([s, e, etiket])
    segler = _bitisikleri_birlestir(segler)

    # 2) Ada yumuşatma (A-B-A). Koşu = aynı etiketli ardışık segmentler; aradaki
    #    boşluk dikkate alınmaz. En kısa aday önce, değişiklik kalmayana dek.
    while True:
        kosular = _kosular(segler)
        adaylar = []
        for k in range(1, len(kosular) - 1):
            i0, i1, etiket, sure = kosular[k]
            if (sure < ada_esik_sn and kosular[k - 1][2] == kosular[k + 1][2]
                    and kosular[k - 1][2] != etiket):
                adaylar.append((sure, k))
        if not adaylar:
            break
        _, k = min(adaylar)
        i0, i1, _, _ = kosular[k]
        yeni_etiket = kosular[k - 1][2]
        for i in range(i0, i1 + 1):
            segler[i][2] = yeni_etiket
        segler = _bitisikleri_birlestir(segler)

    # 3) Boşluk birleştirme: aynı etiketli komşular arasında <= bosluk_sn boşluk.
    birlesik: list[list] = []
    for seg in segler:
        if birlesik and birlesik[-1][2] == seg[2] and seg[0] - birlesik[-1][1] <= bosluk_sn + _EPS:
            birlesik[-1][1] = max(birlesik[-1][1], seg[1])
        else:
            birlesik.append(list(seg))

    # 4) Etiketleri toplam konuşma süresine göre adlandır (eşitlikte ilk görünen önce).
    toplam: dict[int, float] = {}
    ilk: dict[int, int] = {}
    for i, (a, b, etiket) in enumerate(birlesik):
        toplam[etiket] = toplam.get(etiket, 0.0) + (b - a)
        ilk.setdefault(etiket, i)
    sira = sorted(toplam, key=lambda k: (-round(toplam[k], 6), ilk[k]))
    ad = {k: f"spk{i:02d}" for i, k in enumerate(sira)}

    sonuc = []
    for a, b, etiket in birlesik:
        a, b = round(a, 3), round(b, 3)
        if b > a:
            sonuc.append((a, b, ad[etiket]))
    return sonuc


def _bitisikleri_birlestir(segler: list[list]) -> list[list]:
    """Aynı etiketli ve bitişik/örtüşen ardışık segmentleri birleştirir."""
    sonuc: list[list] = []
    for seg in segler:
        if sonuc and sonuc[-1][2] == seg[2] and seg[0] <= sonuc[-1][1] + _EPS:
            sonuc[-1][1] = max(sonuc[-1][1], seg[1])
        else:
            sonuc.append(list(seg))
    return sonuc


def _kosular(segler: list[list]) -> list[tuple[int, int, int, float]]:
    """[(ilk_indis, son_indis, etiket, toplam_sure)]; aynı etiketli ardışık segmentler bir koşudur."""
    kosular = []
    for i, (a, b, etiket) in enumerate(segler):
        if kosular and kosular[-1][2] == etiket:
            i0, _, _, sure = kosular[-1]
            kosular[-1] = (i0, i, etiket, sure + (b - a))
        else:
            kosular.append((i, i, etiket, b - a))
    return kosular


def degisim_noktalari(segmentler: list[tuple[float, float, str]]) -> list[float]:
    """Konuşmacı değişim noktalarını bulur.

    Girdi: segmentler (bkz. segmentleri_birlestir çıktısı).
    Çıktı: saniye cinsinden artan liste; ardışık iki segmentin etiketi
           farklıysa ikincinin başlangıcı. 0 veya 1 segmentte boş liste.
           Örtüşen segment içeren girdide (ör. referans RTTM) tanım
           tek_konusmaci_segmentleri ile aynıdır: örtüşen aralıklar atılır.
           Hesap milisaniye çözünürlüğündedir.
    """
    ms_segler = [(_ms(a), _ms(b), etiket) for a, b, etiket in segmentler]
    nov = tek_konusmaci_segmentleri(ms_segler)
    return [nov[i][0] / 1000 for i in range(1, len(nov))]


def rttm_yaz(
    segmentler: list[tuple[float, float, str]], dosya_id: str, yol: str
) -> None:
    """Segmentleri RTTM dosyasına yazar (10 alan, süre = bitiş - başlangıç, 3 ondalık).

    Girdi: segmentler; dosya_id (uzantısız ses dosyası adı); yol (çıktı .rttm).
    Çıktı: yok. Segment listesi boşsa boş dosya yazılır.
    Hata: OSError (yazılamadı).
    """
    satirlar = []
    for a, b, etiket in segmentler:
        a_ms, b_ms = _ms(a), _ms(b)
        satirlar.append(
            f"SPEAKER {dosya_id} 1 {a_ms / 1000:.3f} {(b_ms - a_ms) / 1000:.3f} "
            f"<NA> <NA> {etiket} <NA> <NA>\n"
        )
    with open(yol, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(satirlar)


def rttm_oku(yol: str) -> list[tuple[float, float, str]]:
    """RTTM dosyasını okur.

    Girdi: yol (.rttm).
    Çıktı: [(bas_sn, bit_sn, etiket)], dosyadaki sırayla (bit = bas + süre).
           Boş dosyada boş liste. Boş satırlar, ';' veya '#' ile başlayan
           satırlar ve SPEAKER dışındaki satırlar atlanır; süresi 0 olan
           segment atlanır.
    Hata: FileNotFoundError (dosya yok), ValueError (satır biçimi bozuk:
          8'den az alan, sayısal olmayan zaman veya negatif süre).
    """
    sonuc = []
    with open(yol, encoding="utf-8") as f:
        for satir_no, satir in enumerate(f, 1):
            alanlar = satir.split()
            if not alanlar or alanlar[0].startswith((";", "#")) or alanlar[0] != "SPEAKER":
                continue
            if len(alanlar) < 8:
                raise ValueError(f"{yol}:{satir_no}: eksik alanlı RTTM satırı")
            try:
                bas = float(alanlar[3])
                sure = float(alanlar[4])
            except ValueError:
                raise ValueError(f"{yol}:{satir_no}: başlangıç/süre sayı değil") from None
            if sure < 0 or bas < 0:
                raise ValueError(f"{yol}:{satir_no}: negatif zaman veya süre")
            if sure == 0:
                continue
            sonuc.append((bas, round(bas + sure, 6), alanlar[7]))
    return sonuc


def _zaman(saniye: float) -> str:
    """saniye -> dd:ss.mmm (dakika:saniye.milisaniye)."""
    ms = _ms(saniye)
    dakika, kalan = divmod(ms, 60_000)
    return f"{dakika:02d}:{kalan // 1000:02d}.{kalan % 1000:03d}"


def tablo_yazdir(segmentler: list[tuple[float, float, str]]) -> str:
    """Segmentleri okunabilir metin tablosuna çevirir (KG-14).

    Girdi: segmentler. Çıktı: çok satırlı str; sütunlar Konuşmacı | Başlangıç |
    Bitiş | Süre, zamanlar dd:ss.mmm; altında konuşmacı başına toplam süre ve yüzde.
    """
    if not segmentler:
        return "Konuşma bulunamadı (segment yok)."
    basliklar = ("Konuşmacı", "Başlangıç", "Bitiş", "Süre")
    satirlar = [(e, _zaman(a), _zaman(b), _zaman(b - a)) for a, b, e in segmentler]
    gen = [max(len(basliklar[i]), *(len(s[i]) for s in satirlar)) for i in range(4)]

    def bicim(alanlar):
        return " | ".join(str(x).ljust(gen[i]) for i, x in enumerate(alanlar)).rstrip()

    cikti = [bicim(basliklar), "-+-".join("-" * g for g in gen)]
    cikti += [bicim(s) for s in satirlar]

    toplam: dict[str, float] = {}
    for a, b, e in segmentler:
        toplam[e] = toplam.get(e, 0.0) + (b - a)
    genel = sum(toplam.values())
    cikti.append("")
    cikti.append("Konuşmacı başına toplam:")
    for e in sorted(toplam, key=lambda k: (-toplam[k], k)):
        yuzde = 100 * toplam[e] / genel if genel > 0 else 0.0
        cikti.append(f"  {e}: {_zaman(toplam[e])} (%{yuzde:.1f})")
    return "\n".join(cikti)
