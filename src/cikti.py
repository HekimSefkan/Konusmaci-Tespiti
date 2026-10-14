"""Segment birleştirme, değişim noktaları, RTTM okuma/yazma. Sahibi: Cevat. Sözleşme: docs/arayuz.md"""
import numpy as np

# segment: (baslangic_sn, bitis_sn, etiket) ; etiket "spk00", "spk01", ...


def segmentleri_birlestir(
    pencereler: list[tuple[float, float]],
    etiketler: np.ndarray,
) -> list[tuple[float, float, str]]:
    """Pencere etiketlerini konuşmacı segmentlerine çevirir.

    Girdi: pencereler [(bas_sn, bit_sn)] zamana göre sıralı;
           etiketler np.ndarray (N,), int (N == len(pencereler)).
    Çıktı: [(bas_sn, bit_sn, "spkNN")], zamana göre sıralı. Ardışık aynı etiketli
           pencereler birleşir; örtüşen farklı etiketli pencerelerde sınır
           örtüşmenin ortasıdır. Etiketler konuşma süresine göre azalan sırada
           spk00, spk01, ... diye yeniden adlandırılır.
    Hata: ValueError (uzunluklar eşit değil).
    """
    raise NotImplementedError


def degisim_noktalari(segmentler: list[tuple[float, float, str]]) -> list[float]:
    """Konuşmacı değişim noktalarını bulur.

    Girdi: segmentler (bkz. segmentleri_birlestir çıktısı).
    Çıktı: saniye cinsinden artan liste; ardışık iki segmentin etiketi
           farklıysa ikincinin başlangıcı. 0 veya 1 segmentte boş liste.
    """
    raise NotImplementedError


def rttm_yaz(
    segmentler: list[tuple[float, float, str]], dosya_id: str, yol: str
) -> None:
    """Segmentleri RTTM dosyasına yazar (10 alan, süre = bitiş - başlangıç, 3 ondalık).

    Girdi: segmentler; dosya_id (uzantısız ses dosyası adı); yol (çıktı .rttm).
    Çıktı: yok. Segment listesi boşsa boş dosya yazılır.
    Hata: OSError (yazılamadı).
    """
    raise NotImplementedError


def rttm_oku(yol: str) -> list[tuple[float, float, str]]:
    """RTTM dosyasını okur.

    Girdi: yol (.rttm).
    Çıktı: [(bas_sn, bit_sn, etiket)], dosyadaki sırayla (bit = bas + süre).
           Boş dosyada boş liste.
    Hata: FileNotFoundError (dosya yok), ValueError (satır biçimi bozuk).
    """
    raise NotImplementedError


def tablo_yazdir(segmentler: list[tuple[float, float, str]]) -> str:
    """Segmentleri okunabilir metin tablosuna çevirir (başlangıç, bitiş, süre, konuşmacı).

    Girdi: segmentler. Çıktı: çok satırlı str (süreler saniye, 2 ondalık).
    """
    raise NotImplementedError
