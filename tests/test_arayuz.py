"""Arayüz sözleşmesi kontrolü: modüller import edilir, fonksiyonlar var mı bakılır.

Çalıştırma: python tests/test_arayuz.py
"""
import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# modül -> {fonksiyon: parametre adları}  (docs/arayuz.md ile aynı olmalı)
SOZLESME = {
    "src.ses_girdi": {"ses_yukle": ["yol", "maks_sure_sn"], "on_isle": ["veri", "sr_girdi"], "wav_denetle": ["kaynak", "ad"]},
    "src.vad": {"konusma_bolgeleri": ["ses", "sr"]},
    "src.oznitelik": {
        "pencerele": ["bolgeler", "pencere_sn", "kaydirma_sn"],
        "vektorler": ["ses", "sr", "pencereler", "yontem"],
        "ecapa_modeli": ["onbellek"],
        "olcek_hesapla": ["X_listesi"],
        "olcekle": ["X", "olcek"],
    },
    "src.kumeleme": {"kumele": ["X", "konusmaci_sayisi", "esik"]},
    "src.cikti": {
        "segmentleri_birlestir": ["pencereler", "etiketler", "ada_esik_sn", "bosluk_sn"],
        "degisim_noktalari": ["segmentler"],
        "rttm_yaz": ["segmentler", "dosya_id", "yol"],
        "rttm_oku": ["yol"],
        "tablo_yazdir": ["segmentler"],
        "temel_araliklar": ["segmentler_ms"],
        "tek_konusmaci_segmentleri": ["segmentler_ms"],
    },
    "src.degerlendirme": {"degerlendir": ["ref_rttm", "hyp_rttm", "collar", "tolerans_sn"]},
    "src.main": {
        "main": ["argv"],
        "kisa_konusma": ["bolgeler"],
        "hat_segmentleri": ["bolgeler", "pencereler", "X", "konusmaci_sayisi", "esik"],
    },
    "src.bolme": {
        "grup": ["sayi"],
        "konusmaci_sayilari": ["rttm_dizin"],
        "bol": ["sayilar", "seed"],
        "liste_yaz": ["kayitlar", "yol"],
        "liste_oku": ["yol"],
        "main": ["argv"],
    },
    "src.toplu": {
        "SesKaynagi": ["zip_yol"],
        "onbellek_anahtarlari": [],
        "onbellek_yolu": ["onbellek", "tur", "kayit"],
        "ozellik_al": ["kaynak", "kayit", "onbellek", "yontemler"],
        "onbellek_doldur": ["zip_yol", "kayitlar", "onbellek", "yontemler", "gunluk_dizin", "log"],
        "deney_calistir": ["zip_yol", "rttm_dizin", "kayitlar", "deney", "cikti_dizin", "onbellek",
                           "esik", "model_yolu", "mfcc_olcek", "log"],
        "yuzdelik_izgarasi": ["yukseklikler", "n"],
        "esik_tara": ["zip_yol", "rttm_dizin", "kayitlar", "onbellek", "yontemler", "izgara",
                      "gunluk_dizin", "log"],
        "esikleri_yaz": ["yol", "sonuc"],
        "esikleri_oku": ["yol"],
        "model_egit": ["zip_yol", "rttm_dizin", "kayitlar", "onbellek", "model_yolu", "gunluk_dizin", "log"],
        "ozet_hesapla": ["satirlar"],
        "ozet_yaz": ["yol", "ozet"],
        "listeleri_oku": ["yollar"],
        "main": ["argv"],
    },
    "src.uyumluluk": {"speechbrain_hazirla": []},
    "src.gorsel": {
        "zaman_cizelgesi": ["ref_rttm", "hyp_rttm", "png_yolu", "baslik"],
        "sonuc_grafigi": ["csv_yolu", "png_yolu"],
    },
    "src.varsayilanlar": {"yukle": [], "esik": ["yontem"], "mfcc_olcek": []},
    "src.degisim_modeli": {
        "ozellikler": ["pencereler", "X_ecapa", "X_mfcc", "bolgeler", "k"],
        "cift_zamanlari": ["pencereler"],
        "baskin_konusmacilar": ["pencereler", "ref_segmentler", "oran"],
        "cift_etiketleri": ["baskin"],
        "tepeler": ["olasiliklar", "zamanlar", "esik", "min_aralik_sn"],
        "kayit_verisi": ["kayit", "pencereler", "X_ecapa", "X_mfcc", "bolgeler", "ref_segmentler", "k"],
        "egit": ["veriler", "seed", "n_kat", "log"],
        "kaydet": ["paket", "yol"],
        "yukle": ["yol"],
        "degisim_tahmin": ["paket", "pencereler", "X_ecapa", "X_mfcc", "bolgeler"],
        "bolgeleri_kes": ["bolgeler", "degisimler"],
        "e5_segmentleri": ["paket", "bolgeler", "pencereler", "X_ecapa", "X_mfcc", "esik"],
    },
}


def test_sozlesme():
    for modul, fonksiyonlar in SOZLESME.items():
        m = importlib.import_module(modul)
        for ad, parametreler in fonksiyonlar.items():
            assert hasattr(m, ad), f"{modul}.{ad} yok"
            gercek = list(inspect.signature(getattr(m, ad)).parameters)
            assert gercek == parametreler, f"{modul}.{ad} imzası uyuşmuyor: {gercek} != {parametreler}"
    from src import degisim_modeli, kumeleme, ses_girdi, toplu
    assert ses_girdi.SR == 16000
    assert isinstance(kumeleme.VARSAYILAN_ESIK, float)
    assert degisim_modeli.OZELLIK_SURUMU == "v2" and len(degisim_modeli.OZELLIKLER) == 6
    assert toplu.CSV_SUTUNLARI[:5] == ["deney", "kayit", "der", "jer", "f1"]
    assert sorted(toplu.DENEYLER) == ["E1", "E2", "E3", "E4", "E5"]


if __name__ == "__main__":
    test_sozlesme()
    print("test_arayuz: OK")
