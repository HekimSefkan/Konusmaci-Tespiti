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
        "segmentleri_hesapla": ["girdi", "yontem", "konusmaci_sayisi", "esik", "maks_sure_sn"],
        "hat_ayrintili": ["girdi", "yontem", "konusmaci_sayisi", "esik", "maks_sure_sn"],
    },
    "src.gorsel_etkilesimli": {
        "konusmaci_esle": ["ref", "hyp"],
        "renkler": ["hyp_etiketler", "ref_etiketler", "esleme"],
        "hata_bolgeleri": ["ref", "hyp", "esleme"],
        "zaman_cizelgesi": ["seritler", "degisimler", "baslik", "vurgu", "satir_px", "lejant"],
        "hata_cizelgesi": ["bolgeler", "sure_sn"],
        "benzerlik_haritasi": ["etiketler", "matris", "adlar"],
    },
    "src.gecmis": {
        "gecmis_dizini": [],
        "kaydet": ["kok", "veri", "ses_yolu", "rttm_yolu", "zaman"],
        "listele": ["kok"],
        "ac": ["kok", "ad"],
        "sil": ["kok", "ad"],
        "guncelle": ["kok", "ad", "veri"],
    },
    "src.test_seti": {
        "zip_kayitlari": ["zip_yol"],
        "rttm_kayitlari": ["rttm_dizin"],
        "eslestir": ["ses_adlari", "rttm_adlari"],
        "test_listesi_yaz": ["zip_yol", "rttm_dizin", "liste_yolu"],
        "karsilastirma": ["dev_ozet", "test_ozet", "deneyler", "dev_ek"],
        "grup_tablosu": ["satirlar", "sayilar"],
        "en_kotu": ["satirlar", "n", "sayilar", "rttm_klasor"],
        "tablo_metni": ["satirlar", "sutunlar", "basamak"],
        "cihaz_bilgisi": [],
        "cihaz_etiketi": ["bilgi"],
        "hiz_olc": ["ses_yolu", "yontem", "tekrar", "isinma", "hat"],
        "hiz_yaz": ["sonuclar", "klasor", "bilgi"],
        "hiz_tablosu": ["klasor"],
        "kisa_ozet": ["sonuc_dizin", "eslesme", "hata_satiri", "maks_satir"],
    },
    "src.hesap": {
        "kayit_acik": [],
        "db_yolu": [],
        "HesapDeposu": ["yol", "oturum_sn", "saat"],
    },
    "src.arayuz": {
        "girdileri_denetle": ["konusmaci_sayisi", "esik"],
        "analiz": ["ses_yolu", "yontem", "konusmaci_sayisi", "esik", "ref_rttm", "cikti_dizin", "hat", "cizelge"],
        "izinli_dizin": [],
        "bilgi_html": ["mesaj"],
        "konusmaci_adlari": ["segmentler", "basit", "ozel"],
        "segment_bilgisi": ["veri", "satir", "mod"],
        "sure_tablosu_df": ["segmentler", "renk", "adlar"],
        "adlari_ayikla": ["veri", "tablo"],
        "cizelge_olustur": ["veri", "mod", "vurgu"],
        "indirme_dosyalari": ["veri", "klasor"],
        "oynatilabilir_kopya": ["yol"],
        "Uygulama": ["depo", "hat", "gecmis_kok"],
        "varsayilan_uygulama": ["hat"],
        "konusmaci_benzerligi": ["pencereler", "X", "segmentler"],
        "veri_olustur": ["s", "ref_notu", "ayarlar", "dosya_adi"],
        "segment_sesi": ["ses", "sr", "bas", "bit"],
        "segment_secimi": ["veri", "ses_bilgi", "satir", "yukle"],
        "ozet_cumlesi": ["segmentler", "adlar"],
        "segment_tablosu_df": ["segmentler", "renk", "adlar"],
        "bilesen_html": ["metrikler"],
        "eslesme_html": ["referans", "segmentler", "tolerans_sn"],
        "asama_html": ["sureler", "islem_sn"],
        "gorunum_verisi": ["veri", "mod"],
        "analiz_et": ["ses_yolu", "yontem", "konusmaci_sayisi", "esik", "ref_rttm", "sunucu_yolu", "hat",
                      "gecmis_kok"],
        "gecmis_tablosu": ["kok"],
        "gecmis_acilis": ["kok", "ad"],
        "gecmis_silme": ["kok", "ad"],
        "segment_tablosu": ["segmentler"],
        "konusmaci_toplamlari": ["segmentler"],
        "degisim_tablosu": ["segmentler"],
        "metrik_metni": ["metrikler"],
        "hata_mesaji": ["dosya", "hata"],
        "ses_yolu_sec": ["yuklenen", "sunucu_yolu"],
        "otomatik_referans": ["ses_yolu"],
        "arayuz_cikti": ["ses_yolu", "yontem", "konusmaci_sayisi", "esik", "ref_rttm", "sunucu_yolu", "hat"],
        "uyari_html": ["mesaj"],
        "kartlar_html": ["s", "ref_notu"],
        "segment_html": ["segmentler"],
        "sure_html": ["segmentler", "renk", "adlar"],
        "degisim_ciftleri": ["segmentler"],
        "degisim_html": ["segmentler", "renk", "adlar"],
        "dogruluk_metni": ["s", "ref_notu"],
        "gorunum_cikti": ["ses_yolu", "yontem", "konusmaci_sayisi", "esik", "ref_rttm", "sunucu_yolu", "hat"],
        "tema": [],
        "arayuz_olustur": ["hat", "uygulama"],
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
    "src.vad_olcum": {
        "cercevelerden_bolgeler": ["kararlar", "cerceve_sn"],
        "yumusat": ["bolgeler", "sure_sn", "min_konusma_ms", "min_sessizlik_ms", "kenar_payi_ms"],
        "maske": ["bolgeler", "sure_sn", "cozunurluk_sn"],
        "hata_olc": ["ref_maske", "hyp_maske", "cozunurluk_sn"],
        "ozet_hesapla": ["satirlar"],
        "webrtc_bolgeleri": ["ses", "sr", "agresiflik"],
        "olc": ["zip_yol", "rttm_dizin", "kayitlar", "cikti_dizin", "limit", "dinleme_dizin", "dinleme_sayisi",
                "parca_sn", "log"],
        "snr_tahmini": ["ses", "sr", "bolgeler", "min_konusmasiz_sn"],
        "dinleme_parcasi": ["bolgeler", "sure_sn", "uzunluk_sn"],
        "dinleme_yaz": ["zip_yol", "satirlar", "dinleme_dizin", "cikti_dizin", "sayi", "parca_sn", "log"],
        "main": ["argv"],
    },
    "src.gorsel": {
        "zaman_cizelgesi": ["ref_rttm", "hyp_rttm", "png_yolu", "baslik"],
        "zaman_seritleri": ["seritler", "png_yolu", "baslik"],
        "renk_haritasi": ["etiketler"],
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
