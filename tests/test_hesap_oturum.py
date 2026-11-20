"""Hesap sistemi, oturum doğrulaması, kullanıcıya özel geçmiş, adlandırma ve indirmeler (sahte hatla).

Model, ses dosyası ve ağ gerekmez; gradio gerektiren testler gradio yoksa atlanır.
"""
import csv
import inspect
import json
import os
import tempfile

import numpy as np
import pytest

from src import arayuz, gecmis, hesap

SEGMENTLER = [(0.0, 2.0, "spk00"), (2.0, 3.5, "spk01"), (4.0, 5.0, "spk00")]


class Saat:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def sahte_hat(girdi, yontem, k, esik):
    sahte_hat.cagri += 1
    return {"segmentler": list(SEGMENTLER), "ses_sn": 6.0, "ses": np.zeros(96000, dtype=np.float32), "sr": 16000,
            "bolgeler": [(0.0, 3.5), (4.0, 5.0)], "pencereler": [(0, 1.5), (2, 3.5), (4, 5)],
            "X": np.array([[1, 0], [0, 1], [1, 0.1]]), "esik": 0.79,
            "sureler": {"okuma": 0.1, "vad": 0.1, "oznitelik": 0.2, "kumeleme": 0.01}}


sahte_hat.cagri = 0


@pytest.fixture
def depo(tmp_path):
    return hesap.HesapDeposu(str(tmp_path / "Hafta7" / "kullanicilar.json"), saat=Saat())


@pytest.fixture
def uyg(tmp_path, depo):
    return arayuz.Uygulama(depo, hat=sahte_hat, gecmis_kok=str(tmp_path / "gecmis"))


def _ses(tmp_path, ad="kayit.wav"):
    yol = tmp_path / ad
    yol.write_bytes(b"x")  # sahte hat dosyayı okumaz
    return str(yol)


# --- Kayıt ve giriş ----------------------------------------------------------------------------

def test_kayit_giris_cikis(depo, tmp_path):
    k = depo.kayit("Ayşe", "Ayse@Ornek.com ", "gizli-sifre-1", "gizli-sifre-1")
    assert k["eposta"] == "ayse@ornek.com" and len(k["id"]) == 32
    anahtar = depo.giris("AYSE@ornek.com", "gizli-sifre-1")
    assert depo.kullanici(anahtar) == {"id": k["id"], "ad": "Ayşe", "eposta": "ayse@ornek.com"}
    depo.cikis(anahtar)
    assert depo.kullanici(anahtar) is None
    icerik = (tmp_path / "Hafta7" / "kullanicilar.json").read_text(encoding="utf-8")
    assert "gizli-sifre-1" not in icerik and '"tuz"' in icerik and '"ozet"' in icerik  # şifre düz yazılmaz
    assert sorted(os.listdir(tmp_path / "Hafta7")) == ["kullanicilar.json"]  # geçici ve kilit dosyası kalmaz


@pytest.mark.parametrize("ad,eposta,sifre,tekrar,mesaj", [
    ("", "a@b.co", "12345678", "12345678", "Ad"),
    ("A", "gecersiz", "12345678", "12345678", "e-posta"),
    ("A", "a@b.co", "1234567", "1234567", "en az 8"),
    ("A", "a@b.co", "12345678", "87654321", "aynı değil"),
])
def test_kayit_dogrulama(depo, ad, eposta, sifre, tekrar, mesaj):
    with pytest.raises(hesap.HesapHatasi, match=mesaj):
        depo.kayit(ad, eposta, sifre, tekrar)


def test_ayni_eposta_iki_kez_kaydolamaz(depo):
    depo.kayit("A", "a@b.co", "12345678", "12345678")
    with pytest.raises(hesap.HesapHatasi, match="kayıtlı bir hesap"):
        depo.kayit("B", " A@B.CO", "87654321", "87654321")


def test_hatali_giris_genel_mesaj(depo):
    depo.kayit("A", "a@b.co", "12345678", "12345678")
    with pytest.raises(hesap.HesapHatasi) as h1:
        depo.giris("a@b.co", "yanlis-sifre")
    with pytest.raises(hesap.HesapHatasi) as h2:
        depo.giris("yok@b.co", "12345678")
    assert str(h1.value) == str(h2.value) == hesap.GENEL_HATA  # e-postanın kayıtlı olup olmadığı belli olmaz


def test_bes_hatali_denemede_kilit(depo):
    depo.kayit("A", "a@b.co", "12345678", "12345678")
    for _ in range(hesap.MAKS_DENEME - 1):
        with pytest.raises(hesap.HesapHatasi, match="hatalı"):
            depo.giris("a@b.co", "yanlis")
    with pytest.raises(hesap.HesapHatasi, match="5 dakika"):
        depo.giris("a@b.co", "yanlis")
    with pytest.raises(hesap.HesapHatasi, match="5 dakika"):
        depo.giris("a@b.co", "12345678")  # kilitliyken doğru şifre de reddedilir
    depo._saat.t += hesap.KILIT_SN + 1
    assert depo.kullanici(depo.giris("a@b.co", "12345678"))


def test_oturum_suresi_dolar(depo):
    depo.kayit("A", "a@b.co", "12345678", "12345678")
    anahtar = depo.giris("a@b.co", "12345678")
    depo._saat.t += hesap.OTURUM_SN + 1
    assert depo.kullanici(anahtar) is None
    assert depo.kullanici(None) is None and depo.kullanici("uydurma") is None and depo.kullanici(123) is None


def test_bayat_kilit_temizlenir(depo, tmp_path):
    kilit = tmp_path / "Hafta7" / "kullanicilar.json.lock"
    kilit.parent.mkdir(parents=True, exist_ok=True)
    kilit.write_text("")
    os.utime(kilit, (0, 0))  # çok eski: çöken süreçten kalmış
    depo.kayit("A", "a@b.co", "12345678", "12345678")
    assert not kilit.exists()


def test_kayit_kapali(uyg, monkeypatch):
    monkeypatch.setenv("KT_KAYIT_ACIK", "0")
    assert not hesap.kayit_acik()
    with pytest.raises(hesap.HesapHatasi, match="kapalı"):
        uyg.kayit("A", "a@b.co", "12345678", "12345678")
    monkeypatch.delenv("KT_KAYIT_ACIK")
    assert hesap.kayit_acik()


# --- Oturumsuz işlem reddi ve kullanıcıya özel geçmiş ---------------------------------------------

@pytest.mark.parametrize("anahtar", [None, "", "uydurma-anahtar"])
def test_gecersiz_oturumla_islem_yapilmaz(uyg, tmp_path, anahtar):
    sahte_hat.cagri = 0
    veri = {"segmentler": [list(s) for s in SEGMENTLER]}
    for cagri in (lambda: uyg.analiz(anahtar, _ses(tmp_path), None, "ecapa", 0, 0, None),
                  lambda: uyg.gorunum(anahtar, veri, "Basit"),
                  lambda: uyg.segment(anahtar, veri, {"yol": "x"}, 0),
                  lambda: uyg.adlandir(anahtar, veri, [["spk00", "Ali"]]),
                  lambda: uyg.gecmis_listesi(anahtar),
                  lambda: uyg.gecmis_ac(anahtar, "x"),
                  lambda: uyg.gecmis_sil(anahtar, "x")):
        with pytest.raises(arayuz.OturumGecersiz):
            cagri()
    assert sahte_hat.cagri == 0 and not os.path.exists(tmp_path / "gecmis")


def test_gecmis_kullaniciya_ozel(uyg, tmp_path):
    a, ka = uyg.kayit("A", "a@b.co", "12345678", "12345678")
    b, kb = uyg.kayit("B", "b@b.co", "12345678", "12345678")
    veri, _, u = uyg.analiz(a, _ses(tmp_path), None, "ecapa", 0, 0, None)
    assert u == "" and os.path.isdir(tmp_path / "gecmis" / ka["id"] / veri["gecmis_adi"])
    assert len(uyg.gecmis_listesi(a)) == 1 and uyg.gecmis_listesi(b) == []
    veri_b, _, u = uyg.gecmis_ac(b, veri["gecmis_adi"])  # B, A'nın kaydını açamaz
    assert veri_b is None and "bulunamadı" in u
    tablo, _, notu = uyg.gecmis_sil(b, veri["gecmis_adi"])  # ... silemez
    assert tablo == [] and "Silinemedi" in notu and len(uyg.gecmis_listesi(a)) == 1
    acik, _, u = uyg.gecmis_ac(a, veri["gecmis_adi"])
    assert u == "" and acik["segmentler"] == veri["segmentler"]


# --- Adlandırma ve indirmeler --------------------------------------------------------------------

def test_adlari_ayikla():
    veri = {"segmentler": [list(s) for s in SEGMENTLER]}
    tablo = [["spk00", "  Ahmet   Kaya "], ["spk01", "Konuşmacı 2"], ["yok", "X"]]
    assert arayuz.adlari_ayikla(veri, tablo) == {"spk00": "Ahmet Kaya"}
    assert arayuz.adlari_ayikla(veri, [["spk00", "Ahmet (spk00)"], ["spk01", "spk01"]]) == {"spk00": "Ahmet"}
    assert arayuz.adlari_ayikla(veri, [["spk00", "x" * 99]])["spk00"] == "x" * arayuz.MAKS_AD


def test_adlandirma_her_yere_yansir_ve_gecmiste_saklanir(uyg, tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("pandas")
    a, ka = uyg.kayit("A", "a@b.co", "12345678", "12345678")
    veri, _, _ = uyg.analiz(a, _ses(tmp_path), None, "ecapa", 0, 0, None)
    yeni = uyg.adlandir(a, veri, [["spk00", "Ahmet Kaya"], ["spk01", "Konuşmacı 2"]])
    assert yeni["ozel_adlar"] == {"spk00": "Ahmet Kaya"} and veri.get("ozel_adlar") == {}  # kopya güncellenir
    b = arayuz.gorunum_verisi(yeni, "Basit")
    assert "Ahmet Kaya" in b["ozet"] and "Ahmet Kaya" in set(b["cizelge"].layout.yaxis.categoryarray)
    assert list(b["segmentler"].data["Konuşmacı"]) == ["Ahmet Kaya", "Konuşmacı 2", "Ahmet Kaya"]
    g = arayuz.gorunum_verisi(yeni, "Gelişmiş")
    assert list(g["sure_tablosu"].data["Ad (düzenlenebilir)"]) == ["Ahmet Kaya (spk00)", "spk01"]
    kayit = gecmis.ac(os.path.join(uyg.gecmis_kok, ka["id"]), veri["gecmis_adi"])
    assert kayit["ozel_adlar"] == {"spk00": "Ahmet Kaya"} and kayit["rttm_yolu"]  # dosya adları korunur
    rttm, csv_yolu, json_yolu = arayuz.indirme_dosyalari(yeni, str(tmp_path / "indir"))
    assert "SPEAKER kayit 1 0.000 2.000 <NA> <NA> Ahmet_Kaya" in open(rttm, encoding="utf-8").read()
    with open(csv_yolu, encoding="utf-8-sig", newline="") as f:
        satirlar = list(csv.reader(f))
    assert satirlar[0] == ["sira", "etiket", "konusmaci", "baslangic_sn", "bitis_sn", "sure_sn"]
    assert satirlar[1] == ["1", "spk00", "Ahmet Kaya", "0.000", "2.000", "2.000"] and satirlar[2][2] == "Konuşmacı 2"
    with open(json_yolu, encoding="utf-8") as f:
        j = json.load(f)
    assert j["konusmaci_adlari"] == {"spk00": "Ahmet Kaya", "spk01": "Konuşmacı 2"}
    assert "ses_yolu" not in j and "oynat_yolu" not in j and j["segmentler"] == veri["segmentler"]


def test_ayni_ad_tekil_kalir():
    adlar = arayuz.konusmaci_adlari(SEGMENTLER, True, {"spk00": "Ali", "spk01": "Ali"})
    assert adlar == {"spk00": "Ali", "spk01": "Ali [spk01]"}


def test_segment_bilgisi_ve_vurgu():
    pytest.importorskip("plotly")
    veri = {"segmentler": [list(s) for s in SEGMENTLER], "ozel_adlar": {}}
    assert arayuz.segment_bilgisi(veri, 1, "Basit") == "**Konuşmacı 2**  \n00:02.000 – 00:03.500 · 1,50 sn  \nSegment 2 / 3"
    sekil = arayuz.cizelge_olustur(veri, "Basit", vurgu=(2.0, 3.5))
    vurgu = [s for s in sekil.layout.shapes if s.type == "rect"]
    assert len(vurgu) == 1 and (vurgu[0].x0, vurgu[0].x1) == (2.0, 3.5)


def test_oynatilabilir_kopya(tmp_path, monkeypatch):
    dis = tmp_path / "drive"
    dis.mkdir()
    (dis / "k.wav").write_bytes(b"abc")
    gecici = tmp_path / "gecici"
    gecici.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(gecici))
    monkeypatch.setattr(tempfile, "tempdir", str(gecici))
    kopya = arayuz.oynatilabilir_kopya(str(dis / "k.wav"))  # geçici klasör dışında: kopyalanır
    assert kopya != str(dis / "k.wav") and kopya.startswith(str(gecici)) and open(kopya, "rb").read() == b"abc"
    assert arayuz.oynatilabilir_kopya(kopya) == kopya  # zaten geçici klasörde
    assert arayuz.oynatilabilir_kopya(None) is None and arayuz.oynatilabilir_kopya(str(dis / "yok.wav")) is None


# --- Gradio işleyicileri: oturum doğrulaması, API kapalı, görünüm geçişi -------------------------

def _acik(guncelleme) -> bool:
    """Arayüz gösterme/gizlemeyi "kt-gizli" CSS sınıfıyla yapar (gradio görünürlüğü kullanılmaz)."""
    return arayuz.GIZLI_SINIF not in (guncelleme.elem_classes or [])


def _demo(uyg):
    pytest.importorskip("gradio")
    return arayuz.arayuz_olustur(uygulama=uyg)


def _fn(demo, ad):
    return next(f.fn for f in demo.fns.values() if f.name == ad)


def test_api_kapali(uyg):
    demo = _demo(uyg)
    assert demo.fns and all(f.api_visibility == "private" for f in demo.fns.values())


KORUMALI = ["calistir", "mod_degisti", "segment_sec", "segment_onceki", "segment_sonraki", "adlandir",
            "indirme_dogrula", "gecmis_yenile", "gecmis_sec", "gecmis_ac", "gecmis_sil"]


@pytest.mark.parametrize("ad", KORUMALI)
def test_islem_gecersiz_oturumda_giris_ekranina_doner(uyg, ad):
    demo = _demo(uyg)
    sahte_hat.cagri = 0
    fn = _fn(demo, ad)
    zorunlu = [p for p in inspect.signature(fn).parameters.values() if p.default is inspect.Parameter.empty]
    cikti = fn("gecersiz", *([None] * (len(zorunlu) - 1)))
    giris, uygulama_ekrani, notu, kullanici, oturum = cikti[:5]
    assert _acik(giris) and not _acik(uygulama_ekrani) and "Oturum geçersiz" in notu
    assert oturum is None and sahte_hat.cagri == 0


def test_giris_kayit_cikis_isleyicileri(uyg):
    demo = _demo(uyg)
    c = _fn(demo, "kayit_ol")("Ayşe", "ayse@ornek.com", "12345678", "12345678")
    assert _acik(c[1]) and not _acik(c[0]) and "Ayşe" in c[3] and uyg.depo.kullanici(c[4])
    hatali = _fn(demo, "giris_yap")("ayse@ornek.com", "yanlis-sifre")
    assert "hatalı" in hatali[2]
    g = _fn(demo, "giris_yap")("ayse@ornek.com", "12345678")
    anahtar = g[4]
    assert _acik(g[1]) and _acik(_fn(demo, "acilis")(anahtar)[1])  # sayfa yenilenince açık kalır
    cikis = _fn(demo, "cikis_yap")(anahtar)
    assert _acik(cikis[0]) and not _acik(cikis[1]) and cikis[4] is None and uyg.depo.kullanici(anahtar) is None
    assert not isinstance(getattr(_fn(demo, "acilis")(anahtar)[1], "elem_classes", None), list)  # açılmaz (skip)


def test_gorunum_gecisi_yeniden_analiz_yapmaz(uyg, tmp_path):
    pytest.importorskip("plotly")
    demo = _demo(uyg)
    anahtar, _ = uyg.kayit("A", "a@b.co", "12345678", "12345678")
    sahte_hat.cagri = 0
    c = _fn(demo, "calistir")(anahtar, _ses(tmp_path, "ikgcq_028.0-118.0.wav"), None, "ecapa", 0, 0, None, "Basit")
    veri = c[6]
    assert sahte_hat.cagri == 1 and veri and _acik(c[8])  # sonuç alanı
    assert _acik(c[10]) and ".rttm" in str(c[11].value)  # indirme satırı ve RTTM
    ilk_gelismis = c[5 + 3 + 6 + 13]  # ekran + uyarı/veri/ses + sabit + icerik[13]
    assert not _acik(ilk_gelismis)
    m = _fn(demo, "mod_degisti")(anahtar, veri, "Gelişmiş")
    assert _acik(m[5 + 13]) and sahte_hat.cagri == 1  # gelişmiş bölüm ilk geçişte görünür, analiz yok
    assert [a.text for a in m[5 + 2].layout.annotations] == ["Referans", "Sistem", "VAD"]
    m = _fn(demo, "mod_degisti")(anahtar, veri, "Basit")
    assert not _acik(m[5 + 13]) and sahte_hat.cagri == 1
