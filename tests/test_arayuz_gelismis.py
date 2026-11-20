"""Web arayüzünün güvenlik, geçmiş, ses dinleme, renk eşleme ve basit/gelişmiş görünüm mantığı.

Sahte hatla çalışır: model, ses dosyası ve gradio gerekmez (plotly/pandas/scipy gerekenler atlanabilir).
"""
import json
import os

import numpy as np
import pytest

from src import arayuz, cikti, degerlendirme, gecmis, gorsel, gorsel_etkilesimli, main as ana

SEGMENTLER = [(0.0, 2.0, "spk00"), (2.0, 3.5, "spk01"), (4.0, 5.0, "spk00")]


def sahte_ayrintili(girdi, yontem, k, esik):
    """main.hat_ayrintili biçiminde sahte çıktı (bellekte; dosya yazılmaz)."""
    pencereler = [(0.0, 1.5), (0.5, 2.0), (2.0, 3.5), (4.0, 5.0)]
    X = np.array([[1, 0], [1, 0.1], [0, 1], [1, 0.05]], dtype=np.float32)
    return {"segmentler": list(SEGMENTLER), "ses_sn": 6.0, "ses": np.linspace(-0.5, 0.5, 6 * 16000, dtype=np.float32),
            "sr": 16000, "bolgeler": [(0.0, 3.5), (4.0, 5.0)], "pencereler": pencereler, "X": X, "esik": 0.79,
            "sureler": {"okuma": 0.1, "vad": 0.2, "oznitelik": 0.3, "kumeleme": 0.01}}


def _dosya(klasor, ad, icerik=b"x"):
    yol = klasor / ad
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_bytes(icerik)
    return str(yol)


# --- 1. Güvenlik: sunucu yolu yalnızca KT_IZINLI_DIZIN altında ---------------------------------

def test_sunucu_yolu_degisken_yoksa_kapali(tmp_path, monkeypatch):
    monkeypatch.delenv("KT_IZINLI_DIZIN", raising=False)
    with pytest.raises(ValueError, match="kapalı"):
        arayuz.ses_yolu_sec(None, _dosya(tmp_path, "a.wav"))
    assert arayuz.izinli_yollar() == [] or "KT_GECMIS_DIZIN" in os.environ


def test_sunucu_yolu_izinli_klasor_ici(tmp_path, monkeypatch):
    izinli = tmp_path / "izinli"
    yol = _dosya(izinli, "alt/kayit.wav")
    monkeypatch.setenv("KT_IZINLI_DIZIN", str(izinli))
    assert arayuz.ses_yolu_sec(None, yol) == os.path.realpath(yol)
    assert arayuz.ses_yolu_sec(None, "alt/kayit.wav") == os.path.realpath(yol)  # göreli yol
    with pytest.raises(FileNotFoundError, match="alt/yok.wav"):
        arayuz.ses_yolu_sec(None, "alt/yok.wav")


@pytest.mark.parametrize("kacis", ["../gizli.wav", "alt/../../gizli.wav"])
def test_sunucu_yolu_kacis_engellenir(tmp_path, monkeypatch, kacis):
    izinli = tmp_path / "izinli"
    (izinli / "alt").mkdir(parents=True)
    _dosya(tmp_path, "gizli.wav")
    monkeypatch.setenv("KT_IZINLI_DIZIN", str(izinli))
    with pytest.raises(ValueError, match="dışında"):
        arayuz.ses_yolu_sec(None, kacis)


def test_disaridaki_yol_mesajda_gorunmez(tmp_path, monkeypatch):
    izinli = tmp_path / "izinli"
    izinli.mkdir()
    disari = _dosya(tmp_path, "baska_klasor/gizli_dosya.wav")
    monkeypatch.setenv("KT_IZINLI_DIZIN", str(izinli))
    durum, *_ = arayuz.arayuz_cikti(None, "ecapa", None, None, None, sunucu_yolu=disari, hat=sahte_ayrintili)
    assert "dışında" in durum and "gizli_dosya" not in durum and "baska_klasor" not in durum
    _, _, u = arayuz.analiz_et(None, "ecapa", 0, 0, None, sunucu_yolu=disari, hat=sahte_ayrintili)
    assert "kt-hata" in u and "gizli_dosya" not in u


def test_sembolik_baglantiyla_kacis_engellenir(tmp_path, monkeypatch):
    izinli = tmp_path / "izinli"
    izinli.mkdir()
    hedef = _dosya(tmp_path, "gizli.wav")
    bag = izinli / "bag.wav"
    try:
        os.symlink(hedef, bag)
    except (OSError, NotImplementedError):
        pytest.skip("sembolik bağlantı oluşturulamıyor (izin yok)")
    monkeypatch.setenv("KT_IZINLI_DIZIN", str(izinli))
    with pytest.raises(ValueError, match="dışında"):
        arayuz.ses_yolu_sec(None, "bag.wav")


def test_app_kimlik(monkeypatch):
    import app

    monkeypatch.delenv("KT_KULLANICI", raising=False)
    monkeypatch.delenv("KT_SIFRE", raising=False)
    assert app.kimlik() is None
    monkeypatch.setenv("KT_KULLANICI", "grup")
    assert app.kimlik() is None  # şifre yoksa koruma yok
    monkeypatch.setenv("KT_SIFRE", "s3")
    assert app.kimlik() == ("grup", "s3")


# --- 2. Geçmiş -----------------------------------------------------------------------------------

def test_gecmis_kaydet_listele_ac_sil(tmp_path, monkeypatch):
    kok = str(tmp_path / "gecmis")
    ses = _dosya(tmp_path, "ikgcq_028.0-118.0.wav", b"RIFFdeneme")
    veri, ses_bilgi, u = arayuz.analiz_et(ses, "ecapa", 0, 0, None, hat=sahte_ayrintili, gecmis_kok=kok)
    assert u == "" and veri["gecmis_adi"] and ses_bilgi["dizi"] is not None
    ad = veri["gecmis_adi"]
    klasor = os.path.join(kok, ad)
    with open(os.path.join(klasor, "sonuc.json"), encoding="utf-8") as f:
        kayit = json.load(f)
    assert kayit["segmentler"][1] == [2.0, 3.5, "spk01"] and kayit["ayarlar"]["yontem"] == "ecapa"
    assert "ses_yolu" not in kayit and "rttm_yolu" not in kayit  # yerel yollar yazılmaz
    assert os.path.isfile(os.path.join(klasor, kayit["rttm"])) and os.path.isfile(os.path.join(klasor, kayit["ses"]))
    satirlar = arayuz.gecmis_tablosu(kok)
    assert len(satirlar) == 1 and satirlar[0][1] == "ikgcq_028.0-118.0.wav" and satirlar[0][2] == 2
    acik, sb, u = arayuz.gecmis_acilis(kok, ad)
    assert u == "" and acik["segmentler"] == kayit["segmentler"] and os.path.isfile(acik["ses_yolu"])
    assert sb == {"yol": acik["ses_yolu"], "dizi": None, "sr": None}
    tablo, secili, notu = arayuz.gecmis_silme(kok, ad)
    assert tablo == [] and secili is None and "Silindi" in notu and not os.path.exists(klasor)


def test_gecmis_buyuk_ses_kopyalanmaz(tmp_path, monkeypatch):
    monkeypatch.setattr(gecmis, "MAKS_SES_BAYT", 5)
    ad = gecmis.kaydet(str(tmp_path / "g"), {"dosya_id": "uzun", "segmentler": []},
                       _dosya(tmp_path, "uzun.wav", b"0123456789"))
    veri = gecmis.ac(str(tmp_path / "g"), ad)
    assert veri["ses_yolu"] is None and "kopyalanmadı" in veri["ses_notu"]


@pytest.mark.parametrize("ad", ["", "..", "../x", "yok", "a/b"])
def test_gecmis_gecersiz_ad(tmp_path, ad):
    (tmp_path / "g").mkdir()
    with pytest.raises(ValueError):
        gecmis.ac(str(tmp_path / "g"), ad)
    with pytest.raises(ValueError):
        gecmis.sil(str(tmp_path / "g"), ad)


def test_gecmis_ayni_saniyede_iki_kayit(tmp_path):
    from datetime import datetime

    z = datetime(2026, 11, 20, 12, 0, 0)
    a = gecmis.kaydet(str(tmp_path), {"dosya_id": "k b", "segmentler": []}, zaman=z)
    b = gecmis.kaydet(str(tmp_path), {"dosya_id": "k b", "segmentler": []}, zaman=z)
    assert a == "20261120-120000_k_b" and b == "20261120-120000_k_b-2"
    assert [k["ad"] for k in gecmis.listele(str(tmp_path))] == [b, a]


def test_gecmis_kapaliysa(monkeypatch):
    monkeypatch.delenv("KT_GECMIS_DIZIN", raising=False)
    assert gecmis.gecmis_dizini() is None and arayuz.gecmis_tablosu(None) == []
    assert "seçin" in arayuz.gecmis_acilis(None, None)[2]


# --- 3. Ses dinleme ------------------------------------------------------------------------------

def test_segment_sesi_numpy_ile_kesilir():
    ses = np.linspace(-1, 1, 16000 * 4, dtype=np.float32)
    sr, parca = arayuz.segment_sesi(ses, 16000, 1.0, 2.5)
    assert sr == 16000 and parca.dtype == np.int16 and len(parca) == 24000
    assert parca[0] == int(ses[16000] * 32767)
    assert len(arayuz.segment_sesi(ses, 16000, 3.5, 9.0)[1]) == 8000  # sona kırpılır
    assert len(arayuz.segment_sesi(ses, 16000, 5.0, 6.0)[1]) == 0


def test_segment_secimi_sesi_bir_kez_yukler():
    veri = {"segmentler": [list(s) for s in SEGMENTLER]}
    cagri = []

    def yukle(yol):
        cagri.append(yol)
        return np.zeros(16000 * 6, dtype=np.float32), 16000

    oyna, sb, notu = arayuz.segment_secimi(veri, {"yol": "k.wav"}, 1, yukle)
    assert oyna[0] == 16000 and len(oyna[1]) == 24000 and "spk01" in notu and "00:02.000" in notu
    oyna, sb, _ = arayuz.segment_secimi(veri, sb, 0, yukle)
    assert len(oyna[1]) == 32000 and cagri == ["k.wav"]  # ikinci seçimde yeniden okunmaz
    assert arayuz.segment_secimi(veri, {"yol": None}, 0)[2].startswith("Bu analizin sesi yok")
    assert arayuz.segment_secimi(veri, sb, 99)[0] is None


def test_segment_secimi_okuma_hatasi_mesaj():
    def yukle(yol):
        raise ValueError(f"{os.path.basename(yol)}: bozuk WAV (deneme)")

    oyna, _, notu = arayuz.segment_secimi({"segmentler": [[0, 1, "spk00"]]}, {"yol": "/x/k.wav"}, 0, yukle)
    assert oyna is None and "k.wav" in notu and "bozuk" in notu


# --- 4. Renk eşleme ve hata bölgeleri ----------------------------------------------------------

def test_renkler_referanssiz_cizelgeyle_ayni():
    hyp_renk, ref_renk = gorsel_etkilesimli.renkler(["spk01", "spk00"])
    assert hyp_renk == gorsel.renk_haritasi(["spk00", "spk01"]) and ref_renk == {}


def test_macar_eslemesi_ve_renkler():
    pytest.importorskip("scipy")
    ref = [(0, 2, "A"), (2, 3.5, "B"), (4, 5, "A")]
    hyp = [(0, 2, "spk01"), (2, 3.5, "spk00"), (4, 5, "spk01"), (5, 6, "spk02")]
    esleme = gorsel_etkilesimli.konusmaci_esle(ref, hyp)
    assert esleme == {"spk01": "A", "spk00": "B"}
    hyp_renk, ref_renk = gorsel_etkilesimli.renkler([e for *_, e in hyp], ["A", "B"], esleme)
    assert hyp_renk["spk01"] == ref_renk["A"] and hyp_renk["spk00"] == ref_renk["B"]
    assert hyp_renk["spk02"] in gorsel_etkilesimli.GRILER


def test_hata_bolgeleri():
    ref = [(0, 2, "A"), (2, 4, "B")]
    hyp = [(0, 1, "x"), (1, 2, "y"), (2, 4, "y"), (4, 5, "x")]
    bolgeler = gorsel_etkilesimli.hata_bolgeleri(ref, hyp, {"x": "A", "y": "B"})
    assert bolgeler == [(1.0, 2.0, "karisiklik"), (4.0, 5.0, "yanlis_alarm")]
    assert gorsel_etkilesimli.hata_bolgeleri([(0, 3, "A")], [(1, 2, "x")], {"x": "A"}) == [
        (0.0, 1.0, "kacirilan"), (2.0, 3.0, "kacirilan")]


def test_eslesen_ciftler_sayilarla_tutarli():
    ref, hyp = [1.0, 5.0, 9.0], [1.3, 4.0, 9.4, 12.0]
    assert degerlendirme._eslesen_ciftler(ref, hyp, 0.5) == [(0, 0), (2, 2)]
    assert degerlendirme._degisim_eslestir(ref, hyp, 0.5) == (2, 2, 1)


def test_benzerlik_matrisi():
    b = arayuz.konusmaci_benzerligi([(0, 1), (1, 2), (2, 3)], np.array([[1, 0], [1, 0], [0, 1]]),
                                    [(0, 2, "spk00"), (2, 3, "spk01")])
    assert b["etiketler"] == ["spk00", "spk01"] and b["matris"] == [[1.0, 0.0], [0.0, 1.0]]
    assert arayuz.konusmaci_benzerligi([(0, 1)], np.array([[1, 0]]), [(0, 1, "spk00")]) is None
    assert arayuz.konusmaci_benzerligi([], None, SEGMENTLER) is None


# --- 5. Basit / gelişmiş görünüm -------------------------------------------------------------

def test_ozet_cumlesi_ve_adlar():
    adlar = arayuz.konusmaci_adlari(SEGMENTLER, basit=True)
    assert adlar == {"spk00": "Konuşmacı 1", "spk01": "Konuşmacı 2"}
    assert arayuz.konusmaci_adlari(SEGMENTLER, basit=False) == {"spk00": "spk00", "spk01": "spk01"}
    c = arayuz.ozet_cumlesi(SEGMENTLER, adlar)
    assert c == "Kayıtta 2 kişi konuşuyor. En çok Konuşmacı 1 konuşuyor (%66,7 pay, 3,0 sn). Konuşmacı 2 kez değişiyor."
    assert "tek kişi" in arayuz.ozet_cumlesi([(0, 2, "spk00")]) and "bulunamadı" in arayuz.ozet_cumlesi([])


def test_basit_ve_gelismis_gorunum(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    pytest.importorskip("pyannote.metrics")
    ses = _dosya(tmp_path, "ikgcq_028.0-118.0.wav")  # data/sample'da aynı adlı RTTM: otomatik referans
    veri, _, u = arayuz.analiz_et(ses, "ecapa", 0, 0, None, hat=sahte_ayrintili)
    assert u == "" and veri["metrikler"] and veri["vad"] and veri["sureler"] and veri["benzerlik"]
    json.dumps(veri)  # geçmişe yazılabilir

    b = arayuz.gorunum_verisi(veri, "Basit")
    assert not b["gelismis"] and b["referans_var"] and "Konuşmacı 1" in b["ozet"]
    assert b["degisimler"] == b["dogruluk"] == "" and b["hata_cizelgesi"] is None and b["benzerlik"] is None
    satirlar = set(b["cizelge"].layout.yaxis.categoryarray)
    assert satirlar == {"Konuşmacı 1", "Konuşmacı 2"}  # tek (sistem) şerit
    assert list(b["segmentler"].data["Konuşmacı"]) == ["Konuşmacı 1", "Konuşmacı 2", "Konuşmacı 1"]
    assert len(b["cizelge"].layout.shapes) == 2  # değişim noktaları kesikli çizgi

    g = arayuz.gorunum_verisi(veri, "Gelişmiş")
    assert g["gelismis"] and "spk00" in g["sureler"] and "Konuşmacı 1" not in g["sureler"]
    satirlar = list(g["cizelge"].layout.yaxis.categoryarray)
    assert any(s.startswith("Referans ·") for s in satirlar) and any(s.startswith("VAD ·") for s in satirlar)
    assert "DER" in g["bilesenler"] and "doğru eşleşen" in g["eslesmeler"] and "Kaçırılan" in g["eslesmeler"]
    assert g["hata_cizelgesi"] is not None
    assert "Kümeleme" in g["asamalar"] and g["benzerlik"] is not None and "SPEAKER" in g["rttm_metni"]
    # tablo ve çizelge renkleri aynı (referans eşlemesiyle)
    ilk_iz = g["cizelge"].data[0]
    assert ilk_iz.marker.color in g["renk"].values() or ilk_iz.marker.color in gorsel.TAB10


def test_gorunum_referanssiz_dogruluk_yok(tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("pandas")
    veri, _, _ = arayuz.analiz_et(_dosya(tmp_path, "kayit.wav"), "ecapa", 0, 0, None, hat=sahte_ayrintili)
    g = arayuz.gorunum_verisi(veri, "Gelişmiş")
    assert not g["referans_var"] and g["dogruluk"] == "" and g["hata_cizelgesi"] is None
    assert g["renk"] == gorsel.renk_haritasi(["spk00", "spk01"])


def test_hat_ayrintili_anahtarlar(monkeypatch):
    monkeypatch.setattr(ana.ses_girdi, "ses_yukle", lambda yol, maks: (np.zeros(16000, dtype=np.float32), 16000))
    monkeypatch.setattr(ana.vad, "konusma_bolgeleri", lambda ses, sr: [])
    r = ana.hat_ayrintili("sessiz.wav")
    assert r["segmentler"] == [] and r["ses_sn"] == 1.0 and r["X"] is None and r["bolgeler"] == []
    assert set(r["sureler"]) == {"okuma", "vad", "oznitelik", "kumeleme"} and r["esik"] == pytest.approx(0.79346)
    assert ana.segmentleri_hesapla("sessiz.wav") == ([], 1.0)


def test_arayuz_kurulur_ortamla(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setenv("KT_IZINLI_DIZIN", str(tmp_path))
    monkeypatch.setenv("KT_GECMIS_DIZIN", str(tmp_path / "gecmis"))
    assert arayuz.izinli_yollar() == [os.path.realpath(tmp_path), os.path.realpath(tmp_path / "gecmis")]
    demo = arayuz.arayuz_olustur(hat=sahte_ayrintili)
    assert demo.title == arayuz.SAYFA_BASLIGI
