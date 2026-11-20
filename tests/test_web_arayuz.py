"""src/arayuz.py: web arayüzü mantığı sahte hatla (model ve ses dosyası olmadan) test edilir."""
import os

import pytest

from src import arayuz, cikti, ses_girdi

SEGMENTLER = [(0.0, 2.0, "spk00"), (2.0, 3.5, "spk01"), (4.0, 5.0, "spk00")]


def sahte_hat(girdi, yontem, konusmaci_sayisi, esik):
    sahte_hat.cagri = (os.path.basename(girdi), yontem, konusmaci_sayisi, esik)
    return list(SEGMENTLER), 6.0


def _dosya(tmp_path, ad, icerik=b"x"):
    yol = tmp_path / ad
    yol.write_bytes(icerik)
    return str(yol)


def _rttm(tmp_path, ad, segler):
    yol = str(tmp_path / ad)
    cikti.rttm_yaz(segler, os.path.splitext(ad)[0], yol)
    return yol


def test_tablolar():
    assert arayuz.segment_tablosu(SEGMENTLER)[1] == ["spk01", 2.0, 3.5, 1.5]
    assert arayuz.konusmaci_toplamlari(SEGMENTLER) == [["spk00", 3.0, 66.7], ["spk01", 1.5, 33.3]]
    assert [r[:2] for r in arayuz.degisim_tablosu(SEGMENTLER)] == [[1, 2.0], [2, 4.0]]
    assert arayuz.konusmaci_toplamlari([]) == [] and arayuz.degisim_tablosu([]) == []


@pytest.mark.parametrize("k,e,beklenen", [
    (None, None, (None, None)), ("", "", (None, None)), (float("nan"), None, (None, None)),
    (0, 0, (None, None)), (0.0, 0.0, (None, None)), ("0", "0", (None, None)),  # 0 = otomatik
    (3, 0.7, (3, 0.7)), (2.0, "0.5", (2, 0.5)), (0, 0.6, (None, 0.6)), (4, 0, (4, None)),
])
def test_girdileri_denetle(k, e, beklenen):
    assert arayuz.girdileri_denetle(k, e) == beklenen


def test_sifir_otomatik_hatta_none_gider(tmp_path):
    pytest.importorskip("matplotlib")
    durum, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "kayit.wav"), "ecapa", 0, 0, None, hat=sahte_hat)
    assert sahte_hat.cagri[2:] == (None, None) and "Hata" not in durum


@pytest.mark.parametrize("k,e", [(1.5, None), (-2, None), (None, -0.1)])
def test_girdileri_denetle_hata(k, e):
    with pytest.raises(ValueError):
        arayuz.girdileri_denetle(k, e)


def test_analiz_sahte_hat_rttm_ve_cizelge(tmp_path):
    pytest.importorskip("matplotlib")
    ses = _dosya(tmp_path, "kayit.wav")
    s = arayuz.analiz(ses, "mfcc", 2, None, cikti_dizin=str(tmp_path / "cikti"), hat=sahte_hat)
    assert sahte_hat.cagri == ("kayit.wav", "mfcc", 2, None)  # eşik boşsa main varsayılanı seçer
    assert s.dosya_id == "kayit" and s.ses_sn == 6.0 and s.islem_sn >= 0
    assert cikti.rttm_oku(s.rttm_yolu) == SEGMENTLER
    assert os.path.basename(s.rttm_yolu) == "kayit.rttm" and os.path.getsize(s.png_yolu) > 0
    assert s.referans is None and s.metrikler is None


def test_analiz_referansla_olcer(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyannote.metrics")
    ref = _rttm(tmp_path, "ref.rttm", SEGMENTLER)
    s = arayuz.analiz(_dosya(tmp_path, "kayit.wav"), ref_rttm=ref, cikti_dizin=str(tmp_path / "c"), hat=sahte_hat)
    assert s.metrikler["der"] == pytest.approx(0.0) and s.metrikler["f1"] == pytest.approx(1.0)
    assert "DER" in arayuz.metrik_metni(s.metrikler) and "%0.0" in arayuz.metrik_metni(s.metrikler)


def test_arayuz_cikti_basarili(tmp_path):
    pytest.importorskip("matplotlib")
    durum, png, seg, top, deg, rttm, metrik = arayuz.arayuz_cikti(
        _dosya(tmp_path, "kayit.wav"), "ecapa", None, None, None, hat=sahte_hat)
    assert "2 konuşmacı" in durum and "3 segment" in durum and "2 değişim" in durum
    assert len(seg) == 3 and len(top) == 2 and len(deg) == 2
    assert os.path.isfile(png) and os.path.isfile(rttm) and "Referans RTTM verilmedi" in metrik


def test_dosya_secilmedi():
    sonuc = arayuz.arayuz_cikti(None, "ecapa", None, None, None, hat=sahte_hat)
    assert len(sonuc) == arayuz.CIKTI_SAYISI and "ses dosyası seçin" in sonuc[0]
    assert sonuc[1] is None and sonuc[5] is None


def test_desteklenmeyen_bicim_mesaji(tmp_path):
    durum, *diger = arayuz.arayuz_cikti(_dosya(tmp_path, "notlar.xyz"), "ecapa", None, None, None)
    assert "notlar.xyz" in durum and "desteklenmeyen" in durum and diger[0] is None


def test_bozuk_wav_mesaji(tmp_path):
    durum, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "bozuk.wav", b"RIFF\0\0\0\0WAVEjunk"), "ecapa",
                                    None, None, None)
    assert "bozuk.wav" in durum and "bozuk veya okunamıyor" in durum


def test_uzun_dosya_mesaji(tmp_path, monkeypatch):
    monkeypatch.setattr(ses_girdi, "wav_denetle", lambda kaynak, ad=None: 2000.0)  # başlık 2000 sn diyor
    durum, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "uzun.wav"), "ecapa", None, None, None)
    assert "uzun.wav" in durum and "çok uzun" in durum and "1800" in durum


def test_bozuk_referans_mesaji(tmp_path):
    ref = _dosya(tmp_path, "ref.rttm", b"SPEAKER x 1 abc 1.0 <NA> <NA> a <NA> <NA>\n")
    durum, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "kayit.wav"), "ecapa", None, None, ref, hat=sahte_hat)
    assert "ref.rttm" in durum and "referans RTTM" in durum


def test_ayar_hatasi_mesaji(tmp_path):
    durum, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "kayit.wav"), "ecapa", -1, None, None, hat=sahte_hat)
    assert "ayarlar" in durum and "konuşmacı sayısı" in durum


def test_beklenmeyen_hata_cokmez(tmp_path):
    def patlayan(*_):
        raise RuntimeError("model yüklenemedi")

    durum, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "kayit.wav"), "ecapa", None, None, None, hat=patlayan)
    assert "kayit.wav" in durum and "beklenmeyen" in durum and "RuntimeError" in durum


def test_sessiz_sonuc(tmp_path):
    pytest.importorskip("matplotlib")
    durum, png, seg, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "sessiz.wav"), "ecapa", None, None, None,
                                              hat=lambda *a: ([], 5.0))
    assert "konuşma bulunamadı" in durum and seg == [] and os.path.isfile(png)


def test_ses_yolu_sec(tmp_path, monkeypatch):
    monkeypatch.setenv("KT_IZINLI_DIZIN", str(tmp_path))
    yuklenen = _dosya(tmp_path, "yuklenen.wav")
    sunucu = _dosya(tmp_path, "sunucu.wav")
    assert arayuz.ses_yolu_sec(yuklenen, None) == yuklenen
    assert arayuz.ses_yolu_sec(yuklenen, "  ") == yuklenen
    assert arayuz.ses_yolu_sec(None, None) is None
    assert arayuz.ses_yolu_sec(yuklenen, f' "{sunucu}" ') == os.path.realpath(sunucu)  # tırnak ve boşluk atılır
    with pytest.raises(FileNotFoundError, match="olmayan.wav"):
        arayuz.ses_yolu_sec(yuklenen, str(tmp_path / "olmayan.wav"))
    with pytest.raises(FileNotFoundError, match="klasör"):
        arayuz.ses_yolu_sec(None, str(tmp_path))


def test_sunucu_yolu_yuklenenin_yerine_gecer(tmp_path, monkeypatch):
    monkeypatch.setenv("KT_IZINLI_DIZIN", str(tmp_path))
    pytest.importorskip("matplotlib")
    durum, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "yuklenen.wav"), "ecapa", None, None, None,
                                    sunucu_yolu=_dosya(tmp_path, "drive_kaydi.wav"), hat=sahte_hat)
    assert sahte_hat.cagri[0] == "drive_kaydi.wav" and "drive_kaydi" in durum


def test_sunucu_yolu_yok_mesaji(tmp_path, monkeypatch):
    monkeypatch.setenv("KT_IZINLI_DIZIN", str(tmp_path))
    durum, png, *_ = arayuz.arayuz_cikti(None, "ecapa", None, None, None,
                                         sunucu_yolu=str(tmp_path / "yok.wav"), hat=sahte_hat)
    assert "yok.wav" in durum and "bulunamadı" in durum and png is None


@pytest.mark.parametrize("ad,beklenen", [
    ("ikgcq_028.0-118.0.wav", "data/sample/ikgcq_028.0-118.0.rttm"),
    ("kisa_1sn.wav", "data/sample/edge/kisa_1sn.rttm"),
    ("baska.wav", None),
])
def test_otomatik_referans(ad, beklenen):
    yol = arayuz.otomatik_referans(os.path.join("herhangi", ad))
    if beklenen is None:
        assert yol is None
    else:
        assert os.path.relpath(yol, arayuz.KOK).replace(os.sep, "/") == beklenen


def test_otomatik_referans_kullanilir_ve_belirtilir(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyannote.metrics")
    durum, *_, metrik = arayuz.arayuz_cikti(_dosya(tmp_path, "ikgcq_028.0-118.0.wav"), "ecapa", None, None, None,
                                            hat=sahte_hat)
    assert "referans: data/sample/ikgcq_028.0-118.0.rttm (otomatik)" in metrik.lower()
    assert "(otomatik)" in durum and "DER" in metrik


def test_yuklenen_referans_otomatigi_ezer(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyannote.metrics")
    ref = _rttm(tmp_path, "benim.rttm", SEGMENTLER)
    *_, metrik = arayuz.arayuz_cikti(_dosya(tmp_path, "ikgcq_028.0-118.0.wav"), "ecapa", None, None, ref,
                                     hat=sahte_hat)
    assert "benim.rttm (yüklenen)" in metrik and "otomatik" not in metrik and "%0.0" in metrik


def test_renkler_cizelgeyle_ayni():
    mpl = pytest.importorskip("matplotlib")
    from matplotlib.colors import to_hex

    for n, ad in ((3, "tab10"), (12, "tab20")):
        etiketler = [f"spk{i:02d}" for i in range(n)]
        harita = mpl.colormaps[ad]
        assert list(gorsel_renk(etiketler).values()) == [to_hex(harita(i)) for i in range(n)]
    # tablo HTML'i aynı rengi kullanır
    assert gorsel_renk(["spk00", "spk01"])["spk01"] in arayuz.segment_html(SEGMENTLER)


def gorsel_renk(etiketler):
    from src import gorsel

    return gorsel.renk_haritasi(etiketler)


def test_degisim_ciftleri_zamanlarla_ayni():
    ciftler = arayuz.degisim_ciftleri(SEGMENTLER)
    assert ciftler == [(2.0, "spk00", "spk01"), (4.0, "spk01", "spk00")]
    assert [t for t, *_ in ciftler] == cikti.degisim_noktalari(SEGMENTLER)


def test_html_parcalari():
    seg = arayuz.segment_html(SEGMENTLER)
    assert seg.count("<tr>") == 4 and "00:02.000" in seg and "1,50 sn" in seg
    sure = arayuz.sure_html(SEGMENTLER)
    assert "%66,7" in sure and "3,0 sn" in sure
    deg = arayuz.degisim_html(SEGMENTLER)
    assert deg.count("<tr>") == 3 and "→" in deg
    assert "Konuşma bulunamadı" in arayuz.segment_html([]) and "Değişim noktası yok" in arayuz.degisim_html([])
    assert arayuz.uyari_html("") == ""
    u = arayuz.uyari_html("**Hata — a<b>.wav:** bozuk.\n\nAyrıntı: x")
    assert 'role="alert"' in u and "<strong>" in u and "a&lt;b&gt;.wav" in u  # HTML kaçırılır


def test_gorunum_cikti_basarili(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyannote.metrics")
    sonuc = arayuz.gorunum_cikti(_dosya(tmp_path, "ikgcq_028.0-118.0.wav"), "ecapa", 0, 0, None, hat=sahte_hat)
    uyari, kartlar, png, seg, sure, deg, dogruluk, rttm = sonuc
    assert len(sonuc) == arayuz.GORUNUM_SAYISI and uyari == ""
    assert "konuşmacı" in kartlar and "6,0 sn" in kartlar and "(otomatik)" in kartlar
    assert os.path.isfile(png) and os.path.isfile(rttm) and "<table>" in seg
    assert "DER" in dogruluk and "(otomatik)" in dogruluk


def test_gorunum_cikti_referanssiz_dogruluk_bos(tmp_path):
    pytest.importorskip("matplotlib")
    sonuc = arayuz.gorunum_cikti(_dosya(tmp_path, "kayit.wav"), "ecapa", None, None, None, hat=sahte_hat)
    assert sonuc[0] == "" and sonuc[6] == ""


def test_gorunum_cikti_hata_kutusu(tmp_path):
    sonuc = arayuz.gorunum_cikti(_dosya(tmp_path, "notlar.xyz"), "ecapa", 0, 0, None)
    assert "kt-hata" in sonuc[0] and "notlar.xyz" in sonuc[0] and sonuc[1:] == ("", None, "", "", "", "", None)


def test_arayuz_kurulur():
    pytest.importorskip("gradio")
    demo = arayuz.arayuz_olustur(hat=sahte_hat)
    assert demo.title == arayuz.SAYFA_BASLIGI
    assert arayuz.tema() is not None and ".kt-kartlar" in arayuz.CSS
