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
    (3, 0.7, (3, 0.7)), (2.0, "0.5", (2, 0.5)),
])
def test_girdileri_denetle(k, e, beklenen):
    assert arayuz.girdileri_denetle(k, e) == beklenen


@pytest.mark.parametrize("k,e", [(0, None), (1.5, None), (-2, None), (None, 0), (None, -0.1)])
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
    durum, *_ = arayuz.arayuz_cikti(_dosya(tmp_path, "kayit.wav"), "ecapa", 0, None, None, hat=sahte_hat)
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


def test_arayuz_kurulur():
    pytest.importorskip("gradio")
    demo = arayuz.arayuz_olustur(hat=sahte_hat)
    assert demo.title == arayuz.BASLIK
