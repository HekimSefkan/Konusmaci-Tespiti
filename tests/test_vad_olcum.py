"""src/vad_olcum.py testleri: ölçüm mantığı numpy ile yerelde; WebRTC/Silero Colab'da."""
import csv

import numpy as np
import pytest

from src import cikti, vad_olcum


# --- saf yardımcılar -------------------------------------------------------

def test_cercevelerden_bolgeler():
    k = [0, 1, 1, 0, 0, 1, 1, 1]
    assert vad_olcum.cercevelerden_bolgeler(k, 0.03) == [(0.03, 0.09), (0.15, 0.24)]
    assert vad_olcum.cercevelerden_bolgeler([], 0.03) == []
    assert vad_olcum.cercevelerden_bolgeler([1, 1], 0.03) == [(0.0, 0.06)]


def test_yumusat_bosluk_kisa_konusma_kenar_payi():
    # 0,2 sn boşluk (< 0,3) birleşir; 0,1 sn'lik konuşma (< 0,25) atılır; 30 ms pay eklenir
    b = vad_olcum.yumusat([(1.0, 2.0), (2.2, 3.0), (5.0, 5.1)], sure_sn=10.0)
    assert b == [(0.97, 3.03)]
    # 0,3 sn boşluk birleşmez (sınır: < min sessizlik)
    assert vad_olcum.yumusat([(1.0, 2.0), (2.3, 3.0)], 10.0) == [(0.97, 2.03), (2.27, 3.03)]
    # pay dosya sınırlarında kırpılır, paylar örtüşürse birleşir
    assert vad_olcum.yumusat([(0.0, 1.0), (9.5, 10.0)], 10.0) == [(0.0, 1.03), (9.47, 10.0)]
    assert vad_olcum.yumusat([], 5.0) == []


def test_maske_ve_hata_olc():
    ref = vad_olcum.maske([(1.0, 3.0)], 5.0)           # 2 sn konuşma, 3 sn sessizlik
    hyp = vad_olcum.maske([(1.5, 3.5)], 5.0)           # 0,5 sn kaçırılan, 0,5 sn yanlış alarm
    assert ref.shape == (500,) and ref.sum() == 200
    o = vad_olcum.hata_olc(ref, hyp)
    assert o == pytest.approx({"ref_konusma_sn": 2.0, "ref_sessiz_sn": 3.0, "kacirilan_sn": 0.5, "yanlis_alarm_sn": 0.5})
    with pytest.raises(ValueError):
        vad_olcum.hata_olc(ref, hyp[:-1])
    assert vad_olcum.maske([(4.9, 7.0)], 5.0).sum() == 10  # sınır aşılmaz


def test_ozet_hesapla_mikro():
    satirlar = [
        {"vad": "webrtc2", "kayit": "a", "ses_sn": 1800, "ref_konusma_sn": 100, "ref_sessiz_sn": 20,
         "kacirilan_sn": 5, "yanlis_alarm_sn": 4, "sure_sn": 3},
        {"vad": "webrtc2", "kayit": "b", "ses_sn": 1800, "ref_konusma_sn": 300, "ref_sessiz_sn": 20,
         "kacirilan_sn": 15, "yanlis_alarm_sn": 4, "sure_sn": 1},
        {"vad": "silero", "kayit": "a", "ses_sn": "1800", "ref_konusma_sn": "100", "ref_sessiz_sn": "0",
         "kacirilan_sn": "1", "yanlis_alarm_sn": "0", "sure_sn": "9"},
    ]
    o = {r["vad"]: r for r in vad_olcum.ozet_hesapla(satirlar)}
    assert [r["vad"] for r in vad_olcum.ozet_hesapla(satirlar)] == ["silero", "webrtc2"]
    w = o["webrtc2"]
    assert w["kayit_sayisi"] == 2 and w["ses_saat"] == 1.0
    assert w["kacirilan_yuzde"] == pytest.approx(5.0) and w["yanlis_alarm_yuzde"] == pytest.approx(2.0)
    assert w["toplam_hata_yuzde"] == pytest.approx(7.0) and w["yanlis_alarm_sessiz_yuzde"] == pytest.approx(20.0)
    assert w["sn_per_saat"] == pytest.approx(4.0)
    assert np.isnan(o["silero"]["yanlis_alarm_sessiz_yuzde"])  # sessiz süre 0


# --- uçtan uca (ses ve VAD'lar yamalı; zip açılmaz) ------------------------

@pytest.fixture
def sahte_ortam(tmp_path, monkeypatch):
    rttm = tmp_path / "rttm"
    rttm.mkdir()
    for k in ("k1", "k2", "k3"):
        cikti.rttm_yaz([(1.0, 4.0, "A"), (5.0, 8.0, "B")], k, str(rttm / f"{k}.rttm"))
    cagrilar = []

    def oku(self, kayit):
        return np.zeros(16000 * 10, dtype=np.float32), 16000

    def calistir(ad, ses, sr):
        cagrilar.append(ad)
        return {"silero": [(1.0, 8.0)], "webrtc2": [(1.0, 4.0), (5.0, 8.0)], "webrtc3": [(2.0, 4.0)]}[ad]

    monkeypatch.setattr(vad_olcum.SesKaynagi, "oku", oku)
    monkeypatch.setattr(vad_olcum, "_vad_calistir", calistir)
    monkeypatch.setattr(vad_olcum.vad, "_silero_modeli", lambda: None)
    return {"rttm": str(rttm), "cikti": str(tmp_path / "sonuc"), "cagrilar": cagrilar, "zip": str(tmp_path / "yok.zip")}


def test_olc_csv_ozet_ve_devam(sahte_ortam):
    o = sahte_ortam
    ozet = vad_olcum.olc(o["zip"], o["rttm"], ["k1", "k2", "k3"], o["cikti"], limit=2, log=lambda *a: None)
    d = {r["vad"]: r for r in ozet}
    assert d["webrtc2"]["toplam_hata_yuzde"] == pytest.approx(0.0)
    assert d["silero"]["yanlis_alarm_yuzde"] == pytest.approx(100 / 6, abs=1e-3)   # 1 sn / 6 sn (3 ondalık)
    assert d["webrtc3"]["kacirilan_yuzde"] == pytest.approx(400 / 6, abs=1e-3)     # 4 sn / 6 sn
    assert d["silero"]["kayit_sayisi"] == 2 and len(o["cagrilar"]) == 6
    with open(f"{o['cikti']}/{vad_olcum.OZET_CSV}", encoding="utf-8") as f:
        assert next(csv.reader(f)) == vad_olcum.OZET_SUTUNLARI
    # devam: yapılan kayıtlar yeniden çalıştırılmaz, yalnızca k3 eklenir
    vad_olcum.olc(o["zip"], o["rttm"], ["k1", "k2", "k3"], o["cikti"], log=lambda *a: None)
    assert len(o["cagrilar"]) == 9
    with open(f"{o['cikti']}/{vad_olcum.KAYIT_CSV}", encoding="utf-8") as f:
        satirlar = list(csv.DictReader(f))
    assert len(satirlar) == 9 and list(satirlar[0]) == vad_olcum.KAYIT_SUTUNLARI


def test_cli(sahte_ortam, tmp_path, capsys):
    o = sahte_ortam
    liste = tmp_path / "liste.txt"
    liste.write_text("k1\nk2\n", encoding="utf-8")
    assert vad_olcum.main(["--zip", o["zip"], "--rttm-dizin", o["rttm"], "--liste", str(liste),
                           "--cikti-dizin", o["cikti"]]) == 0
    assert "webrtc3" in capsys.readouterr().out
    assert vad_olcum.main(["--zip", o["zip"], "--rttm-dizin", o["rttm"], "--liste", str(tmp_path / "yok.txt"),
                           "--cikti-dizin", o["cikti"]]) == 1


# --- gerçek WebRTC (Colab) -------------------------------------------------

def test_webrtc_bolgeleri():
    pytest.importorskip("webrtcvad")
    sr = 16000
    r = np.random.default_rng(0)
    t = np.arange(sr * 4) / sr
    konusma = (t >= 1.0) & (t < 3.0)
    x = np.where(konusma, 0.3 * np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t)), 0.0)
    x = (x + 0.001 * r.normal(size=t.size)).astype(np.float32)
    for agresiflik in (2, 3):
        b = vad_olcum.webrtc_bolgeleri(x, sr, agresiflik)
        assert b == sorted(b) and all(0 <= a < c <= 4.0 for a, c in b)
