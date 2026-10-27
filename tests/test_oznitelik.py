"""src/oznitelik.py testleri. pencerele saf Python; vektorler librosa/speechbrain ister (Colab)."""
import numpy as np
import pytest

from src import oznitelik


# --- pencerele (yerelde çalışır) -----------------------------------------

def test_bos_girdi():
    assert oznitelik.pencerele([]) == []


def test_kisa_bolge_tek_pencere():
    assert oznitelik.pencerele([(2.0, 3.0)]) == [(2.0, 3.0)]
    assert oznitelik.pencerele([(2.0, 3.5)]) == [(2.0, 3.5)]  # tam pencere_sn


def test_tam_bolunen_bolge():
    assert oznitelik.pencerele([(0.0, 3.0)]) == [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0)]


def test_kuyruk_sona_hizalanir_ve_siniri_asmaz():
    p = oznitelik.pencerele([(0.0, 3.4)])
    assert p == [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0), (1.9, 3.4)]
    assert all(b <= 3.4 for _, b in p)


def test_cok_bolge_sinir_asmaz_ve_sirali():
    bolgeler = [(10.0, 12.2), (0.3, 4.0), (5.0, 5.4)]
    p = oznitelik.pencerele(bolgeler)
    assert p == sorted(p)
    for a, b in p:
        assert any(ba - 1e-9 <= a and b <= bb + 1e-9 for ba, bb in bolgeler)
        assert b - a <= 1.5 + 1e-9
    # her bölgenin başı ve sonu kapsanır
    for ba, bb in bolgeler:
        assert any(abs(a - ba) < 1e-9 for a, _ in p) and any(abs(b - bb) < 1e-9 for _, b in p)


def test_parametre_hatalari():
    with pytest.raises(ValueError):
        oznitelik.pencerele([(0, 3)], pencere_sn=0)
    with pytest.raises(ValueError):
        oznitelik.pencerele([(0, 3)], kaydirma_sn=-1)


def test_bilinmeyen_yontem_ve_bos_pencere():
    ses = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError):
        oznitelik.vektorler(ses, 16000, [(0, 1)], yontem="xvector")
    assert oznitelik.vektorler(ses, 16000, [], "mfcc").shape == (0, 26)
    assert oznitelik.vektorler(ses, 16000, [], "ecapa").shape == (0, 192)


# --- genel ölçek (yerelde çalışır) ----------------------------------------

def test_olcek_hesapla_ve_olcekle():
    r = np.random.default_rng(0)
    A = r.normal(5.0, 2.0, size=(300, 26))
    B = r.normal(5.0, 2.0, size=(200, 26))
    olcek = oznitelik.olcek_hesapla([A, np.zeros((0, 26)), B])
    assert olcek["pencere_sayisi"] == 500 and len(olcek["ortalama"]) == 26
    Z = oznitelik.olcekle(np.vstack([A, B]), olcek)
    assert Z.dtype == np.float32
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-4) and np.allclose(Z.std(axis=0), 1, atol=1e-3)
    # kayıt başına değil, genel: tek kaydın ortalaması sıfırlanmaz
    tek = oznitelik.olcekle(A + 3.0, olcek)
    assert tek.mean() > 1.0
    assert oznitelik.olcekle(A, None) is not None and np.allclose(oznitelik.olcekle(A, None), A)
    sabit = oznitelik.olcek_hesapla([np.ones((10, 26))])  # std 0 -> alt sınır, bölme hatası yok
    assert np.all(np.isfinite(oznitelik.olcekle(np.ones((3, 26)), sabit)))
    with pytest.raises(ValueError):
        oznitelik.olcek_hesapla([np.zeros((0, 26))])


# --- vektorler (librosa / speechbrain: Colab) -----------------------------

def _sentetik(sure_sn=4.0, sr=16000):
    t = np.arange(int(sure_sn * sr)) / sr
    r = np.random.default_rng(0)
    return (0.1 * np.sin(2 * np.pi * 220 * t) + 0.01 * r.normal(size=t.size)).astype(np.float32)


def test_mfcc_vektorleri():
    pytest.importorskip("librosa")
    ses = _sentetik()
    p = oznitelik.pencerele([(0.0, 4.0)])
    X = oznitelik.vektorler(ses, 16000, p, "mfcc")
    assert X.shape == (len(p), 26) and X.dtype == np.float32 and np.all(np.isfinite(X))


def test_ecapa_vektorleri():
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    from src.uyumluluk import speechbrain_hazirla

    speechbrain_hazirla()  # importorskip speechbrain'i içe aktarır: yama önce gelmeli
    pytest.importorskip("speechbrain")
    ses = _sentetik()
    p = oznitelik.pencerele([(0.0, 4.0)]) + [(3.9, 4.0)]  # farklı uzunluklar: toplu dolgu
    X = oznitelik.vektorler(ses, 16000, p, "ecapa")
    assert X.shape == (len(p), 192) and X.dtype == np.float32 and np.all(np.isfinite(X))
    assert oznitelik.ecapa_modeli() is oznitelik.ecapa_modeli()  # bir kez yüklenir
