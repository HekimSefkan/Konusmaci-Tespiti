"""src/degisim_modeli.py testleri (sentetik vektörler; model yalnızca tmp_path'e yazılır)."""
import numpy as np
import pytest

from src import degisim_modeli as dm
from src.degerlendirme import _degisim_eslestir
from sentetik_veri import kayit_uret

pytest.importorskip("sklearn")


# --- özellikler ----------------------------------------------------------

def test_ozellikler_sekil_ve_degerler():
    P = [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0), (5.0, 6.5)]  # son pencere başka bölgede
    bolgeler = [(0.0, 3.0), (5.0, 6.5)]
    e = np.ones((4, 192), dtype=np.float32)
    m = np.ones((4, 26), dtype=np.float32)
    F = dm.ozellikler(P, e, m, bolgeler)
    assert F.shape == (3, len(dm.OZELLIKLER)) and F.dtype == np.float32
    assert np.allclose(F[:, 0], 1) and np.allclose(F[:, 1], 1) and np.allclose(F[:, 3], 1)
    assert F[:, 4] == pytest.approx([-0.75, -0.75, 2.0])  # bosluk_sn
    assert F[:, 5].tolist() == [1.0, 1.0, 0.0]            # ayni_bolge
    assert dm.ozellikler(P[:1], e[:1], m[:1], bolgeler).shape == (0, 6)


def test_ozellikler_degisimi_yakalar():
    r = np.random.default_rng(0)
    a, b = r.normal(size=192), r.normal(size=192)
    e = np.array([a, a, a, b, b, b], dtype=np.float32)
    P = [(i * 0.75, i * 0.75 + 1.5) for i in range(6)]
    F = dm.ozellikler(P, e, np.ones((6, 26)), [(0, 5.25)])
    assert np.argmin(F[:, 0]) == 2 and np.argmin(F[:, 1]) == 2  # değişim 2-3 arasında


def test_cift_zamanlari():
    assert dm.cift_zamanlari([(0, 1.5), (0.75, 2.25), (4, 5)]).tolist() == [1.125, 3.125]
    assert dm.cift_zamanlari([(0, 1)]).shape == (0,)


# --- etiketler -----------------------------------------------------------

def test_baskin_konusmaci_ve_belirsiz_pencere():
    ref = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]
    P = [(0.0, 1.5), (9.0, 10.5), (9.5, 11.0), (20.0, 21.5), (19.4, 20.9)]
    # 9.0-10.5: A 1,0 / 1,5 -> A ; 9.5-11.0: A 0,5, B 1,0 -> B ; 20-21.5: sessiz -> None
    # 19.4-20.9: B 0,6 / 1,5 (< yarı) -> None
    assert dm.baskin_konusmacilar(P, ref) == ["A", "A", "B", None, None]


def test_baskin_ayni_konusmaci_ortusmesi_iki_kez_sayilmaz():
    ref = [(0.0, 1.0, "A"), (0.5, 1.4, "A")]  # birleşimi 1,4 sn < 3 sn'nin yarısı
    assert dm.baskin_konusmacilar([(0.0, 3.0)], ref) == [None]


def test_baskin_esitlikte_alfabetik():
    ref = [(0.0, 2.0, "B"), (0.0, 2.0, "A")]
    assert dm.baskin_konusmacilar([(0.0, 1.5)], ref) == ["A"]


def test_cift_etiketleri():
    y, maske = dm.cift_etiketleri(["A", "A", "B", None, "B"])
    assert y[:2].tolist() == [0, 1] and maske.tolist() == [True, True, False, False]
    assert dm.cift_etiketleri(["A"])[0].shape == (0,)


# --- tepe seçme ----------------------------------------------------------

def test_tepeler():
    p = [0.1, 0.9, 0.2, 0.8, 0.85, 0.1]
    z = [1, 2, 3, 4, 5, 6]
    assert dm.tepeler(p, z, 0.5, 1.5) == [2.0, 5.0]
    assert dm.tepeler(p, z, 0.5, 3.5) == [2.0]
    assert dm.tepeler(p, z, 0.95, 0.5) == []
    assert dm.tepeler([], [], 0.5, 1.0) == []


def test_bolgeleri_kes():
    assert dm.bolgeleri_kes([(0, 10), (12, 20)], [5.0, 11.0, 12.0, 15.0]) == [
        (0, 5.0), (5.0, 10), (12, 15.0), (15.0, 20)]


# --- eğitim / kaydetme / yükleme -----------------------------------------

def _veriler(n=8):
    kayitlar = {f"k{i}": kayit_uret(i) for i in range(n)}
    return kayitlar, [dm.kayit_verisi(k, v["pencereler"], v["X_ecapa"], v["X_mfcc"], v["bolgeler"], v["ref"])
                      for k, v in kayitlar.items()]


def test_egit_kaydet_yukle_dongusu(tmp_path):
    kayitlar, veriler = _veriler()
    paket = dm.egit(veriler, n_kat=4, log=lambda *a: None)
    for anahtar in ("model", "secilen_model", "ozellik_surumu", "olasilik_esigi", "min_aralik_sn",
                    "egitim_tarihi", "egitim_kayit_sayisi", "cv", "baglam_k"):
        assert anahtar in paket
    assert len(paket["cv"]) == 7 and paket["egitim_kayit_sayisi"] == 8
    assert paket["olasilik_esigi"] in dm.OLASILIK_IZGARASI
    assert paket["min_aralik_sn"] in dm.MIN_ARALIK_IZGARASI

    yol = tmp_path / "model" / "degisim_modeli.joblib"
    dm.kaydet(paket, str(yol))
    geri = dm.yukle(str(yol))
    assert geri["secilen_model"] == paket["secilen_model"]

    # Görülmemiş bir kayıtta değişimleri bulmalı
    v = kayit_uret(99)
    hyp = dm.degisim_tahmin(geri, v["pencereler"], v["X_ecapa"], v["X_mfcc"], v["bolgeler"])
    ref_d = dm.cikti.degisim_noktalari(v["ref"])
    tp, fp, fn = _degisim_eslestir(ref_d, hyp, 1.0)
    assert tp / len(ref_d) >= 0.5


def test_yukle_surum_uyusmazligi(tmp_path):
    import joblib

    yol = tmp_path / "eski.joblib"
    joblib.dump({"ozellik_surumu": "v0"}, yol)
    with pytest.raises(ValueError):
        dm.yukle(str(yol))


def test_egit_yetersiz_veri():
    _, veriler = _veriler(1)
    with pytest.raises(ValueError):
        dm.egit(veriler, log=lambda *a: None)


def test_e5_segmentleri():
    pytest.importorskip("pyannote.metrics")
    from src.degerlendirme import _segmentleri_degerlendir

    _, veriler = _veriler()
    paket = dm.egit(veriler, n_kat=4, log=lambda *a: None)
    v = kayit_uret(123)
    seg = dm.e5_segmentleri(paket, v["bolgeler"], v["pencereler"], v["X_ecapa"], v["X_mfcc"], esik=0.5)
    assert seg == sorted(seg) and {e for _, _, e in seg} <= {"spk00", "spk01", "spk02", "spk03"}
    assert _segmentleri_degerlendir(v["ref"], seg)["der"] < 0.25
    # sessiz ve kısa konuşma kuralları hat_segmentleri ile aynı
    assert dm.e5_segmentleri(paket, [], [], np.zeros((0, 192)), np.zeros((0, 26))) == []
    assert dm.e5_segmentleri(paket, [(0.0, 1.0)], [(0.0, 1.0)], np.ones((1, 192)), np.ones((1, 26))) == [
        (0.0, 1.0, "spk00")]
