"""src/kumeleme.py testleri (rastgele sentetik vektörler)."""
import numpy as np
import pytest

from src import kumeleme

pytest.importorskip("scipy")


def _iki_kume(n=20, d=26, tohum=0):
    r = np.random.default_rng(tohum)
    merkez_a = r.normal(size=d)
    merkez_b = -merkez_a + r.normal(scale=0.3, size=d)  # ters yönde: kosinüs mesafesi büyük
    A = merkez_a + r.normal(scale=0.05, size=(n, d))
    B = merkez_b + r.normal(scale=0.05, size=(n, d))
    return np.vstack([A, B]).astype(np.float32)


def test_iki_belirgin_kume_esikle():
    X = _iki_kume()
    et = kumeleme.kumele(X)
    assert et.shape == (40,) and et.dtype.kind == "i"
    assert len(set(et[:20])) == 1 and len(set(et[20:])) == 1 and et[0] != et[20]
    assert set(et.tolist()) == {0, 1}


def test_konusmaci_sayisi_ile():
    X = _iki_kume()
    et = kumeleme.kumele(X, konusmaci_sayisi=2)
    assert len(set(et[:20])) == 1 and len(set(et[20:])) == 1 and et[0] != et[20]
    et1 = kumeleme.kumele(X, konusmaci_sayisi=1)
    assert set(et1.tolist()) == {0}


def test_tek_konusmaci_tek_kume():
    r = np.random.default_rng(1)
    X = (r.normal(size=26) + r.normal(scale=0.05, size=(30, 26))).astype(np.float32)
    assert set(kumeleme.kumele(X).tolist()) == {0}


def test_esik_etkisi():
    X = _iki_kume()
    # Çok büyük eşik (> 2) her şeyi birleştirir, sıfır eşik neredeyse her noktayı ayırır.
    assert set(kumeleme.kumele(X, esik=2.1).tolist()) == {0}
    assert len(set(kumeleme.kumele(X, esik=0.0).tolist())) > 2


def test_kenar_durumlar():
    bos = kumeleme.kumele(np.zeros((0, 26), dtype=np.float32))
    assert bos.shape == (0,)
    assert kumeleme.kumele(np.ones((1, 26), dtype=np.float32)).tolist() == [0]
    with pytest.raises(ValueError):
        kumeleme.kumele(np.zeros(26, dtype=np.float32))
    with pytest.raises(ValueError):
        kumeleme.kumele(np.ones((3, 26), dtype=np.float32), konusmaci_sayisi=0)


def test_sifir_vektor_hata_vermez():
    X = _iki_kume()
    X[0] = 0
    et = kumeleme.kumele(X, konusmaci_sayisi=2)
    assert et.shape == (40,) and np.all(np.isfinite(et))


def test_etiketler_ilk_gorunme_sirasinda():
    X = _iki_kume()[::-1].copy()
    et = kumeleme.kumele(X)
    assert et[0] == 0
