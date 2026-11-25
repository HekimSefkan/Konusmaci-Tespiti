"""Opt-in singleton-support cleanup tests."""
import numpy as np
import pytest

from src import kumeleme

pytest.importorskip("scipy")


def test_yalniz_yakin_tek_pencere_kumesi_ana_kumeye_baglanir():
    X = np.array([
        [1.0, 0.0], [0.999, 0.01],  # major cluster 10
        [0.0, 1.0], [0.01, 0.999],  # major cluster 20
        [1.0, 0.01],                 # singleton near 10
        [-1.0, 0.0],                 # singleton too far from either major cluster
    ])
    labels = np.array([10, 10, 20, 20, 30, 40])

    merged = kumeleme._destek_kumelerini_birlestir(X, labels, en_cok_pencere=1, esik=0.2)

    assert merged[4] == 10
    assert merged[5] == 40


def test_destek_temizligi_varsayilaninda_kapali():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(24, 8)).astype(np.float32)

    assert np.array_equal(kumeleme.kumele(X), kumeleme.kumele_destekli(X, min_destek_pencere=0))


def test_destek_temizligi_bilinen_k_sayisiyla_kullanilamaz():
    X = np.random.default_rng(7).normal(size=(8, 4)).astype(np.float32)

    with pytest.raises(ValueError, match="yalnızca otomatik"):
        kumeleme.kumele_destekli(X, konusmaci_sayisi=2, min_destek_pencere=1)


def test_destek_sayisi_tamsayi_ve_negatif_olmali():
    X = np.random.default_rng(8).normal(size=(8, 4)).astype(np.float32)

    with pytest.raises(ValueError, match="tamsayı"):
        kumeleme.kumele_destekli(X, min_destek_pencere=0.5)
    with pytest.raises(ValueError, match="tamsayı"):
        kumeleme.kumele_destekli(X, min_destek_pencere=-1)
