"""src/vad.py testleri. Silero VAD (torch + silero-vad) ister: Colab'da çalışır."""
import numpy as np
import pytest

from src import vad

SR = 16000


@pytest.fixture
def silero():
    pytest.importorskip("torch")
    return pytest.importorskip("silero_vad")


def test_bos_ses():
    # model yüklenmeden döner: yerelde de çalışır
    assert vad.konusma_bolgeleri(np.zeros(0, dtype=np.float32), SR) == []


def test_sessizlik_bos_liste(silero):
    assert vad.konusma_bolgeleri(np.zeros(3 * SR, dtype=np.float32), SR) == []


def test_cikti_sirali_ortusmeyen_ve_sinirlar_icinde(silero):
    r = np.random.default_rng(0)
    t = np.arange(6 * SR) / SR
    zarf = (np.sin(2 * np.pi * 0.5 * t) > 0).astype(np.float32)  # 1 sn ses, 1 sn sessizlik
    x = (zarf * (0.3 * np.sin(2 * np.pi * 180 * t) + 0.05 * r.normal(size=t.size))).astype(np.float32)
    b = vad.konusma_bolgeleri(x, SR)
    assert b == sorted(b)
    for (a1, b1), (a2, _) in zip(b, b[1:]):
        assert b1 < a2
    assert all(0 <= a < bb <= 6.0 for a, bb in b)
    assert vad._silero_modeli() is vad._silero_modeli()  # model bir kez yüklenir
