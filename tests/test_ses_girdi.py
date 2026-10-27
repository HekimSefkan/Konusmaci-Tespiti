"""src/ses_girdi.py testleri. Ön işleme numpy/scipy ile yerelde; dosya okuma soundfile ister (Colab)."""
import numpy as np
import pytest

from src import ses_girdi
from src.ses_girdi import SR, on_isle


def _sinus(sr, sure_sn=1.0, frekans=440.0, genlik=0.5):
    t = np.arange(int(sr * sure_sn)) / sr
    return genlik * np.sin(2 * np.pi * frekans * t)


def _dbfs(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# --- on_isle (yerelde çalışır) ------------------------------------------

def test_mono_kanal_ortalamasi():
    sol = _sinus(SR)
    stereo = np.stack([sol, np.zeros_like(sol)], axis=1)
    x = on_isle(stereo, SR)
    assert x.ndim == 1 and x.dtype == np.float32 and len(x) == SR


def test_yeniden_ornekleme():
    x = on_isle(_sinus(44100, sure_sn=2.0), 44100)
    assert len(x) == 2 * SR


def test_rms_normalizasyonu():
    x = on_isle(_sinus(SR, genlik=0.01), SR)
    assert _dbfs(x) == pytest.approx(-20.0, abs=0.1)
    assert np.max(np.abs(x)) <= 0.99 + 1e-6


def test_tepe_siniri():
    # Seyrek darbe: -20 dBFS'e çekmek tepeyi 1'in üstüne çıkarırdı -> tepe 0,99'da kalır
    x = np.zeros(SR)
    x[::4000] = 1.0
    y = on_isle(x, SR)
    assert np.max(np.abs(y)) == pytest.approx(0.99, abs=1e-4)
    assert _dbfs(y) < -20.0


def test_sessiz_ses_degismez():
    y = on_isle(np.zeros(SR), SR)
    assert np.all(y == 0)


def test_on_isle_hatalari():
    with pytest.raises(ValueError):
        on_isle(np.zeros(0), SR)
    with pytest.raises(ValueError):
        on_isle(np.array([0.1, np.nan, 0.2]), SR)


# --- ses_yukle hata yolları (yerelde çalışır) ----------------------------

def test_dosya_yok(tmp_path):
    with pytest.raises(FileNotFoundError, match="yok.wav"):
        ses_girdi.ses_yukle(str(tmp_path / "yok.wav"))


def test_desteklenmeyen_bicim(tmp_path):
    yol = tmp_path / "not.txt"
    yol.write_text("ses değil", encoding="utf-8")
    with pytest.raises(ValueError, match="not.txt"):
        ses_girdi.ses_yukle(str(yol))


# --- dosya okuma (soundfile: Colab) --------------------------------------

def test_wav_okuma_stereo_44k(tmp_path):
    sf = pytest.importorskip("soundfile")
    yol = tmp_path / "stereo.wav"
    x = _sinus(44100, sure_sn=1.0, genlik=0.3)
    sf.write(str(yol), np.stack([x, x], axis=1), 44100)
    ses, sr = ses_girdi.ses_yukle(str(yol))
    assert sr == 16000 and ses.dtype == np.float32 and ses.shape == (16000,)
    assert _dbfs(ses) == pytest.approx(-20.0, abs=0.2)


def test_bozuk_wav(tmp_path):
    pytest.importorskip("soundfile")
    yol = tmp_path / "bozuk.wav"
    yol.write_bytes(b"bu bir wav dosyasi degil" * 10)
    with pytest.raises(ValueError, match="bozuk.wav"):
        ses_girdi.ses_yukle(str(yol))
