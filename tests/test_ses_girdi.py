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
    # RIFF imzası olmayan dosya: başlık denetimi soundfile'dan önce yakalar (yerelde çalışır)
    yol = tmp_path / "bozuk.wav"
    yol.write_bytes(b"bu bir wav dosyasi degil" * 10)
    with pytest.raises(ValueError, match="bozuk.wav"):
        ses_girdi.ses_yukle(str(yol))


# --- WAV başlık denetimi (standart kütüphane: yerelde çalışır) ------------

def _wav_baytlari(sure_sn=1.0, sr=16000, kanal=1):
    """wave modülüyle bellekte sentetik 16 bit WAV (diske yazılmaz)."""
    import io
    import wave

    x = (8000 * np.sin(2 * np.pi * 300 * np.arange(int(sr * sure_sn)) / sr)).astype("<i2")
    x = np.repeat(x[:, None], kanal, axis=1)
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as w:
        w.setnchannels(kanal)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())
    return tampon.getvalue()


def test_wav_denetle_saglam():
    assert ses_girdi.wav_denetle(_wav_baytlari(1.0)) == pytest.approx(1.0)
    assert ses_girdi.wav_denetle(_wav_baytlari(0.5, sr=8000, kanal=2)) == pytest.approx(0.5)


def test_wav_denetle_yarida_kesilmis():
    b = _wav_baytlari(1.0)
    with pytest.raises(ValueError, match=r"x\.wav: bozuk/kesik WAV: başlıkta 1\.00 sn .*dosyada 0\.50 sn"):
        ses_girdi.wav_denetle(b[: len(b) // 2], "x.wav")


def test_wav_denetle_bir_cerceve_payi():
    mono, stereo = _wav_baytlari(1.0), _wav_baytlari(1.0, kanal=2)
    ses_girdi.wav_denetle(mono[:-2])      # 1 çerçeve (2 bayt) eksik: kabul
    ses_girdi.wav_denetle(stereo[:-4])    # stereo 1 çerçeve (4 bayt) eksik: kabul
    with pytest.raises(ValueError):
        ses_girdi.wav_denetle(mono[:-4])  # 2 çerçeve eksik: kesik


def test_wav_denetle_tutarsiz_baslik():
    import struct

    b = _wav_baytlari(0.1)
    with pytest.raises(ValueError, match="RIFF/WAVE"):
        ses_girdi.wav_denetle(b"RIFX" + b[4:])
    sifir_kanal = b[:22] + struct.pack("<H", 0) + b[24:]  # fmt: kanal sayısı 0
    with pytest.raises(ValueError, match="fmt alanları"):
        ses_girdi.wav_denetle(sifir_kanal)
    with pytest.raises(ValueError, match="data"):
        ses_girdi.wav_denetle(b[:36])  # data bölümü hiç yok
    with pytest.raises(ValueError):
        ses_girdi.wav_denetle(b[:10])


def test_ses_yukle_kesik_wav(tmp_path):
    # bozuk_yarim.wav gibi: başlık sağlam, veri yarıda kesik -> ValueError (soundfile'a gelmeden)
    b = _wav_baytlari(1.0)
    yol = tmp_path / "bozuk_yarim.wav"
    yol.write_bytes(b[: len(b) // 2])
    try:
        with pytest.raises(ValueError, match="bozuk_yarim.wav"):
            ses_girdi.ses_yukle(str(yol))
    finally:
        yol.unlink()
