"""src/main.py komut satırı testleri."""
import numpy as np
import pytest

from src import main as ana


def test_dosya_yok_cikis_1(tmp_path, capsys):
    kod = ana.main([str(tmp_path / "yok.wav"), "-o", str(tmp_path / "c.rttm")])
    assert kod == 1
    assert "yok.wav" in capsys.readouterr().err


def test_desteklenmeyen_bicim_cikis_1(tmp_path, capsys):
    yol = tmp_path / "metin.txt"
    yol.write_text("x", encoding="utf-8")
    assert ana.main([str(yol), "-o", str(tmp_path / "c.rttm")]) == 1
    assert "metin.txt" in capsys.readouterr().err


def test_sessiz_dosya_bos_rttm(tmp_path):
    # Colab: soundfile + silero-vad gerekir. Sessiz dosya -> boş RTTM, çıkış 0.
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("torch")
    pytest.importorskip("silero_vad")
    girdi = tmp_path / "sessiz.wav"
    sf.write(str(girdi), np.zeros(3 * 16000, dtype=np.float32), 16000)
    cikti_yolu = tmp_path / "sessiz.rttm"
    assert ana.main([str(girdi), "-o", str(cikti_yolu)]) == 0
    assert cikti_yolu.read_text(encoding="utf-8") == ""
