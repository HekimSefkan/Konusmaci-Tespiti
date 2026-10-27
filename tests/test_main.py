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


# --- yönteme göre varsayılan eşik ve MFCC ölçeği (ses ve model gerekmez) ---

@pytest.fixture
def sahte_hat(monkeypatch):
    """Ses okuma, VAD, öznitelik ve kümelemeyi yamalar; kumele'ye gelenleri kaydeder."""
    from src import kumeleme, oznitelik, ses_girdi, vad

    kayit = {}
    r = np.random.default_rng(0)
    monkeypatch.setattr(ses_girdi, "ses_yukle", lambda yol: (np.zeros(16000 * 6, np.float32), 16000))
    monkeypatch.setattr(vad, "konusma_bolgeleri", lambda ses, sr: [(0.0, 6.0)])

    def vektorler(ses, sr, pencereler, yontem):
        kayit["yontem"] = yontem
        kayit["X_ham"] = r.normal(3.0, 2.0, size=(len(pencereler), oznitelik.BOYUT[yontem])).astype(np.float32)
        return kayit["X_ham"]

    def kumele(X, konusmaci_sayisi=None, esik=None):
        kayit["X"], kayit["esik"] = np.asarray(X), esik
        return np.zeros(len(X), dtype=int)

    monkeypatch.setattr(oznitelik, "vektorler", vektorler)
    monkeypatch.setattr(kumeleme, "kumele", kumele)
    return kayit


def test_varsayilan_yontem_ecapa_ve_esigi(sahte_hat, tmp_path):
    from src import varsayilanlar

    assert ana.main([str(tmp_path / "x.wav"), "-o", str(tmp_path / "x.rttm")]) == 0
    assert sahte_hat["yontem"] == "ecapa"
    assert sahte_hat["esik"] == varsayilanlar.esik("ecapa")
    assert np.array_equal(sahte_hat["X"], sahte_hat["X_ham"])  # ECAPA'ya ölçek uygulanmaz
    assert (tmp_path / "x.rttm").read_text(encoding="utf-8").startswith("SPEAKER x 1 0.000")


def test_mfcc_esigi_ve_olcegi(sahte_hat, tmp_path):
    from src import oznitelik, varsayilanlar

    olcek = varsayilanlar.mfcc_olcek()
    if olcek is None:
        with pytest.warns(UserWarning, match="MFCC ölçeği"):
            ana.main([str(tmp_path / "x.wav"), "-o", str(tmp_path / "x.rttm"), "--yontem", "mfcc"])
    else:
        ana.main([str(tmp_path / "x.wav"), "-o", str(tmp_path / "x.rttm"), "--yontem", "mfcc"])
    assert sahte_hat["yontem"] == "mfcc"
    assert sahte_hat["esik"] == varsayilanlar.esik("mfcc")
    assert np.allclose(sahte_hat["X"], oznitelik.olcekle(sahte_hat["X_ham"], olcek))


def test_esik_ve_konusmaci_sayisi_onceligi(sahte_hat, tmp_path):
    ana.main([str(tmp_path / "x.wav"), "-o", str(tmp_path / "x.rttm"), "--esik", "0.42"])
    assert sahte_hat["esik"] == 0.42
    ana.main([str(tmp_path / "x.wav"), "-o", str(tmp_path / "x.rttm"), "--konusmaci-sayisi", "2"])
    assert sahte_hat["esik"] is None  # sayı verilince eşik kullanılmaz
