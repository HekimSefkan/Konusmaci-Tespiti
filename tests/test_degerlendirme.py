"""src/degerlendirme.py testleri (elle yazılmış RTTM'ler)."""
import math
import os

import pytest

from src import degerlendirme
from src.degerlendirme import _degisim_eslestir

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rttm(tmp_path, ad, segler):
    yol = tmp_path / f"{ad}.rttm"
    yol.write_text(
        "".join(f"SPEAKER {ad} 1 {a:.3f} {b - a:.3f} <NA> <NA> {e} <NA> <NA>\n" for a, b, e in segler),
        encoding="utf-8",
    )
    return str(yol)


# --- Değişim noktası eşleştirme (pyannote gerekmez) ----------------------

def test_eslestirme_temel():
    assert _degisim_eslestir([10, 20], [10.3, 19.9, 30], 0.5) == (2, 1, 0)


def test_eslestirme_en_yakin_once():
    # 10.4 hem 10'a (0,4) hem 10.6'ya (0,2) yakın: en yakın çift (10.6) önce eşleşir
    assert _degisim_eslestir([10.0, 10.6], [10.4], 0.5) == (1, 0, 1)


def test_eslestirme_bire_bir():
    # iki tahmin aynı referansa yakın: yalnızca biri eşleşir
    assert _degisim_eslestir([5.0], [4.9, 5.2], 0.5) == (1, 1, 0)


def test_eslestirme_tolerans_siniri_dahil():
    assert _degisim_eslestir([5.0], [5.5], 0.5) == (1, 0, 0)
    assert _degisim_eslestir([5.0], [5.501], 0.5) == (0, 1, 1)


def test_eslestirme_bos():
    assert _degisim_eslestir([], [], 0.5) == (0, 0, 0)
    assert _degisim_eslestir([], [1.0], 0.5) == (0, 1, 0)


# --- DER / JER (pyannote.metrics) ----------------------------------------

@pytest.fixture
def pyannote():
    """DER/JER testleri pyannote.metrics ister; yoksa yalnızca bu testler atlanır."""
    return pytest.importorskip("pyannote.metrics")

REF = [(0.0, 10.0, "A"), (10.0, 20.0, "B")]


def test_ayni_rttm(tmp_path, pyannote):
    r = _rttm(tmp_path, "ref", REF)
    h = _rttm(tmp_path, "hyp", [(0.0, 10.0, "x"), (10.0, 20.0, "y")])  # etiket adları farklı olabilir
    s = degerlendirme.degerlendir(r, h)
    assert s["der"] == pytest.approx(0) and s["jer"] == pytest.approx(0)
    assert s["f1"] == 1 and (s["tp"], s["fp"], s["fn"]) == (1, 0, 0)
    beklenen = {"der", "jer", "f1", "precision", "recall", "tp", "fp", "fn",
                "yanlis_alarm_sn", "kacirilan_sn", "karisiklik_sn", "referans_sn"}
    assert set(s) == beklenen


def test_bilinen_karisiklik(tmp_path, pyannote):
    r = _rttm(tmp_path, "ref", REF)
    h = _rttm(tmp_path, "hyp", [(0.0, 10.0, "A"), (10.0, 15.0, "B"), (15.0, 20.0, "A")])
    s = degerlendirme.degerlendir(r, h, collar=0.0)
    assert s["karisiklik_sn"] == pytest.approx(5.0)
    assert s["der"] == pytest.approx(0.25)
    # değişim: ref [10], hyp [10, 15] -> tp 1, fp 1
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 0)
    assert s["precision"] == pytest.approx(0.5) and s["recall"] == 1
    assert s["f1"] == pytest.approx(2 / 3)


def test_kacirilan_ve_ortusme_dahil(tmp_path, pyannote):
    # Referansta 5-10 arası örtüşme; tahmin yalnız A: kaçırılan 10 sn / toplam 20 sn
    r = _rttm(tmp_path, "ref", [(0.0, 10.0, "A"), (5.0, 15.0, "B")])
    h = _rttm(tmp_path, "hyp", [(0.0, 10.0, "A")])
    s = degerlendirme.degerlendir(r, h, collar=0.0)
    assert s["kacirilan_sn"] == pytest.approx(10.0)
    assert s["referans_sn"] == pytest.approx(20.0)
    assert s["der"] == pytest.approx(0.5)


def test_collar(tmp_path, pyannote):
    r = _rttm(tmp_path, "ref", REF)
    h = _rttm(tmp_path, "hyp", [(0.0, 10.2, "A"), (10.2, 20.0, "B")])
    assert degerlendirme.degerlendir(r, h, collar=0.0)["der"] == pytest.approx(0.01)
    assert degerlendirme.degerlendir(r, h, collar=0.5)["der"] == pytest.approx(0.0)


def test_sessiz_referans(tmp_path, pyannote):
    r = _rttm(tmp_path, "ref", [])
    h = _rttm(tmp_path, "hyp", [(0.0, 2.0, "A")])
    s = degerlendirme.degerlendir(r, h)
    assert math.isnan(s["der"]) and math.isnan(s["jer"])
    assert s["yanlis_alarm_sn"] == pytest.approx(2.0) and s["referans_sn"] == 0
    assert s["f1"] == 1 and (s["tp"], s["fp"], s["fn"]) == (0, 0, 0)

    h2 = _rttm(tmp_path, "hyp2", [(0.0, 2.0, "A"), (2.0, 4.0, "B")])
    s2 = degerlendirme.degerlendir(r, h2)
    assert s2["yanlis_alarm_sn"] == pytest.approx(4.0)
    assert (s2["precision"], s2["recall"], s2["f1"]) == (0.0, 1.0, 0.0) and s2["fp"] == 1


def test_iki_taraf_bos(tmp_path, pyannote):
    r = _rttm(tmp_path, "ref", [])
    s = degerlendirme.degerlendir(r, r)
    assert math.isnan(s["der"]) and s["yanlis_alarm_sn"] == 0 and s["f1"] == 1


def test_ornek_veride_ayni_dosya_der_sifir(pyannote):
    yol = os.path.join(KOK, "data", "sample", "migzj_066.5-136.5.rttm")  # %30 örtüşmeli
    s = degerlendirme.degerlendir(yol, yol)
    assert s["der"] == pytest.approx(0) and s["f1"] == 1 and s["tp"] == 12


def test_hatalar(tmp_path, pyannote):
    r = _rttm(tmp_path, "ref", REF)
    with pytest.raises(FileNotFoundError):
        degerlendirme.degerlendir(r, str(tmp_path / "yok.rttm"))
    bozuk = tmp_path / "bozuk.rttm"
    bozuk.write_text("SPEAKER x 1 abc\n", encoding="utf-8")
    with pytest.raises(ValueError):
        degerlendirme.degerlendir(r, str(bozuk))
