"""src/gorsel.py testleri: data/sample RTTM'leri ve sentetik CSV; çıktılar tmp_path'e."""
import os

import pytest

from src import cikti, gorsel

pytest.importorskip("matplotlib")

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORNEK = os.path.join(KOK, "data", "sample")
PNG_IMZA = b"\x89PNG\r\n\x1a\n"


def _png_mi(yol) -> bool:
    with open(yol, "rb") as f:
        return f.read(8) == PNG_IMZA


# --- zaman çizelgesi ------------------------------------------------------

def test_zaman_cizelgesi(tmp_path):
    ref = os.path.join(ORNEK, "migzj_066.5-136.5.rttm")  # 4 konuşmacı, örtüşmeli
    # Sistem çıktısı yerine farklı etiketli bir RTTM: aynı segmentler, iki etikete indirgenmiş
    hyp = tmp_path / "hyp.rttm"
    segler = cikti.rttm_oku(ref)
    cikti.rttm_yaz([(a, b, "spk00" if i % 2 else "spk01") for i, (a, b, _) in enumerate(segler)], "migzj", str(hyp))
    png = tmp_path / "cizim" / "migzj.png"
    assert gorsel.zaman_cizelgesi(ref, str(hyp), str(png), baslik="migzj") == str(png)
    assert _png_mi(png) and png.stat().st_size > 5000


def test_zaman_cizelgesi_bos_sistem(tmp_path):
    ref = os.path.join(ORNEK, "edge", "kisa_1sn.rttm")
    bos = os.path.join(ORNEK, "edge", "sessiz_5sn.rttm")  # boş RTTM
    png = tmp_path / "kisa.png"
    gorsel.zaman_cizelgesi(ref, bos, str(png))
    gorsel.zaman_cizelgesi(bos, bos, str(tmp_path / "ikisi_bos.png"))
    assert _png_mi(png) and _png_mi(tmp_path / "ikisi_bos.png")


def test_zaman_cizelgesi_hatalar(tmp_path):
    with pytest.raises(FileNotFoundError):
        gorsel.zaman_cizelgesi(str(tmp_path / "yok.rttm"), os.path.join(ORNEK, "abjxc_tam.rttm"),
                               str(tmp_path / "x.png"))


def test_pyplot_kullanilmaz(tmp_path):
    import sys

    gorsel.zaman_cizelgesi(os.path.join(ORNEK, "abjxc_tam.rttm"), os.path.join(ORNEK, "abjxc_tam.rttm"),
                           str(tmp_path / "a.png"))
    plt = sys.modules.get("matplotlib.pyplot")
    assert plt is None or plt.get_fignums() == []  # açık pencere/şekil bırakmaz
