"""data/tools/rttm_istatistik.py ile src/cikti.py aynı değişim tanımını kullanıyor mu?"""
import glob
import importlib.util
import os
import subprocess
import sys

import pytest

from src import cikti

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARAC = os.path.join(KOK, "data", "tools", "rttm_istatistik.py")
ORNEKLER = sorted(glob.glob(os.path.join(KOK, "data", "sample", "**", "*.rttm"), recursive=True))

# Ortak fonksiyona taşımadan önce aracın verdiği değişim sayıları (regresyon tabanı).
BEKLENEN_DEGISIM = {
    "abjxc_tam": 0,
    "ikgcq_028.0-118.0": 8,
    "luvfz_113.0-203.0": 23,
    "migzj_066.5-136.5": 12,
    "sldwj_000.0-090.0": 10,
    "ikgcq_028.0-118.0_yapay_gurultu_snr10dB": 8,
    "ikgcq_028.0-118.0_yapay_gurultu_snr5dB": 8,
    "kisa_1sn": 0,
}


@pytest.fixture(scope="module")
def arac():
    spec = importlib.util.spec_from_file_location("rttm_istatistik", ARAC)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def test_ornek_dosyalar_var():
    assert len(ORNEKLER) >= 9


def test_degisim_sayilari_degismedi(arac):
    bulunan = {}
    for yol in ORNEKLER:
        for kayit, segler in arac.rttm_oku(yol).items():
            if segler:
                bulunan[kayit] = arac.istatistik(segler)["degisim"]
    assert bulunan == BEKLENEN_DEGISIM


@pytest.mark.parametrize("yol", ORNEKLER, ids=os.path.basename)
def test_arac_ve_cikti_ayni_noktalar(arac, yol):
    arac_ms = [t for segler in arac.rttm_oku(yol).values() for t, _, _ in arac.degisim_noktalari(segler)]
    cikti_ms = [round(t * 1000) for t in cikti.degisim_noktalari(cikti.rttm_oku(yol))]
    assert cikti_ms == arac_ms


def test_arac_numpy_olmadan_calisir():
    kod = (
        "import sys, runpy; sys.modules['numpy'] = None\n"
        f"sys.argv = ['rttm_istatistik.py', {os.path.join(KOK, 'data', 'sample')!r}]\n"
        f"runpy.run_path({ARAC!r}, run_name='__main__')\n"
    )
    sonuc = subprocess.run([sys.executable, "-c", kod], capture_output=True, text=True, encoding="utf-8")
    assert sonuc.returncode == 0, sonuc.stderr
    assert "luvfz_113.0-203.0" in sonuc.stdout
