import pytest
from src.vad_ornek_olcum import hata


def test_ortusen_referans_iki_kez_sayilmaz():
    s = hata([(0, 1, 'a'), (.5, 1.5, 'b')], [(0, .5)], 2)
    assert s == pytest.approx({'referans_sn': 1.5, 'kacirilan_sn': 1., 'yanlis_alarm_sn': 0.})


def test_sessizlikte_yanlis_alarm():
    assert hata([], [(0, 1)], 2) == pytest.approx(
        {'referans_sn': 0., 'kacirilan_sn': 0., 'yanlis_alarm_sn': 1.})
