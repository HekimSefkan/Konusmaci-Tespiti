"""src/varsayilanlar.json ve src/varsayilanlar.py testleri (yalnızca standart kütüphane + numpy)."""
import json

import pytest

from src import kumeleme, varsayilanlar


def test_dosya_bicimi():
    with open(varsayilanlar.DOSYA, encoding="utf-8") as f:
        veri = json.load(f)
    assert {"kaynak", "tarih", "esik", "mfcc_olcek"} <= set(veri)
    assert set(veri["esik"]) == {"mfcc", "ecapa"}


def test_ayarlanmis_esikler():
    # Hafta 4, VoxConverse dev eğitim yarısı eşik taraması
    assert varsayilanlar.esik("mfcc") == pytest.approx(0.77122)
    assert varsayilanlar.esik("ecapa") == pytest.approx(0.79346)
    assert kumeleme.VARSAYILAN_ESIK == varsayilanlar.esik("ecapa")
    with pytest.raises(ValueError):
        varsayilanlar.esik("xvector")


def test_mfcc_olcegi():
    # Hafta 4 eğitim yarısı: 42520 MFCC penceresinden (defter 10. bölüm "REPO SABİTLERİ")
    olcek = varsayilanlar.mfcc_olcek()
    assert olcek is not None
    assert len(olcek["ortalama"]) == 26 and len(olcek["std"]) == 26
    assert all(isinstance(v, float) for v in olcek["ortalama"] + olcek["std"])
    assert all(s > 0 for s in olcek["std"])
    assert olcek["pencere_sayisi"] == 42520
    assert olcek["ortalama"][0] == pytest.approx(107.267103) and olcek["std"][-1] == pytest.approx(1.765113)


def test_kaynak_ve_tarih():
    veri = varsayilanlar.yukle()
    assert veri["tarih"] == "2026-10-27T13:58:49"
    assert "egitim.txt" in veri["kaynak"] and "not" not in veri


def test_bozuk_dosya_reddedilir(tmp_path, monkeypatch):
    yol = tmp_path / "v.json"
    yol.write_text(json.dumps({"esik": {"mfcc": 0.7}, "mfcc_olcek": None}), encoding="utf-8")
    monkeypatch.setattr(varsayilanlar, "DOSYA", str(yol))
    monkeypatch.setattr(varsayilanlar, "_veri", None)
    with pytest.raises(ValueError, match="ecapa"):
        varsayilanlar.yukle()
    yol.write_text(json.dumps({"esik": {"mfcc": 0.7, "ecapa": 0.8},
                               "mfcc_olcek": {"ortalama": [0] * 26, "std": [1] * 25}}), encoding="utf-8")
    monkeypatch.setattr(varsayilanlar, "_veri", None)
    with pytest.raises(ValueError, match="mfcc_olcek"):
        varsayilanlar.yukle()
