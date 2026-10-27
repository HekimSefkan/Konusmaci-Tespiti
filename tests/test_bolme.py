"""src/bolme.py testleri (sentetik RTTM'ler, yalnızca standart kütüphane)."""
import os

import pytest

from src import bolme


def _rttm_klasoru(tmp_path, sayilar):
    for kayit, n in sayilar.items():
        satirlar = "".join(
            f"SPEAKER {kayit} 1 {i * 2:.3f} 1.500 <NA> <NA> spk{i:02d} <NA> <NA>\n" for i in range(n))
        (tmp_path / f"{kayit}.rttm").write_text(satirlar, encoding="utf-8")
    return str(tmp_path)


def _sentetik_sayilar():
    # 1: 7 kayıt, 2: 31, 3-4: 60, 5+: 118 -> toplam 216 (VoxConverse dev boyutu)
    sayilar = {}
    for g, (adet, n) in {"a": (7, 1), "b": (31, 2), "c": (60, 3), "d": (118, 6)}.items():
        for i in range(adet):
            sayilar[f"{g}{i:03d}"] = n
    return sayilar


def test_grup():
    assert [bolme.grup(n) for n in (0, 1, 2, 3, 4, 5, 12)] == ["1", "1", "2", "3-4", "3-4", "5+", "5+"]


def test_konusmaci_sayilari(tmp_path):
    klasor = _rttm_klasoru(tmp_path, {"x": 1, "y": 4})
    (tmp_path / "notlar.txt").write_text("rttm değil", encoding="utf-8")
    assert bolme.konusmaci_sayilari(klasor) == {"x": 1, "y": 4}
    with pytest.raises(FileNotFoundError):
        bolme.konusmaci_sayilari(str(tmp_path / "yok"))


def test_tabakali_ve_dengeli():
    sayilar = _sentetik_sayilar()
    egitim, deg = bolme.bol(sayilar)
    assert len(egitim) + len(deg) == 216 and not set(egitim) & set(deg)
    assert abs(len(egitim) - len(deg)) <= 1
    for g in bolme.GRUPLAR:
        uyeler = [k for k, n in sayilar.items() if bolme.grup(n) == g]
        n_e = sum(k in set(egitim) for k in uyeler)
        assert abs(n_e - (len(uyeler) - n_e)) <= 1, g
    assert egitim == sorted(egitim) and deg == sorted(deg)


def test_tekrarlanabilir_ve_seed_etkili():
    sayilar = _sentetik_sayilar()
    assert bolme.bol(sayilar) == bolme.bol(dict(reversed(list(sayilar.items()))))
    assert bolme.bol(sayilar, seed=68) != bolme.bol(sayilar, seed=1)


def test_liste_yaz_oku_ve_cli(tmp_path):
    rttm = tmp_path / "rttm"
    rttm.mkdir()
    klasor = _rttm_klasoru(rttm, {f"k{i}": 1 + i % 6 for i in range(20)})
    cikti = tmp_path / "bolme"
    assert bolme.main(["--rttm-dizin", klasor, "--cikti-dizin", str(cikti)]) == 0
    egitim = bolme.liste_oku(str(cikti / "egitim.txt"))
    deg = bolme.liste_oku(str(cikti / "degerlendirme.txt"))
    assert (egitim, deg) == bolme.bol(bolme.konusmaci_sayilari(klasor))
    (cikti / "elle.txt").write_text("# yorum\n\nk1\n k2 \n", encoding="utf-8")
    assert bolme.liste_oku(str(cikti / "elle.txt")) == ["k1", "k2"]
    assert bolme.main(["--rttm-dizin", str(tmp_path / "yok"), "--cikti-dizin", str(cikti)]) == 1


# --- repodaki liste dosyaları (data/bolme/) --------------------------------

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTE_DIZIN = os.path.join(KOK, "data", "bolme")


def test_repo_listeleri():
    egitim = bolme.liste_oku(os.path.join(LISTE_DIZIN, "egitim.txt"))
    deg = bolme.liste_oku(os.path.join(LISTE_DIZIN, "degerlendirme.txt"))
    assert len(egitim) == 108 and len(deg) == 108
    assert egitim == sorted(set(egitim)) and deg == sorted(set(deg))
    assert not set(egitim) & set(deg)
    # örnek kayıtların (data/sample) yarıları
    assert {"abjxc", "ikgcq", "migzj"} <= set(egitim) and {"luvfz", "sldwj"} <= set(deg)


def test_betik_repo_listelerini_uretir():
    """VoxConverse dev RTTM'leri yalnızca Colab'da var: KT_VOX_RTTM yoksa atlanır."""
    vox = os.environ.get("KT_VOX_RTTM")
    if not vox or not os.path.isdir(vox):
        pytest.skip("KT_VOX_RTTM (VoxConverse dev RTTM klasörü) yok; karşılaştırma Colab'da yapılır")
    egitim, deg = bolme.bol(bolme.konusmaci_sayilari(vox))
    assert egitim == bolme.liste_oku(os.path.join(LISTE_DIZIN, "egitim.txt"))
    assert deg == bolme.liste_oku(os.path.join(LISTE_DIZIN, "degerlendirme.txt"))
