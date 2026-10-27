"""src/cikti.py testleri (yalnızca standart kütüphane + numpy, sentetik veri)."""
import numpy as np
import pytest

from src import cikti


def _p(*pencereler):
    return [tuple(map(float, p)) for p in pencereler]


# --- segmentleri_birlestir -------------------------------------------------

def test_bos_girdi():
    assert cikti.segmentleri_birlestir([], np.array([], dtype=int)) == []


def test_uzunluk_uyusmazligi():
    with pytest.raises(ValueError):
        cikti.segmentleri_birlestir(_p((0, 1.5)), np.array([0, 1]))


def test_ayni_etiketli_ortusen_pencereler_birlesir():
    p = _p((0, 1.5), (0.75, 2.25), (1.5, 3.0))
    assert cikti.segmentleri_birlestir(p, np.array([4, 4, 4])) == [(0.0, 3.0, "spk00")]


def test_farkli_etikette_sinir_ortusmenin_ortasi():
    # A: 0-3.0 (4 pencere), B: 2.25-6.0; örtüşme 2.25-3.0 -> sınır 2.625
    p = _p((0, 1.5), (0.75, 2.25), (1.5, 3.0), (2.25, 3.75), (3.0, 4.5), (3.75, 5.25), (4.5, 6.0))
    seg = cikti.segmentleri_birlestir(p, np.array([0, 0, 0, 1, 1, 1, 1]))
    assert [s[:2] for s in seg] == [(0.0, 2.625), (2.625, 6.0)]
    # B daha uzun konuşuyor -> spk00
    assert [s[2] for s in seg] == ["spk01", "spk00"]


def test_ada_yumusatma_kisa_ada_komsuya_katilir():
    # A 0-4, B 4-4.5 (0,5 sn < 1,0), A 4.5-8 : bitişik, örtüşmesiz pencereler
    p = _p((0, 2), (2, 4), (4, 4.5), (4.5, 6.5), (6.5, 8))
    seg = cikti.segmentleri_birlestir(p, np.array([0, 0, 1, 0, 0]))
    assert seg == [(0.0, 8.0, "spk00")]


def test_ada_esikten_uzunsa_korunur():
    p = _p((0, 2), (2, 4), (4, 6), (6, 8))
    seg = cikti.segmentleri_birlestir(p, np.array([0, 1, 1, 0]), ada_esik_sn=1.0)
    # B koşusu 4 sn >= 1 sn: korunur. A ve B 4'er sn: eşitlikte ilk görünen (A) spk00.
    assert seg == [(0.0, 2.0, "spk00"), (2.0, 6.0, "spk01"), (6.0, 8.0, "spk00")]


def test_ada_yumusatmada_bosluk_dikkate_alinmaz():
    # A 0-3, (10 sn sessizlik), B 13-13.5, (sessizlik), A 20-23: B yine A olur
    p = _p((0, 3), (13, 13.5), (20, 23))
    seg = cikti.segmentleri_birlestir(p, np.array([0, 1, 0]))
    assert {s[2] for s in seg} == {"spk00"}
    assert seg == [(0.0, 3.0, "spk00"), (13.0, 13.5, "spk00"), (20.0, 23.0, "spk00")]


def test_kenardaki_kisa_kosu_yumusatilmaz():
    p = _p((0, 0.5), (0.5, 5))
    seg = cikti.segmentleri_birlestir(p, np.array([1, 0]))
    assert len(seg) == 2


def test_bosluk_birlestirme_esigi():
    # aynı etiket, 0,3 sn boşluk -> birleşir; 0,31 sn -> ayrı kalır
    seg = cikti.segmentleri_birlestir(_p((0, 2), (2.3, 4)), np.array([0, 0]))
    assert seg == [(0.0, 4.0, "spk00")]
    seg = cikti.segmentleri_birlestir(_p((0, 2), (2.31, 4)), np.array([0, 0]))
    assert seg == [(0.0, 2.0, "spk00"), (2.31, 4.0, "spk00")]


def test_etiketler_sureye_gore_siralanir():
    p = _p((0, 2), (2, 3.5), (3.5, 10), (10, 12))
    seg = cikti.segmentleri_birlestir(p, np.array([5, 9, 7, 9]), ada_esik_sn=0)
    sure = {}
    for a, b, e in seg:
        sure[e] = sure.get(e, 0) + b - a
    # 7 (6,5 sn) -> spk00, 9 (3,5 sn) -> spk01, 5 (2 sn) -> spk02
    assert sure["spk00"] == pytest.approx(6.5)
    assert sure["spk01"] == pytest.approx(3.5)
    assert sure["spk02"] == pytest.approx(2.0)


# --- degisim_noktalari ---------------------------------------------------

def test_degisim_noktalari_temel():
    assert cikti.degisim_noktalari([]) == []
    assert cikti.degisim_noktalari([(0, 1, "a")]) == []
    seg = [(0, 1, "a"), (1, 2, "b"), (2.5, 3, "b"), (3, 4, "a")]
    assert cikti.degisim_noktalari(seg) == [1.0, 3.0]


def test_degisim_noktalari_ortusme_atilir():
    # A 0-10, B 8-15: örtüşme 8-10 atılır -> değişim B'nin tek başına başladığı 10'da
    assert cikti.degisim_noktalari([(0, 10, "A"), (8, 15, "B")]) == [10.0]


def test_degisim_noktalari_kayan_nokta_gurultusu():
    # 0.1 + 0.2 gibi toplamlar sahte kısa aralık üretmemeli
    seg = [(0.0, 0.1 + 0.2, "a"), (0.3, 1.0, "a"), (1.0, 2.0, "b")]
    assert cikti.degisim_noktalari(seg) == [1.0]


def test_tek_konusmaci_segmentleri_ms():
    nov = cikti.tek_konusmaci_segmentleri([(0, 1000, "A"), (500, 1500, "B"), (2000, 3000, "B")])
    assert nov == [(0, 500, "A"), (1000, 3000, "B")]


# --- RTTM yaz / oku ------------------------------------------------------

def test_rttm_yaz_oku_dongusu(tmp_path):
    seg = [(0.0, 1.234, "spk00"), (1.234, 5.5, "spk01"), (6.0, 7.0005, "spk00")]
    yol = tmp_path / "x.rttm"
    cikti.rttm_yaz(seg, "kayit1", str(yol))
    satirlar = yol.read_text(encoding="utf-8").splitlines()
    assert satirlar[0] == "SPEAKER kayit1 1 0.000 1.234 <NA> <NA> spk00 <NA> <NA>"
    assert all(len(s.split()) == 10 for s in satirlar)
    geri = cikti.rttm_oku(str(yol))
    assert [e for _, _, e in geri] == ["spk00", "spk01", "spk00"]
    for (a, b, _), (a2, b2, _) in zip(seg, geri):
        assert a2 == pytest.approx(a, abs=1e-3) and b2 == pytest.approx(b, abs=1e-3)


def test_rttm_bos(tmp_path):
    yol = tmp_path / "bos.rttm"
    cikti.rttm_yaz([], "bos", str(yol))
    assert yol.read_text(encoding="utf-8") == ""
    assert cikti.rttm_oku(str(yol)) == []


def test_rttm_oku_atlananlar_ve_hatalar(tmp_path):
    yol = tmp_path / "k.rttm"
    yol.write_text(
        "\n;; yorum\n# yorum\nSPKR-INFO f 1 <NA> <NA> <NA> unknown A <NA> <NA>\n"
        "SPEAKER f 1 1.0 0.000 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER f 1 2.0 1.500 <NA> <NA> B <NA> <NA>\n",
        encoding="utf-8",
    )
    assert cikti.rttm_oku(str(yol)) == [(2.0, 3.5, "B")]
    for bozuk in ("SPEAKER f 1 2.0 1.5\n", "SPEAKER f 1 x 1.5 <NA> <NA> A <NA> <NA>\n",
                  "SPEAKER f 1 2.0 -1 <NA> <NA> A <NA> <NA>\n"):
        yol.write_text(bozuk, encoding="utf-8")
        with pytest.raises(ValueError):
            cikti.rttm_oku(str(yol))
    with pytest.raises(FileNotFoundError):
        cikti.rttm_oku(str(tmp_path / "yok.rttm"))


def test_tablo_yazdir():
    metin = cikti.tablo_yazdir([(0.0, 61.5, "spk00"), (61.5, 81.5, "spk01")])
    assert "Konuşmacı" in metin and "Başlangıç" in metin
    assert "01:01.500" in metin
    assert "%75.5" in metin  # 61,5 / 81,5
    assert "Konuşma bulunamadı" in cikti.tablo_yazdir([])
