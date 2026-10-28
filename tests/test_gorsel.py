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


# --- sonuç grafiği --------------------------------------------------------

def _csv_yaz(yol, basliklar, satirlar):
    import csv

    with open(yol, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(basliklar)
        w.writerows(satirlar)
    return str(yol)


def _deney_satiri(deney, kayit, der, ref, hata, tp, fp, fn, jer="0.2"):
    # CSV_SUTUNLARI sırası: deney,kayit,der,jer,f1,precision,recall,tp,fp,fn,yanlis_alarm_sn,kacirilan_sn,
    # karisiklik_sn,referans_sn,ses_sn,sure_sn
    return [deney, kayit, der, jer, "0.5", "0.5", "0.5", tp, fp, fn, hata, 0, 0, ref, 60, 1]


def test_sonuc_grafigi_ozet(tmp_path):
    ozet = _csv_yaz(tmp_path / "ozet.csv",
                    ["deney", "kayit_sayisi", "der", "jer_ort", "mikro_f1", "toplam_sure_sn", "toplam_ses_sn", "atlanan"],
                    [["E1", 108, 0.281, 0.425, 0.320, 1, 1, 0], ["E4", 108, 0.061, 0.105, 0.638, 1, 1, 0]])
    png = tmp_path / "ozet.png"
    assert gorsel.sonuc_grafigi(ozet, str(png)) == str(png) and _png_mi(png)
    assert [v[0] for v in gorsel._sonuclari_oku(ozet)] == ["E1", "E4"]
    assert gorsel._sonuclari_oku(ozet)[1][1:] == pytest.approx((0.061, 0.105, 0.638))


def test_sonuc_grafigi_deney_csv_listesi(tmp_path):
    from src import toplu

    e2 = _csv_yaz(tmp_path / "E2.csv", toplu.CSV_SUTUNLARI,
                  [_deney_satiri("E2", "a", 0.1, 30, 3, 3, 1, 0), _deney_satiri("E2", "b", 0.5, 10, 5, 0, 2, 0, jer="nan")])
    e4 = _csv_yaz(tmp_path / "E4.csv", toplu.CSV_SUTUNLARI, [_deney_satiri("E4", "a", 0.05, 30, 1.5, 4, 0, 0)])
    veriler = gorsel._sonuclari_oku([e4, e2])
    assert [v[0] for v in veriler] == ["E2", "E4"]
    assert veriler[0][1] == pytest.approx(8 / 40)       # DER toplam saniyelerden
    assert veriler[0][2] == pytest.approx(0.2)          # JER: NaN'sız ortalama
    assert veriler[0][3] == pytest.approx(2 * 0.5 / 1.5)  # mikro F1: P = 3/6, R = 1
    assert _png_mi(gorsel.sonuc_grafigi([e4, e2], str(tmp_path / "deneyler.png")))
    assert _png_mi(gorsel.sonuc_grafigi(e4, str(tmp_path / "tek.png")))


def test_sonuc_grafigi_hatalar(tmp_path):
    bos = tmp_path / "bos.csv"
    bos.write_text("deney,der\n", encoding="utf-8")
    with pytest.raises(ValueError):
        gorsel.sonuc_grafigi(str(bos), str(tmp_path / "x.png"))
    tuhaf = _csv_yaz(tmp_path / "tuhaf.csv", ["a", "b"], [[1, 2]])
    with pytest.raises(ValueError, match="ozet.csv veya deney"):
        gorsel.sonuc_grafigi(tuhaf, str(tmp_path / "x.png"))
    with pytest.raises(ValueError):
        gorsel.sonuc_grafigi([], str(tmp_path / "x.png"))
    with pytest.raises(FileNotFoundError):
        gorsel.sonuc_grafigi(str(tmp_path / "yok.csv"), str(tmp_path / "x.png"))
