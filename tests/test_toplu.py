"""src/toplu.py testleri: sentetik önbellek + elle yazılmış RTTM (zip açılmaz, ses yok)."""
import csv
import io
import math
import zipfile

import numpy as np
import pytest

from src import toplu
from sentetik_veri import kayit_uret, ortam_kur

pytest.importorskip("pyannote.metrics")
pytest.importorskip("sklearn")

SESSIZ = lambda *a: None  # noqa: E731


@pytest.fixture
def ortam(tmp_path):
    kayitlar = {f"kayit{i}": kayit_uret(i) for i in range(6)}
    rttm_dizin, onbellek = ortam_kur(tmp_path, kayitlar)
    return {"kok": tmp_path, "rttm": rttm_dizin, "onbellek": onbellek, "kayitlar": sorted(kayitlar),
            "zip": str(tmp_path / "olmayan.zip"), "cikti": str(tmp_path / "sonuclar")}


def _csv(yol):
    with open(yol, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_onbellek_anahtarlari_parametreleri_icerir():
    a = toplu.onbellek_anahtarlari()
    assert a["vad"] == "v250_s300"
    assert a["mfcc"].startswith("p1.5_k0.75_v250_s300") and "mfcc13c1" in a["mfcc"]
    assert a["ecapa"].endswith("ecapa-voxceleb")


@pytest.mark.parametrize("deney", ["E1", "E2", "E3", "E4"])
def test_deney_csv_ozet_ve_devam(ortam, deney):
    o = ortam
    ozet = toplu.deney_calistir(o["zip"], o["rttm"], o["kayitlar"], deney, o["cikti"], o["onbellek"],
                                esik=0.5, log=SESSIZ)
    satirlar = _csv(f"{o['cikti']}/{deney}.csv")
    assert list(satirlar[0])[:5] == ["deney", "kayit", "der", "jer", "f1"]
    assert list(satirlar[0]) == toplu.CSV_SUTUNLARI
    assert sorted(r["kayit"] for r in satirlar) == o["kayitlar"]
    assert ozet["kayit_sayisi"] == 6 and ozet["atlanan"] == 0 and ozet["hatali"] == 0
    assert ozet["der"] < 0.3  # sentetik veri kolay
    # sure_sn önbellekteki hesaplama süresini içerir: vad 0,5 + (ecapa 1,0 | mfcc 0,2)
    beklenen = 1.5 if deney in ("E3", "E4") else 0.7
    assert all(float(r["sure_sn"]) >= beklenen for r in satirlar)

    # Yeniden çalıştırma: kayıtlar atlanır, CSV ve özet satırı tekrarlanmaz
    toplu.deney_calistir(o["zip"], o["rttm"], o["kayitlar"], deney, o["cikti"], o["onbellek"],
                         esik=0.5, log=SESSIZ)
    assert len(_csv(f"{o['cikti']}/{deney}.csv")) == 6
    assert [r["deney"] for r in _csv(f"{o['cikti']}/ozet.csv")] == [deney]


def test_e5_model_egit_ve_calistir(ortam):
    o = ortam
    model = str(o["kok"] / "modeller" / "degisim_modeli.joblib")
    paket = toplu.model_egit(o["zip"], o["rttm"], o["kayitlar"], o["onbellek"], model, log=SESSIZ)
    assert paket["egitim_kayit_sayisi"] == 6
    assert len(paket["mfcc_olcek"]["std"]) == 26 and paket["ozellik_surumu"] == "v2"
    ozet = toplu.deney_calistir(o["zip"], o["rttm"], o["kayitlar"], "E5", o["cikti"], o["onbellek"],
                                esik=0.5, model_yolu=model, log=SESSIZ)
    assert ozet["kayit_sayisi"] == 6 and ozet["der"] < 0.3
    with pytest.raises(ValueError):
        toplu.deney_calistir(o["zip"], o["rttm"], o["kayitlar"], "E5", o["cikti"], o["onbellek"], log=SESSIZ)


def test_esik_tara_ve_json(ortam):
    o = ortam
    sonuc = toplu.esik_tara(o["zip"], o["rttm"], o["kayitlar"], o["onbellek"],
                            izgara=(0.2, 0.5, 0.8, 1.2, 1.9), log=SESSIZ)
    assert sonuc["mfcc"] in sonuc["izgara"]["mfcc"] and sonuc["ecapa"] in sonuc["izgara"]["ecapa"]
    assert sonuc["genisletme"] == {"mfcc": 0, "ecapa": 0}  # sabit ızgara verilince genişletilmez
    assert len(sonuc["mfcc_olcek"]["ortalama"]) == 26
    assert len(sonuc["tarama"]) == 10 and sonuc["hatali"] == {"mfcc": 0, "ecapa": 0}
    en_iyi = min(r["der"] for r in sonuc["tarama"] if r["yontem"] == "ecapa")
    assert en_iyi < 0.3
    # 1.9: her şey tek küme -> DER yüksek; seçilmemeli
    assert sonuc["ecapa"] != 1.9
    sonuc["tarama"][0]["jer_ort"] = math.nan
    yol = str(o["kok"] / "modeller" / "esikler.json")
    toplu.esikleri_yaz(yol, sonuc)
    geri = toplu.esikleri_oku(yol)
    assert geri["ecapa"] == sonuc["ecapa"] and geri["tarama"][0]["jer_ort"] is None


def test_hatali_kayit_atlanir_ve_gunluge_yazilir(ortam):
    o = ortam
    kayitlar = o["kayitlar"] + ["onbellekte_yok"]
    (o["kok"] / "rttm" / "onbellekte_yok.rttm").write_text("", encoding="utf-8")
    ozet = toplu.deney_calistir(o["zip"], o["rttm"], kayitlar, "E4", o["cikti"], o["onbellek"], log=SESSIZ)
    assert ozet["hatali"] == 1 and ozet["atlanan"] == 1 and ozet["kayit_sayisi"] == 6
    gunluk = (o["kok"] / "sonuclar" / toplu.HATA_GUNLUGU).read_text(encoding="utf-8")
    assert "onbellekte_yok" in gunluk and "zip bulunamadı" in gunluk


def test_ozet_hesapla():
    satirlar = [
        {"yanlis_alarm_sn": "1", "kacirilan_sn": "2", "karisiklik_sn": "3", "referans_sn": "30",
         "jer": "0.2", "tp": "3", "fp": "1", "fn": "0", "sure_sn": "5", "ses_sn": "60"},
        {"yanlis_alarm_sn": "4", "kacirilan_sn": "0", "karisiklik_sn": "0", "referans_sn": "0",
         "jer": "nan", "tp": "0", "fp": "2", "fn": "0", "sure_sn": "1", "ses_sn": "10"},
    ]
    o = toplu.ozet_hesapla(satirlar)
    assert o["der"] == pytest.approx(10 / 30) and o["jer_ort"] == pytest.approx(0.2)
    assert o["mikro_f1"] == pytest.approx(2 * 0.5 * 1 / 1.5)  # P = 3/6, R = 3/3
    assert o["toplam_sure_sn"] == 6 and o["toplam_ses_sn"] == 70 and o["kayit_sayisi"] == 2


def test_cli_esikler_dosyasi(ortam, capsys):
    o = ortam
    liste = o["kok"] / "liste.txt"
    liste.write_text("\n".join(o["kayitlar"]) + "\n", encoding="utf-8")
    esikler = o["kok"] / "esikler.json"
    toplu.esikleri_yaz(str(esikler), {"mfcc": 0.3, "ecapa": 0.45})
    kod = toplu.main(["--deney", "E4", "--zip", o["zip"], "--rttm-dizin", o["rttm"], "--liste", str(liste),
                      "--cikti-dizin", o["cikti"], "--onbellek", o["onbellek"], "--esikler", str(esikler)])
    assert kod == 0
    cikis = capsys.readouterr().out
    assert "eşik 0.45" in cikis and "E4" in cikis
    # --esik önceliklidir; E1 eşik kullanmaz
    toplu.main(["--deney", "E2", "--zip", o["zip"], "--rttm-dizin", o["rttm"], "--liste", str(liste),
                "--cikti-dizin", o["cikti"], "--onbellek", o["onbellek"], "--esikler", str(esikler),
                "--esik", "0.6"])
    assert "eşik 0.6" in capsys.readouterr().out
    assert toplu.main(["--deney", "E1", "--zip", o["zip"], "--rttm-dizin", o["rttm"], "--liste",
                       str(o["kok"] / "yok.txt"), "--cikti-dizin", o["cikti"], "--onbellek", o["onbellek"]]) == 1


def test_zipten_okuma(tmp_path):
    # Colab: soundfile gerekir. Zip içindeki WAV belleğe okunur, ön işlemden geçer.
    sf = pytest.importorskip("soundfile")
    t = np.arange(8000 * 2) / 8000
    tampon = io.BytesIO()
    sf.write(tampon, (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 8000, format="WAV")
    zip_yol = tmp_path / "ornek.zip"
    with zipfile.ZipFile(zip_yol, "w") as z:
        z.writestr("audio/", "")
        z.writestr("audio/kayitx.wav", tampon.getvalue())
    with toplu.SesKaynagi(str(zip_yol)) as k:
        ses, sr = k.oku("kayitx")
        assert sr == 16000 and ses.shape == (32000,) and ses.dtype == np.float32
        with pytest.raises(ValueError):
            k.oku("olmayan")


def test_yuzdelik_izgarasi():
    r = np.random.default_rng(0)
    h = r.gamma(2.0, 0.1, size=5000)
    g = toplu.yuzdelik_izgarasi(h)
    assert g == sorted(g) and 30 <= len(g) <= toplu.YUZDELIK_SAYISI
    assert np.percentile(h, 1) - 1e-5 <= g[0] and g[-1] <= np.percentile(h, 99) + 1e-5  # 5 ondalığa yuvarlı
    adim = np.diff(100 - np.geomspace(99, 1, toplu.YUZDELIK_SAYISI))
    assert adim[-1] < adim[0]  # yüzdelikler üst uca doğru sıklaşır
    assert toplu.yuzdelik_izgarasi([]) == []


def test_genislet():
    h = np.linspace(0.1, 1.0, 1000)
    alt = toplu._genislet([0.2, 0.5, 0.8], h, "alt", 0)
    ust = toplu._genislet([0.2, 0.5, 0.8], h, "ust", 0)
    assert alt and all(v < 0.2 for v in alt) and ust and all(v > 0.8 for v in ust)
    assert max(ust) == pytest.approx(1.01)  # her şeyin birleştiği değer de denenir
    assert all(v > 0.8 for v in toplu._genislet([0.2, 0.5, 0.8], h, "ust", 1))


def test_esik_tara_veriden_izgara(ortam):
    o = ortam
    sonuc = toplu.esik_tara(o["zip"], o["rttm"], o["kayitlar"], o["onbellek"], log=SESSIZ)
    for y in ("mfcc", "ecapa"):
        g = sonuc["izgara"][y]
        assert g == sorted(g) and len(g) >= 20 and sonuc[y] in g
        en_iyi = min(r["der"] for r in sonuc["tarama"] if r["yontem"] == y)
        assert en_iyi < 0.3, y
    # kenara düşen her durum uyarı olarak yazılır
    for y, n in sonuc["genisletme"].items():
        assert (n > 0) == any(u.startswith(f"{y}:") and "genişletildi" in u for u in sonuc["uyarilar"])


def test_e2_mfcc_olcegi_uygulanir(ortam):
    o = ortam
    from src import oznitelik
    olcek = oznitelik.olcek_hesapla([kayit_uret(i)["X_mfcc"] for i in range(6)])
    loglar = []
    ozet = toplu.deney_calistir(o["zip"], o["rttm"], o["kayitlar"], "E2", o["cikti"], o["onbellek"],
                                esik=0.5, mfcc_olcek=olcek, log=loglar.append)
    assert ozet["kayit_sayisi"] == 6 and any("MFCC genel ölçekli" in s for s in loglar)
