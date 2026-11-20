"""src/test_seti.py: test listesi, özet tabloları, hız ölçümü ve kısa özet (sahte veriyle; ses ve model yok)."""
import json
import math
import os
import zipfile

import pytest

from src import cikti, test_seti, toplu


def _rttm(klasor, kayit, segler):
    os.makedirs(klasor, exist_ok=True)
    cikti.rttm_yaz(segler, kayit, os.path.join(klasor, f"{kayit}.rttm"))


def _satir(kayit, der, f1, tp, fn, ref=100.0, jer=0.2):
    hata = der * ref
    return {"deney": "E4", "kayit": kayit, "der": str(der), "jer": str(jer), "f1": str(f1), "precision": "1",
            "recall": "1", "tp": str(tp), "fp": "0", "fn": str(fn), "yanlis_alarm_sn": str(hata),
            "kacirilan_sn": "0", "karisiklik_sn": "0", "referans_sn": str(ref), "ses_sn": "120", "sure_sn": "3"}


def test_zip_ve_rttm_eslestirme(tmp_path):
    zip_yol = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_yol, "w") as z:  # içerik ses değil; yalnızca adlar okunur
        for ad in ("audio/aaa.wav", "audio/bbb.wav", "audio/yalniz_ses.wav", "audio/notlar.txt"):
            z.writestr(ad, b"x")
    rttm = tmp_path / "test_rttm"
    for k in ("aaa", "bbb", "yalniz_rttm"):
        _rttm(str(rttm), k, [(0.0, 1.0, "spk00")])
    assert test_seti.zip_kayitlari(str(zip_yol)) == ["aaa", "bbb", "yalniz_ses"]
    liste = tmp_path / "liste.txt"
    e = test_seti.test_listesi_yaz(str(zip_yol), str(rttm), str(liste))
    assert e == {"ortak": ["aaa", "bbb"], "yalniz_ses": ["yalniz_ses"], "yalniz_rttm": ["yalniz_rttm"]}
    assert toplu.listeleri_oku([str(liste)]) == ["aaa", "bbb"]
    with pytest.raises(FileNotFoundError):
        test_seti.rttm_kayitlari(str(tmp_path / "yok"))


def test_karsilastirma_dev_ve_test_yan_yana(tmp_path):
    dev = tmp_path / "dev_ozet.csv"
    test = tmp_path / "test_ozet.csv"
    for yol, deger in ((dev, 0.061), (test, 0.12)):
        for d in ("E2", "E4", "E5"):
            toplu.ozet_yaz(str(yol), {"deney": d, "kayit_sayisi": 108 if yol == dev else 232, "der": deger,
                                      "jer_ort": deger * 2, "mikro_f1": 0.6, "toplam_sure_sn": 1, "toplam_ses_sn": 1,
                                      "atlanan": 0})
    e5 = tmp_path / "E5_ayarli_ozet.json"
    e5.write_text(json.dumps({"kayit_sayisi": 108, "der": 0.05, "jer_ort": 0.1, "mikro_f1": 0.7}), encoding="utf-8")
    satirlar = test_seti.karsilastirma(str(dev), str(test), dev_ek={"E5": str(e5), "E4": str(tmp_path / "yok.json")})
    e4 = next(r for r in satirlar if r["deney"] == "E4")
    assert e4["dev_der"] == pytest.approx(0.061) and e4["test_der"] == pytest.approx(0.12)
    assert e4["dev_kayit"] == 108 and e4["test_kayit"] == 232
    e5s = next(r for r in satirlar if r["deney"] == "E5")
    assert e5s["dev_der"] == pytest.approx(0.05) and e5s["dev_f1"] == pytest.approx(0.7)  # Hafta 5 JSON'u
    bos = test_seti.karsilastirma(None, str(test), deneyler=("E2",))[0]
    assert math.isnan(bos["dev_der"]) and bos["test_der"] == pytest.approx(0.12)


def test_grup_tablosu():
    satirlar = [_satir("a", 0.1, 1.0, 1, 0), _satir("b", 0.3, 0.5, 2, 2), _satir("c", 0.2, 0.0, 0, 0),
                _satir("listede_yok", 0.9, 0.0, 0, 1)]
    sayilar = {"a": 2, "b": 2, "c": 7}
    g = {r["grup"]: r for r in test_seti.grup_tablosu(satirlar, sayilar)}
    assert [r for r in test_seti.GRUPLAR] == list(g)
    assert g["2"]["kayit_sayisi"] == 2 and g["2"]["der"] == pytest.approx(0.2)  # (10 + 30) / 200
    assert g["2"]["makro_f1"] == pytest.approx(0.75)
    assert g["5+"]["kayit_sayisi"] == 1 and math.isnan(g["5+"]["makro_f1"])  # değişimsiz kayıt makroya girmez
    assert g["1"]["kayit_sayisi"] == 0 and math.isnan(g["1"]["der"])


def test_en_kotu(tmp_path):
    satirlar = [_satir(f"k{i}", i / 10, 0.5, 1, 1) for i in range(12)]
    satirlar.append({**_satir("nan", 0, 0, 0, 0), "der": "nan"})
    klasor = tmp_path / "E4"
    _rttm(str(klasor), "k11", [(0, 1, "spk00"), (1, 2, "spk01")])
    en = test_seti.en_kotu(satirlar, 10, {"k11": 3}, str(klasor))
    assert [r["kayit"] for r in en[:3]] == ["k11", "k10", "k9"] and len(en) == 10
    assert en[0]["ref_konusmaci"] == 3 and en[0]["sistem_konusmaci"] == 2
    assert math.isnan(en[1]["sistem_konusmaci"]) and "nan" not in [r["kayit"] for r in en]


def test_tablo_metni():
    m = test_seti.tablo_metni([{"deney": "E4", "test_kayit": 232.0, "test_der": 0.1234}],
                              ["deney", "test_kayit", "test_der"])
    assert m.splitlines()[2].split() == ["E4", "232", "0.123"]
    assert test_seti.tablo_metni([], ["a"]) == "(boş)"


def test_cihaz_bilgisi_ve_etiket():
    b = test_seti.cihaz_bilgisi()
    assert b["tur"] in ("cpu", "gpu") and b["ad"]
    assert test_seti.cihaz_etiketi({"tur": "gpu", "ad": "Tesla T4"}) == "gpu_tesla_t4"
    assert test_seti.cihaz_etiketi({"tur": "cpu", "ad": "Intel(R) Xeon(R) CPU @ 2.20GHz"}) == "cpu_intel_r_xeon_r_cpu_2_20ghz"
    assert test_seti.cihaz_etiketi({"tur": "cpu", "ad": ""}) == "cpu"


def test_hiz_olc_isinma_haric_medyan():
    cagri = []

    def sahte(yol, yontem):
        cagri.append(yontem)
        k = len(cagri)  # 1. çağrı ısınma (çok yavaş), sonrakiler 1..5 kat
        olcek = 100.0 if k == 1 else float(k - 1)
        return {"ses_sn": 90.0, "sureler": {"okuma": 0.1 * olcek, "vad": 0.2 * olcek, "oznitelik": 1.0 * olcek,
                                             "kumeleme": 0.01 * olcek}}

    s = test_seti.hiz_olc("x.wav", "ecapa", tekrar=5, isinma=1, hat=sahte)
    assert len(cagri) == 6 and len(s["olcumler"]) == 5
    assert s["medyan"]["oznitelik"] == pytest.approx(3.0) and s["medyan"]["toplam"] == pytest.approx(1.31 * 3)
    assert s["hedef_sn"] == pytest.approx(45.0) and s["gecti"] and s["rtf"] == pytest.approx(1.31 * 3 / 90)
    yavas = test_seti.hiz_olc("x.wav", "mfcc", tekrar=1, isinma=0,
                              hat=lambda y, m: {"ses_sn": 60.0, "sureler": {"okuma": 40.0}})
    assert yavas["hedef_sn"] == pytest.approx(30.0) and not yavas["gecti"]  # süreye orantılı hedef


def test_hiz_yaz_cihazlar_birikir(tmp_path):
    s = test_seti.hiz_olc("x.wav", "ecapa", tekrar=1, isinma=0,
                          hat=lambda y, m: {"ses_sn": 90.0, "sureler": {a: 1.0 for a in test_seti.ASAMALAR}})
    g = test_seti.hiz_yaz([s], str(tmp_path), {"tur": "gpu", "ad": "Tesla T4", "torch_thread": 2})
    c = test_seti.hiz_yaz([s], str(tmp_path), {"tur": "cpu", "ad": "Xeon", "torch_thread": 2})
    assert os.path.basename(g) == "hiz_gpu_tesla_t4.json" and os.path.basename(c) == "hiz_cpu_xeon.json"
    tablo = test_seti.hiz_tablosu(str(tmp_path))
    assert "GPU Tesla T4" in tablo and "CPU Xeon" in tablo and tablo.count("GEÇTİ") == 2
    assert test_seti.hiz_tablosu(str(tmp_path / "yok")) == "(hız ölçümü yok)"


def test_kisa_ozet_60_satiri_gecmez(tmp_path):
    test_seti._csv_yaz(str(tmp_path / "ozet.csv"), test_seti.karsilastirma(None, str(tmp_path / "yok.csv")),
                       test_seti.KARSILASTIRMA_SUTUNLARI)
    test_seti._csv_yaz(str(tmp_path / "gruplar_E4.csv"),
                       test_seti.grup_tablosu([_satir("a", 0.1, 1, 1, 0)], {"a": 2}), test_seti.GRUP_SUTUNLARI)
    test_seti._csv_yaz(str(tmp_path / "en_kotu_E4.csv"),
                       test_seti.en_kotu([_satir(f"k{i}", i / 10, 0.5, 1, 1) for i in range(12)]),
                       test_seti.EN_KOTU_SUTUNLARI)
    (tmp_path / "deneyler").mkdir()
    (tmp_path / "deneyler" / toplu.HATA_GUNLUGU).write_text("".join(f"hata {i}\n" for i in range(50)),
                                                           encoding="utf-8")
    m = test_seti.kisa_ozet(str(tmp_path), {"ortak": ["a"] * 232, "yalniz_ses": [], "yalniz_rttm": ["x"]})
    satirlar = m.splitlines()
    assert len(satirlar) <= 60 and satirlar[0].startswith("# Dev")
    assert "yalniz RTTM 1" in m.replace("yalnız", "yalniz") and "değerlendirilen 232" in m
    assert "hata 49" in m or satirlar[-1].startswith("...")  # hata kuyruğu ya da kesilme notu
    kisa = test_seti.kisa_ozet(str(tmp_path), maks_satir=10)
    assert len(kisa.splitlines()) == 10 and kisa.splitlines()[-1].startswith("...")


# --- Test zip'inin kaynağı ------------------------------------------------------------------------

class _Yanit:
    def __init__(self, basliklar, durum=200):
        self.headers, self.status = basliklar, durum

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _http_hata(kod):
    import urllib.error

    return urllib.error.HTTPError("u", kod, "hata", {}, None)


def test_uzak_kontrol_head_basarisizsa_get(monkeypatch):
    import urllib.request

    cagri = []

    def sahte(istek, timeout=None):
        cagri.append(istek.get_method())
        if istek.get_method() == "HEAD":
            raise _http_hata(401)
        return _Yanit({"Content-Range": "bytes 0-0/4272764397"}, 206)

    monkeypatch.setattr(urllib.request, "urlopen", sahte)
    k = test_seti.uzak_kontrol("https://ornek/zip")
    assert cagri == ["HEAD", "GET"] and k["durum"] == "ok" and k["boyut"] == 4272764397

    monkeypatch.setattr(urllib.request, "urlopen", lambda istek, timeout=None: (_ for _ in ()).throw(_http_hata(403)))
    assert test_seti.uzak_kontrol("https://ornek/zip")["durum"] == "yetkisiz"

    def ag_hatasi(istek, timeout=None):
        raise OSError("bağlantı yok")

    monkeypatch.setattr(urllib.request, "urlopen", ag_hatasi)
    assert test_seti.uzak_kontrol("https://ornek/zip")["durum"] == "hata"


def test_zip_hazirla_drive_md5_dogru_ve_onbellek(tmp_path):
    icerik = b"resmi test zip" * 100
    md5 = __import__("hashlib").md5(icerik).hexdigest()
    drive = tmp_path / "drive.zip"
    drive.write_bytes(icerik)
    yerel = str(tmp_path / "yerel" / "test.zip")
    b = test_seti.zip_hazirla(yerel, str(drive), ["https://a"], md5, indir=None,
                              kontrol=lambda u: pytest.fail("adres denenmemeli"), log=lambda *a: None)
    assert b["kaynak"] == "drive" and b["md5_dogru"] and open(yerel, "rb").read() == icerik
    b2 = test_seti.zip_hazirla(yerel, None, [], md5, log=lambda *a: None)  # yan dosyadan: yeniden hesaplanmaz
    assert b2["md5_dogru"] and b2["kaynak"] == "drive"


def test_zip_hazirla_kaynak_sirasi(tmp_path):
    icerik = b"dogru icerik"
    md5 = __import__("hashlib").md5(icerik).hexdigest()
    drive = tmp_path / "drive.zip"
    drive.write_bytes(b"bozuk kopya")
    yerel = str(tmp_path / "test.zip")
    with open(yerel + ".kaynak", "w") as f:
        f.write("https://eski")
    with open(yerel, "wb") as f:
        f.write(b"baska adresten yarim")
    indirilen = []

    def indir(url, hedef):
        indirilen.append((url, os.path.exists(hedef)))
        with open(hedef, "wb") as f:
            f.write(icerik if url == "https://kaist" else b"yanlis")

    durumlar = {"https://oxford": {"durum": "yetkisiz", "kod": 401}, "https://kaist": {"durum": "ok"},
                "https://yanlis": {"durum": "ok"}}
    b = test_seti.zip_hazirla(yerel, str(drive), ["https://oxford", "https://yanlis", "https://kaist"], md5,
                              indir=indir, kontrol=lambda u: durumlar[u], log=lambda *a: None)
    assert b["kaynak"] == "https://kaist" and b["md5_dogru"]
    assert drive.read_bytes() == b"bozuk kopya"  # Drive'daki dosyaya dokunulmaz
    assert [d["durum"] for d in b["denemeler"]][:2] == ["md5 yanlış (" + __import__("hashlib").md5(b"bozuk kopya").hexdigest() + ")", "yetkisiz"]
    assert indirilen == [("https://yanlis", False), ("https://kaist", False)]  # yarım dosya silindi
    hic = test_seti.zip_hazirla(str(tmp_path / "yok.zip"), None, ["https://oxford"], md5,
                                kontrol=lambda u: {"durum": "yetkisiz", "kod": 401}, log=lambda *a: None)
    assert hic["kaynak"] is None and hic["denemeler"][-1]["durum"] == "yetkisiz"


def test_hf_wavlari_zip_olur(tmp_path):
    klasor = tmp_path / "audio" / "test"
    klasor.mkdir(parents=True)
    for ad in ("aaa", "bbb"):
        (klasor / f"{ad}.wav").write_bytes(b"x")  # içerik ses değil; yalnızca paketleme sınanır
    wavlar = test_seti.klasor_wavlari(str(tmp_path))
    assert sorted(wavlar) == ["aaa", "bbb"]
    zip_yol = str(tmp_path / "hf.zip")
    assert test_seti.wav_zip_olustur(wavlar, zip_yol) == 2
    with zipfile.ZipFile(zip_yol) as z:
        assert z.namelist() == ["audio/aaa.wav", "audio/bbb.wav"]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())
    assert test_seti.zip_kayitlari(zip_yol) == ["aaa", "bbb"]


def test_kaynak_json_ve_ozet(tmp_path):
    test_seti.kaynak_yaz(str(tmp_path), {"kaynak": "https://kaist", "md5": "x", "md5_dogru": True})
    assert "Veri kaynağı: https://kaist | MD5: doğru" in test_seti.kisa_ozet(str(tmp_path))
    assert "uygulanamadı" in test_seti.kaynak_metni({"kaynak": "hf:depo", "md5_dogru": None})