"""VoxConverse test seti değerlendirmesi ve hız ölçümü için küçük yardımcılar. Sözleşme: docs/arayuz.md (3.19)

Yeni algoritma yoktur: deneyler src/toplu.py ile, hat src/main.py ile çalışır. Bu modül test listesi
üretir (zip ↔ RTTM eşleştirme), dev/test özetlerini yan yana koyar, konuşmacı sayısı gruplarına göre
tablo ve en kötü kayıtları çıkarır, tek kayıtta aşama aşama hız ölçer ve defterin son hücresi için
kısa özet metni hazırlar. torch fonksiyon içinde import edilir; diğer her şey standart kütüphane + numpy.
"""
import csv
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
import zipfile

from . import bolme, toplu
from . import main as ana

GRUPLAR = ("1", "2", "3-4", "5+")
HEDEF_SES_SN = 90.0      # Hafta 2 hedefi: 90 sn ses en fazla 45 sn'de işlenir
HEDEF_SURE_SN = 45.0
ASAMALAR = ("okuma", "vad", "oznitelik", "kumeleme")
KARSILASTIRMA_SUTUNLARI = ["deney", "dev_kayit", "dev_der", "dev_jer", "dev_f1",
                           "test_kayit", "test_der", "test_jer", "test_f1"]
GRUP_SUTUNLARI = ["grup", "kayit_sayisi", "der", "jer_ort", "mikro_f1", "makro_f1"]
EN_KOTU_SUTUNLARI = ["kayit", "der", "jer", "f1", "ses_sn", "ref_konusmaci", "sistem_konusmaci"]


# ---------------------------------------------------------------------------
# Test listesi
# ---------------------------------------------------------------------------

def zip_kayitlari(zip_yol: str) -> list[str]:
    """Zip içindeki WAV'ların kayıt kimlikleri (uzantısız dosya adı), sıralı. Sesler okunmaz."""
    with zipfile.ZipFile(zip_yol) as z:
        return sorted({os.path.splitext(os.path.basename(a))[0] for a in z.namelist()
                       if a.lower().endswith(".wav")})


def rttm_kayitlari(rttm_dizin: str) -> list[str]:
    """Klasördeki <kayit>.rttm dosyalarının kayıt kimlikleri, sıralı. Hata: FileNotFoundError."""
    if not os.path.isdir(rttm_dizin):
        raise FileNotFoundError(f"RTTM klasörü yok: {rttm_dizin}")
    return sorted(os.path.splitext(a)[0] for a in os.listdir(rttm_dizin) if a.lower().endswith(".rttm"))


def eslestir(ses_adlari, rttm_adlari) -> dict[str, list[str]]:
    """{"ortak", "yalniz_ses", "yalniz_rttm"}: değerlendirilecek kayıtlar ortak olanlardır."""
    s, r = set(ses_adlari), set(rttm_adlari)
    return {"ortak": sorted(s & r), "yalniz_ses": sorted(s - r), "yalniz_rttm": sorted(r - s)}


def test_listesi_yaz(zip_yol: str, rttm_dizin: str, liste_yolu: str) -> dict[str, list[str]]:
    """Zip ile RTTM klasörünü eşleştirir, ortak kayıtları liste dosyasına yazar; eslestir sonucunu döndürür."""
    e = eslestir(zip_kayitlari(zip_yol), rttm_kayitlari(rttm_dizin))
    bolme.liste_yaz(e["ortak"], liste_yolu)
    return e


# ---------------------------------------------------------------------------
# Test zip'inin kaynağı: Drive → resmi yansılar (MD5 doğrulamalı) → Hugging Face yansısı
# ---------------------------------------------------------------------------

TEST_ZIP_MD5 = "834558bbd9b1ffd2d4893181556ceddd"   # KAIST sayfasındaki resmi test WAV zip MD5'i
KAIST_TEST_ZIP = "https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_test_wav.zip"
OXFORD_TEST_ZIP = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip"
HF_TEST_REPO = "ggfox00000/dia-voxconverse-test"   # son çare: tek tek WAV'lar (audio/test/*.wav)
_TARAYICI = {"User-Agent": "Mozilla/5.0 (konusmaci-tespiti)"}


def md5_dosya(yol: str, blok: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(blok), b""):
            h.update(parca)
    return h.hexdigest()


def uzak_kontrol(url: str, zaman_asimi: int = 60) -> dict:
    """Adrese erişilebiliyor mu: önce HEAD, başarısızsa GET (Range: bytes=0-0).

    Dönüş {"durum": "ok" | "yetkisiz" | "hata", "kod", "boyut", "mesaj"}; GET de 401/403 verirse
    "yetkisiz" (sıradaki kaynağa geçilir). Boyut Content-Length veya Content-Range'den.
    """
    import urllib.error
    import urllib.request

    son = {"durum": "hata", "kod": None, "boyut": None, "mesaj": ""}
    for yontem, ek in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        try:
            istek = urllib.request.Request(url, method=yontem, headers={**_TARAYICI, **ek})
            with urllib.request.urlopen(istek, timeout=zaman_asimi) as r:
                boyut = r.headers.get("Content-Length")
                aralik = r.headers.get("Content-Range") or ""
                if "/" in aralik:
                    boyut = aralik.rsplit("/", 1)[1]
                return {"durum": "ok", "kod": getattr(r, "status", None),
                        "boyut": int(boyut) if boyut and str(boyut).isdigit() else None, "mesaj": yontem}
        except urllib.error.HTTPError as e:
            son = {"durum": "yetkisiz" if e.code in (401, 403) else "hata", "kod": e.code, "boyut": None,
                   "mesaj": f"{yontem}: HTTP {e.code}"}
        except Exception as e:
            son = {"durum": "hata", "kod": None, "boyut": None, "mesaj": f"{yontem}: {type(e).__name__}: {e}"}
    return son


def _wget(url: str, hedef: str) -> None:
    subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", hedef, url], check=True)


def zip_hazirla(yerel: str, drive: str | None, adresler, md5_beklenen: str = TEST_ZIP_MD5, indir=None,
                kontrol=None, log=print) -> dict:
    """Test zip'ini yerel diske hazırlar; kaynak sırası: önceden doğrulanmış yerel kopya, Drive, adresler.

    Her zip MD5 ile doğrulanır; tutmayan yerel kopya silinir (Drive'daki dosyaya dokunulmaz).
    Adreslerde önce uzak_kontrol; "ok" değilse (401/403 dahil) sıradakine geçilir. İndirme wget -c ile
    kaldığı yerden sürer; yarım dosya başka bir adresten geldiyse silinip baştan indirilir.
    Dönüş {"kaynak": "drive" | adres | None, "md5", "md5_beklenen", "md5_dogru", "boyut", "denemeler"};
    hiçbir kaynak olmazsa kaynak None (Hugging Face yansısına geçilir).
    """
    indir, kontrol = indir or _wget, kontrol or uzak_kontrol
    os.makedirs(os.path.dirname(yerel) or ".", exist_ok=True)
    yan, kaynak_dosyasi = yerel + ".md5.json", yerel + ".kaynak"
    denemeler: list[dict] = []
    if os.path.isfile(yerel) and os.path.isfile(yan):
        with open(yan, encoding="utf-8") as f:
            j = json.load(f)
        if j.get("md5") == md5_beklenen and j.get("boyut") == os.path.getsize(yerel):
            log(f"Yerel zip önceden doğrulanmış ({j.get('kaynak')}).")
            return {**j, "md5_dogru": True, "denemeler": [{"kaynak": "yerel", "durum": "önceden doğrulanmış"}]}

    def dogrula(kaynak: str) -> dict | None:
        log(f"MD5 hesaplanıyor ({kaynak})...")
        md5 = md5_dosya(yerel)
        bilgi = {"kaynak": kaynak, "md5": md5, "md5_beklenen": md5_beklenen, "md5_dogru": md5 == md5_beklenen,
                 "boyut": os.path.getsize(yerel)}
        denemeler.append({"kaynak": kaynak, "durum": "md5 doğru" if bilgi["md5_dogru"] else f"md5 yanlış ({md5})"})
        if bilgi["md5_dogru"]:
            with open(yan, "w", encoding="utf-8") as f:
                json.dump(bilgi, f, ensure_ascii=False)
            return {**bilgi, "denemeler": denemeler}
        log(f"UYARI: MD5 tutmadı ({kaynak}): {md5}; bu kopya kullanılmıyor.")
        os.remove(yerel)
        return None

    if drive and os.path.isfile(drive):
        log("Drive'daki zip yerel diske kopyalanıyor...")
        shutil.copyfile(drive, yerel)
        if (b := dogrula("drive")) is not None:
            return b
    else:
        denemeler.append({"kaynak": "drive", "durum": "yok"})
    for url in adresler:
        if not url:
            continue
        k = kontrol(url)
        log(f"{url}: {k['durum']} {k.get('mesaj') or ''}".rstrip())
        if k["durum"] != "ok":
            denemeler.append({"kaynak": url, "durum": k["durum"], "kod": k.get("kod"), "mesaj": k.get("mesaj")})
            continue
        onceki = open(kaynak_dosyasi, encoding="utf-8").read().strip() if os.path.isfile(kaynak_dosyasi) else ""
        if os.path.isfile(yerel) and onceki != url:
            os.remove(yerel)  # başka adresten yarım kalmış dosya; -c ile devam edilmez
        with open(kaynak_dosyasi, "w", encoding="utf-8") as f:
            f.write(url)
        try:
            log(f"İndiriliyor: {url} (wget -c)")
            indir(url, yerel)
        except Exception as e:
            denemeler.append({"kaynak": url, "durum": "indirme hatası", "mesaj": f"{type(e).__name__}: {e}"})
            continue
        if (b := dogrula(url)) is not None:
            return b
    return {"kaynak": None, "md5": None, "md5_beklenen": md5_beklenen, "md5_dogru": None, "boyut": None,
            "denemeler": denemeler}


def klasor_wavlari(klasor: str) -> dict[str, str]:
    """Klasördeki (alt klasörler dahil) WAV'lar: {kayit kimliği: yol}."""
    return {os.path.splitext(os.path.basename(y))[0]: y
            for y in sorted(glob.glob(os.path.join(klasor, "**", "*.wav"), recursive=True))}


def wav_zip_olustur(wavlar: dict[str, str], zip_yolu: str) -> int:
    """WAV'ları sıkıştırmadan (ZIP_STORED) audio/<kayit>.wav olarak zip'e toplar (toplu.SesKaynagi okur).
    Atomik yazılır; yazılan dosya sayısını döndürür."""
    os.makedirs(os.path.dirname(zip_yolu) or ".", exist_ok=True)
    gecici = zip_yolu + ".yaziliyor"
    with zipfile.ZipFile(gecici, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for kayit in sorted(wavlar):
            z.write(wavlar[kayit], f"audio/{kayit}.wav")
    os.replace(gecici, zip_yolu)
    return len(wavlar)


def kaynak_yaz(sonuc_dizin: str, bilgi: dict) -> str:
    """Kullanılan veri kaynağını ve MD5 sonucunu kaynak.json'a yazar; yolu döndürür."""
    os.makedirs(sonuc_dizin, exist_ok=True)
    yol = os.path.join(sonuc_dizin, "kaynak.json")
    with open(yol, "w", encoding="utf-8") as f:
        json.dump({**bilgi, "tarih": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, ensure_ascii=False, indent=1)
    return yol


def kaynak_metni(bilgi: dict | None) -> str:
    if not bilgi:
        return "Veri kaynağı: bilinmiyor"
    md5 = {True: "doğru", False: "YANLIŞ", None: "uygulanamadı"}[bilgi.get("md5_dogru")]
    return f"Veri kaynağı: {bilgi.get('kaynak')} | MD5: {md5}"


# ---------------------------------------------------------------------------
# Özet tabloları
# ---------------------------------------------------------------------------

def _csv_oku(yol: str) -> list[dict]:
    return toplu._csv_oku(yol)


def _csv_yaz(yol: str, satirlar: list[dict], sutunlar: list[str]) -> None:
    os.makedirs(os.path.dirname(yol) or ".", exist_ok=True)
    gecici = yol + ".yaziliyor"
    with open(gecici, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sutunlar, extrasaction="ignore")
        w.writeheader()
        w.writerows(satirlar)
    os.replace(gecici, yol)


def _ozet_sozluk(yol: str | None) -> dict[str, dict]:
    """toplu ozet.csv -> {deney: {"kayit_sayisi", "der", "jer_ort", "mikro_f1"}}; dosya yoksa {}."""
    if not yol or not os.path.isfile(yol):
        return {}
    return {r["deney"]: {k: toplu._sayi(r.get(k)) for k in ("kayit_sayisi", "der", "jer_ort", "mikro_f1")}
            for r in _csv_oku(yol)}


def karsilastirma(dev_ozet: str | None, test_ozet: str, deneyler=("E2", "E4", "E5"),
                  dev_ek: dict[str, str] | None = None) -> list[dict]:
    """Dev değerlendirme yarısı ve test özetlerini yan yana koyar (KARSILASTIRMA_SUTUNLARI).

    dev_ozet: Hafta 4 ozet.csv. dev_ek: {deney: json yolu}; verilen ve var olan JSON (toplu.ozet_hesapla
    biçimi, ör. Hafta 5 E5_ayarli_ozet.json) o deneyin dev değerini değiştirir. Eksik değer NaN.
    """
    dev, test = _ozet_sozluk(dev_ozet), _ozet_sozluk(test_ozet)
    for deney, yol in (dev_ek or {}).items():
        if yol and os.path.isfile(yol):
            with open(yol, encoding="utf-8") as f:
                j = json.load(f)
            dev[deney] = {k: toplu._sayi(j.get(k)) for k in ("kayit_sayisi", "der", "jer_ort", "mikro_f1")}
    bos = {k: math.nan for k in ("kayit_sayisi", "der", "jer_ort", "mikro_f1")}
    satirlar = []
    for d in deneyler:
        a, b = dev.get(d, bos), test.get(d, bos)
        satirlar.append({"deney": d, "dev_kayit": a["kayit_sayisi"], "dev_der": a["der"], "dev_jer": a["jer_ort"],
                         "dev_f1": a["mikro_f1"], "test_kayit": b["kayit_sayisi"], "test_der": b["der"],
                         "test_jer": b["jer_ort"], "test_f1": b["mikro_f1"]})
    return satirlar


def grup_tablosu(satirlar: list[dict], sayilar: dict[str, int]) -> list[dict]:
    """Deney CSV satırlarını referans konuşmacı sayısı grubuna (1, 2, 3-4, 5+) göre özetler.

    Her grup: kayıt sayısı, DER (toplam saniyelerden), JER ortalaması, mikro F1 ve makro F1 (referansta
    değişim olan kayıtların F1 ortalaması). sayilar: {kayit: referans konuşmacı sayısı}; listede
    olmayan kayıt atlanır. Kaydı olmayan grup kayıt_sayisi 0 ve NaN değerlerle yazılır.
    """
    gruplar: dict[str, list[dict]] = {g: [] for g in GRUPLAR}
    for r in satirlar:
        if r["kayit"] in sayilar:
            gruplar[bolme.grup(sayilar[r["kayit"]])].append(r)
    sonuc = []
    for g in GRUPLAR:
        rr = gruplar[g]
        if not rr:
            sonuc.append({"grup": g, "kayit_sayisi": 0, "der": math.nan, "jer_ort": math.nan,
                          "mikro_f1": math.nan, "makro_f1": math.nan})
            continue
        o = toplu.ozet_hesapla(rr)
        degisimli = [toplu._sayi(r["f1"]) for r in rr if toplu._sayi(r["tp"]) + toplu._sayi(r["fn"]) > 0]
        sonuc.append({"grup": g, "kayit_sayisi": o["kayit_sayisi"], "der": o["der"], "jer_ort": o["jer_ort"],
                      "mikro_f1": o["mikro_f1"], "makro_f1": statistics.mean(degisimli) if degisimli else math.nan})
    return sonuc


def en_kotu(satirlar: list[dict], n: int = 10, sayilar: dict[str, int] | None = None,
            rttm_klasor: str | None = None) -> list[dict]:
    """DER'e göre en kötü n kayıt (EN_KOTU_SUTUNLARI). rttm_klasor verilirse sistem konuşmacı sayısı
    o klasördeki <kayit>.rttm'den okunur; DER'i NaN olan kayıtlar (referansta konuşma yok) atlanır."""
    from . import cikti

    gecerli = [r for r in satirlar if not math.isnan(toplu._sayi(r["der"]))]
    gecerli.sort(key=lambda r: (-toplu._sayi(r["der"]), r["kayit"]))
    sonuc = []
    for r in gecerli[:n]:
        sistem = math.nan
        if rttm_klasor:
            yol = os.path.join(rttm_klasor, f"{r['kayit']}.rttm")
            if os.path.isfile(yol):
                sistem = len({e for *_, e in cikti.rttm_oku(yol)})
        sonuc.append({"kayit": r["kayit"], "der": toplu._sayi(r["der"]), "jer": toplu._sayi(r["jer"]),
                      "f1": toplu._sayi(r["f1"]), "ses_sn": toplu._sayi(r["ses_sn"]),
                      "ref_konusmaci": (sayilar or {}).get(r["kayit"], math.nan), "sistem_konusmaci": sistem})
    return sonuc


def _hucre(x, basamak: int = 3) -> str:
    if isinstance(x, float):
        return "–" if math.isnan(x) else f"{x:.{basamak}f}"
    return str(x)


def tablo_metni(satirlar: list[dict], sutunlar: list[str], basamak: int = 3) -> str:
    """Satırları sabit genişlikli metin tablosuna çevirir (defter çıktısı için)."""
    if not satirlar:
        return "(boş)"
    hucreler = [[_hucre(r.get(s, ""), 0 if s.endswith(("kayit", "sayisi", "konusmaci")) else basamak)
                 for s in sutunlar] for r in satirlar]
    gen = [max(len(s), *(len(h[i]) for h in hucreler)) for i, s in enumerate(sutunlar)]
    satir = lambda h: "  ".join(x.rjust(g) if i else x.ljust(g) for i, (x, g) in enumerate(zip(h, gen)))  # noqa: E731
    return "\n".join([satir(sutunlar), "  ".join("-" * g for g in gen), *(satir(h) for h in hucreler)])


# ---------------------------------------------------------------------------
# Hız ölçümü
# ---------------------------------------------------------------------------

def cihaz_bilgisi() -> dict:
    """{"tur": "gpu"|"cpu", "ad", "torch_thread"}: GPU varsa adı, yoksa işlemci modeli."""
    bilgi = {"tur": "cpu", "ad": "", "torch_thread": None}
    try:
        import torch

        bilgi["torch_thread"] = torch.get_num_threads()
        if torch.cuda.is_available():
            return {**bilgi, "tur": "gpu", "ad": torch.cuda.get_device_name(0)}
    except ImportError:
        pass
    ad = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for satir in f:
                if satir.lower().startswith("model name"):
                    ad = satir.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {**bilgi, "ad": ad or platform.machine() or "bilinmiyor"}


def cihaz_etiketi(bilgi: dict) -> str:
    """Dosya adına uygun etiket: gpu_tesla_t4, cpu_intel_r_xeon_r_cpu_2_20ghz ..."""
    ad = re.sub(r"[^a-z0-9]+", "_", (bilgi.get("ad") or "").lower()).strip("_")[:40]
    return f"{bilgi.get('tur', 'cpu')}_{ad}" if ad else bilgi.get("tur", "cpu")


def hiz_olc(ses_yolu: str, yontem: str, tekrar: int = 5, isinma: int = 1, hat=None) -> dict:
    """Tek kayıtta hız: isinma + tekrar çalıştırma (main.hat_ayrintili), aşama başına medyan.

    Modeller ölçümden önce yüklenmiş olmalıdır (ilk ısınma çalıştırması da yükler; ölçüme girmez).
    Aşamalar: okuma (yükleme), vad, oznitelik, kumeleme; toplam = aşamaların toplamı.
    Dönüş: {"yontem", "ses_sn", "tekrar", "isinma", "medyan": {aşama: sn, "toplam"}, "olcumler": [...],
    "rtf", "hedef_sn" (ses süresine orantılı: 90 sn için 45 sn), "gecti"}.
    """
    hat = hat or ana.hat_ayrintili
    for _ in range(isinma):
        hat(ses_yolu, yontem)
    olcumler = []
    for _ in range(tekrar):
        t0 = time.perf_counter()
        r = hat(ses_yolu, yontem)
        duvar = time.perf_counter() - t0
        s = {a: float(r["sureler"].get(a, 0.0)) for a in ASAMALAR}
        olcumler.append({**s, "toplam": sum(s.values()), "duvar": duvar})
        ses_sn = float(r["ses_sn"])
    medyan = {k: statistics.median(o[k] for o in olcumler) for k in (*ASAMALAR, "toplam", "duvar")}
    hedef = HEDEF_SURE_SN * ses_sn / HEDEF_SES_SN
    return {"yontem": yontem, "ses_sn": ses_sn, "tekrar": tekrar, "isinma": isinma, "medyan": medyan,
            "olcumler": olcumler, "rtf": medyan["toplam"] / ses_sn if ses_sn else math.nan,
            "hedef_sn": hedef, "gecti": medyan["toplam"] <= hedef}


def hiz_yaz(sonuclar: list[dict], klasor: str, bilgi: dict | None = None) -> str:
    """Hız sonuçlarını cihaz etiketiyle hiz_<etiket>.json olarak yazar (cihazlar birbirini ezmez); yolu döndürür."""
    bilgi = bilgi or cihaz_bilgisi()
    os.makedirs(klasor, exist_ok=True)
    yol = os.path.join(klasor, f"hiz_{cihaz_etiketi(bilgi)}.json")
    with open(yol, "w", encoding="utf-8") as f:
        json.dump({"cihaz": bilgi, "tarih": time.strftime("%Y-%m-%dT%H:%M:%S"), "sonuclar": sonuclar}, f,
                  ensure_ascii=False, indent=1)
    return yol


def hiz_tablosu(klasor: str) -> str:
    """Klasördeki tüm hiz_*.json dosyalarını (GPU ve CPU) tek tabloda gösterir."""
    satirlar = []
    for yol in sorted(glob.glob(os.path.join(klasor, "hiz_*.json"))):
        with open(yol, encoding="utf-8") as f:
            j = json.load(f)
        cihaz = f"{j['cihaz'].get('tur', '?').upper()} {j['cihaz'].get('ad', '')}".strip()[:28]
        for s in j["sonuclar"]:
            m = s["medyan"]
            satirlar.append({"cihaz": cihaz, "yontem": s["yontem"], "okuma": m["okuma"], "vad": m["vad"],
                             "oznitelik": m["oznitelik"], "kumeleme": m["kumeleme"], "toplam": m["toplam"],
                             "hedef": s["hedef_sn"], "sonuc": "GEÇTİ" if s["gecti"] else "KALDI"})
    if not satirlar:
        return "(hız ölçümü yok)"
    return tablo_metni(satirlar, ["cihaz", "yontem", *ASAMALAR, "toplam", "hedef", "sonuc"], basamak=2)


# ---------------------------------------------------------------------------
# Kopyalanacak kısa özet
# ---------------------------------------------------------------------------

def kisa_ozet(sonuc_dizin: str, eslesme: dict | None = None, hata_satiri: int = 20, maks_satir: int = 60) -> str:
    """Defterin son hücresi için en fazla maks_satir satırlık özet: veri kaynağı ve MD5 (kaynak.json),
    karşılaştırma, grup tablosu, E4 en kötü 10, hız tablosu, eşleşmeyen kayıt sayısı ve varsa hata
    günlüğünün son satırları."""
    parcalar = []
    kaynak_yolu = os.path.join(sonuc_dizin, "kaynak.json")
    if os.path.isfile(kaynak_yolu):
        with open(kaynak_yolu, encoding="utf-8") as f:
            parcalar.append(f"# {kaynak_metni(json.load(f))}")
    oz = _csv_oku(os.path.join(sonuc_dizin, "ozet.csv"))
    if oz:
        sayisal = [{k: (v if k == "deney" else toplu._sayi(v)) for k, v in r.items()} for r in oz]
        parcalar += ["# Dev (değerlendirme yarısı) ve test", tablo_metni(sayisal, KARSILASTIRMA_SUTUNLARI)]
    for baslik, ad, sutunlar in (("# E4 konuşmacı sayısı grupları (test)", "gruplar_E4.csv", GRUP_SUTUNLARI),
                                 ("# E4 en kötü 10 (test)", "en_kotu_E4.csv", EN_KOTU_SUTUNLARI)):
        rr = _csv_oku(os.path.join(sonuc_dizin, ad))
        if rr:
            sayisal = [{k: (v if k in ("grup", "kayit") else toplu._sayi(v)) for k, v in r.items()} for r in rr]
            parcalar += [baslik, tablo_metni(sayisal, sutunlar)]
    parcalar += ["# Hız (medyan sn; hedef: 90 sn ses ≤ 45 sn)", hiz_tablosu(os.path.join(sonuc_dizin, "hiz"))]
    if eslesme is not None:
        parcalar.append(f"# Eşleşmeyen: yalnız ses {len(eslesme['yalniz_ses'])}, yalnız RTTM "
                        f"{len(eslesme['yalniz_rttm'])}, değerlendirilen {len(eslesme['ortak'])}")
    hatalar = []
    for yol in sorted(glob.glob(os.path.join(sonuc_dizin, "**", toplu.HATA_GUNLUGU), recursive=True)):
        with open(yol, encoding="utf-8") as f:
            hatalar += [s.rstrip("\n") for s in f]
    if hatalar:
        parcalar += [f"# Hata günlüğü (son {hata_satiri} satır)", *hatalar[-hata_satiri:]]
    satirlar = "\n".join(parcalar).splitlines()
    if len(satirlar) > maks_satir:
        satirlar = satirlar[:maks_satir - 1] + [f"... ({len(satirlar) - maks_satir + 1} satır kesildi)"]
    return "\n".join(satirlar)
