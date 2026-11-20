"""Web arayüzü mantığı (Gradio). Sözleşme: docs/arayuz.md (3.15). Başlatma: python app.py [--share]

gradio yalnızca tema ve arayuz_olustur içinde import edilir; diğer fonksiyonlar gradio ve model
olmadan çalışır ve test edilir (hat parametresiyle sahte hat verilebilir).
Modeller (ECAPA, Silero VAD) ilk istekte bir kez yüklenir ve süreç boyunca tutulur
(oznitelik.ecapa_modeli, vad._silero_modeli); GPU varsa ECAPA onu kullanır.
Güvenlik: "Sunucudaki ses yolu" yalnızca KT_IZINLI_DIZIN altını açar; geçmiş KT_GECMIS_DIZIN'e yazılır
(src/gecmis.py); şifre koruması app.py'de (KT_KULLANICI, KT_SIFRE).
"""
import html
import math
import os
import re
import tempfile
import time
from bisect import bisect_right
from dataclasses import dataclass

from . import cikti, degerlendirme, gecmis, gorsel, gorsel_etkilesimli, main as ana, ses_girdi, varsayilanlar

BASLIK = "Konuşmacı Değişimi Tespiti"
SAYFA_BASLIGI = f"{BASLIK} — Grup_68"
ACIKLAMA = (
    "Bir ses kaydı yükleyin; sistem kimin ne zaman konuştuğunu bulur ve konuşmacı değişim noktalarını "
    f"listeler (WAV, MP3, FLAC; en fazla {ses_girdi.MAKS_SURE_SN // 60} dk)."
)
YUKLEME_NOTU = "İlk analizde modeller yüklenir, 1–2 dk sürebilir."
METRIK_NOTU = (
    "**DER** (diarization error rate): yanlış alarm + kaçırılan konuşma + konuşmacı karışıklığı süresinin "
    "referans konuşma süresine oranı; düşük olan iyidir. **JER**: konuşmacı başına hata oranlarının "
    "ortalaması. **Değişim F1**: referanstaki konuşmacı değişim noktalarının ±0,5 sn içinde bulunma "
    "başarısı (0–1); yüksek olan iyidir. DER/JER ±0,25 sn collar ile, örtüşen konuşma dahil hesaplanır."
)
YONTEMLER = (("ECAPA (önerilen)", "ecapa"), ("MFCC", "mfcc"))
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORNEK_RTTM_DIZINLERI = ("data/sample", "data/sample/edge")  # otomatik referans aranan klasörler (köke göre)
COLLAR = 0.5       # pyannote tam genişlik: ±0,25 sn
TOLERANS_SN = 0.5  # değişim noktası eşleşme toleransı


@dataclass
class Sonuc:
    """analiz çıktısı."""
    dosya_id: str
    segmentler: list            # [(bas_sn, bit_sn, etiket)]
    ses_sn: float               # ses süresi
    islem_sn: float             # hattın çalışma süresi (model yükleme dahil olabilir)
    rttm_yolu: str
    png_yolu: str
    referans: list | None       # referans segmentleri (verildiyse)
    metrikler: dict | None      # degerlendirme sözlüğü (referans verildiyse)
    ayrinti: dict | None = None  # main.hat_ayrintili ara bilgileri (hat sözlük döndürdüyse)


def _bos_mu(x) -> bool:
    return x is None or (isinstance(x, str) and not x.strip()) or (isinstance(x, float) and math.isnan(x))


def girdileri_denetle(konusmaci_sayisi, esik) -> tuple[int | None, float | None]:
    """Arayüzden gelen isteğe bağlı sayıları denetler; boş değer veya 0 = otomatik (None).

    Konuşmacı sayısı None ise sayı eşikle bulunur; eşik None ise yöntemin varsayilanlar.json eşiği.
    Hata: ValueError (konuşmacı sayısı negatif veya tam sayı değil; eşik negatif).
    """
    k = None
    if not _bos_mu(konusmaci_sayisi):
        k_f = float(konusmaci_sayisi)
        if k_f != int(k_f) or k_f < 0:
            raise ValueError(f"konuşmacı sayısı 0 (otomatik) veya pozitif bir tam sayı olmalı (girilen: {konusmaci_sayisi})")
        k = int(k_f) or None
    e = None
    if not _bos_mu(esik):
        e = float(esik)
        if e < 0:
            raise ValueError(f"eşik 0 (otomatik) veya pozitif olmalı (girilen: {esik})")
        e = e or None
    return k, e


IZINLI_ORTAM = "KT_IZINLI_DIZIN"


def izinli_dizin() -> str | None:
    """KT_IZINLI_DIZIN'in gerçek yolu (sembolik bağlantılar çözülmüş); tanımlı değilse None."""
    deger = os.environ.get(IZINLI_ORTAM, "").strip()
    return os.path.realpath(os.path.expanduser(deger)) if deger else None


def _icinde_mi(yol: str, kok: str) -> bool:
    try:
        return os.path.normcase(os.path.commonpath([yol, kok])) == os.path.normcase(kok)
    except ValueError:  # farklı sürücüler (Windows)
        return False


def ses_yolu_sec(yuklenen: str | None, sunucu_yolu: str | None) -> str | None:
    """Kullanılacak ses yolu: sunucu_yolu doluysa o (yüklenen dosya yok sayılır), değilse yüklenen.

    Sunucu yolu yalnızca KT_IZINLI_DIZIN altındaki dosyalar için açılır: yol (göreliyse izinli klasöre
    göre) os.path.realpath ile çözülür, böylece ../ ve sembolik bağlantıyla kaçış engellenir.
    Mesajlarda izinli klasör dışındaki yollar gösterilmez; içerideki yollar klasöre göre gösterilir.
    Hata: ValueError (değişken tanımlı değil, klasör dışında, okunamıyor), FileNotFoundError (yok, klasör).
    """
    if _bos_mu(sunucu_yolu):
        return yuklenen or None
    kok = izinli_dizin()
    if kok is None:
        raise ValueError(f"sunucudaki ses yolu kapalı ({IZINLI_ORTAM} tanımlı değil)")
    ham = os.path.expanduser(sunucu_yolu.strip().strip('"').strip("'"))
    yol = os.path.realpath(ham if os.path.isabs(ham) else os.path.join(kok, ham))
    if not _icinde_mi(yol, kok):
        raise ValueError("bu yol izin verilen klasörün dışında")
    goreli = os.path.relpath(yol, kok).replace(os.sep, "/")
    if not os.path.exists(yol):
        raise FileNotFoundError(f"izinli klasörde '{goreli}' bulunamadı")
    if not os.path.isfile(yol):
        raise FileNotFoundError(f"'{goreli}' bir klasör, ses dosyası değil")
    try:
        with open(yol, "rb") as f:
            f.read(1)
    except OSError as e:
        raise ValueError(f"'{goreli}' okunamadı ({type(e).__name__})") from e
    return yol


def otomatik_referans(ses_yolu: str) -> str | None:
    """Ses dosyasının uzantısız adıyla aynı adlı RTTM data/sample/ veya data/sample/edge/ içinde varsa yolu."""
    dosya_id = os.path.splitext(os.path.basename(ses_yolu))[0]
    for dizin in ORNEK_RTTM_DIZINLERI:
        yol = os.path.join(KOK, *dizin.split("/"), f"{dosya_id}.rttm")
        if os.path.isfile(yol):
            return yol
    return None


def analiz(ses_yolu: str, yontem: str = ana.VARSAYILAN_YONTEM, konusmaci_sayisi: int | None = None,
           esik: float | None = None, ref_rttm: str | None = None, cikti_dizin: str | None = None,
           hat=None, cizelge: bool = True) -> Sonuc:
    """Sesi hattan geçirir; RTTM ve (cizelge True ise) zaman çizelgesi PNG'si yazar, referans varsa ölçer.

    hat: (girdi, yontem, konusmaci_sayisi, esik) -> (segmentler, ses_sn) veya main.hat_ayrintili
    biçiminde sözlük (Sonuc.ayrinti'ye konur); varsayılan main.hat_ayrintili (eşik None ise
    varsayilanlar.json). Referans RTTM, hat çalışmadan önce okunur (bozuksa hemen hata).
    cikti_dizin verilmezse geçici klasör. cizelge False ise png_yolu "".
    Hata: FileNotFoundError, ValueError (girdi/RTTM sorunları; mesajda dosya adı).
    """
    if not ses_yolu:
        raise ValueError("ses dosyası seçilmedi")
    if yontem not in varsayilanlar.YONTEMLER:
        raise ValueError(f"bilinmeyen yöntem {yontem!r}")
    referans = None
    if ref_rttm:
        try:
            referans = cikti.rttm_oku(ref_rttm)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"{os.path.basename(ref_rttm)}: referans RTTM okunamadı ({e})") from e
    hat = hat or ana.hat_ayrintili
    t0 = time.perf_counter()
    r = hat(ses_yolu, yontem, konusmaci_sayisi, esik)
    islem_sn = time.perf_counter() - t0
    ayrinti = r if isinstance(r, dict) else None
    segmentler, ses_sn = (r["segmentler"], r["ses_sn"]) if ayrinti else r
    dosya_id = os.path.splitext(os.path.basename(ses_yolu))[0]
    cikti_dizin = cikti_dizin or tempfile.mkdtemp(prefix="kt_arayuz_")
    os.makedirs(cikti_dizin, exist_ok=True)
    rttm_yolu = os.path.join(cikti_dizin, f"{dosya_id}.rttm")
    cikti.rttm_yaz(segmentler, dosya_id, rttm_yolu)
    png_yolu = ""
    if cizelge:
        seritler = {"Referans": referans, "Sistem": segmentler} if referans is not None else {"Sistem": segmentler}
        png_yolu = gorsel.zaman_seritleri(seritler, os.path.join(cikti_dizin, f"{dosya_id}.png"), dosya_id)
    metrikler = None
    if referans is not None:
        metrikler = degerlendirme._segmentleri_degerlendir(referans, segmentler, COLLAR, TOLERANS_SN)
    return Sonuc(dosya_id, segmentler, ses_sn, islem_sn, rttm_yolu, png_yolu, referans, metrikler, ayrinti)


def segment_tablosu(segmentler) -> list[list]:
    """[[konuşmacı, başlangıç sn, bitiş sn, süre sn]] (3 ondalık)."""
    return [[e, round(a, 3), round(b, 3), round(b - a, 3)] for a, b, e in segmentler]


def konusmaci_toplamlari(segmentler) -> list[list]:
    """[[konuşmacı, toplam süre sn, yüzde]]; toplam süreye göre azalan."""
    toplam: dict[str, float] = {}
    for a, b, e in segmentler:
        toplam[e] = toplam.get(e, 0.0) + (b - a)
    genel = sum(toplam.values())
    return [[e, round(s, 3), round(100 * s / genel, 1) if genel > 0 else 0.0]
            for e, s in sorted(toplam.items(), key=lambda x: (-x[1], x[0]))]


def degisim_tablosu(segmentler) -> list[list]:
    """[[sıra, zaman sn, dk:ss.mmm]]: cikti.degisim_noktalari ile aynı tanım."""
    return [[i, round(t, 3), cikti._zaman(t)] for i, t in enumerate(cikti.degisim_noktalari(segmentler), 1)]


def _yuzde(x: float) -> str:
    return "–" if x is None or x != x else f"%{100 * x:.1f}"


def metrik_metni(metrikler: dict | None) -> str:
    """Referans ölçümlerinin Markdown özeti; referans yoksa açıklama."""
    if not metrikler:
        return "Referans RTTM verilmedi; DER/JER/F1 hesaplanmadı."
    m = metrikler
    satirlar = [
        "| Ölçü | Değer |", "|---|---|",
        f"| DER | {_yuzde(m['der'])} |",
        f"| JER | {_yuzde(m['jer'])} |",
        f"| Değişim F1 | {m['f1']:.3f} (kesinlik {m['precision']:.3f}, duyarlık {m['recall']:.3f}; "
        f"eşleşen {m['tp']}, fazla {m['fp']}, kaçırılan {m['fn']}) |",
        f"| Yanlış alarm / kaçırılan / karışıklık | {m['yanlis_alarm_sn']:.2f} / {m['kacirilan_sn']:.2f} / "
        f"{m['karisiklik_sn']:.2f} sn |",
    ]
    if m["der"] != m["der"]:
        satirlar.append("\nReferansta konuşma yok; DER/JER tanımsız (yanlış alarm süresine bakın).")
    return "\n".join(satirlar)


def hata_mesaji(dosya: str | None, hata: BaseException) -> str:
    """Kullanıcıya gösterilecek, dosya adını içeren Türkçe hata mesajı (Markdown)."""
    ad = os.path.basename(dosya) if dosya else "(dosya yok)"
    ayrinti = str(hata)
    if ayrinti.startswith(f"{ad}: "):
        ayrinti = ayrinti[len(ad) + 2:]
    kucuk = ayrinti.lower()
    if isinstance(hata, FileNotFoundError):
        ozet = "dosya bulunamadı"
    elif not isinstance(hata, ValueError):
        ozet = f"beklenmeyen bir hata oluştu ({type(hata).__name__})"
    elif "desteklenmeyen" in kucuk:
        ozet = "desteklenmeyen dosya biçimi; yalnızca WAV, MP3 ve FLAC kabul edilir"
    elif "kabul ediliyor" in kucuk:
        ozet = f"dosya çok uzun; en fazla {ses_girdi.MAKS_SURE_SN} sn ({ses_girdi.MAKS_SURE_SN // 60} dk) kabul ediliyor"
    elif "bozuk" in kucuk or "kesik" in kucuk or "okunamadı" in kucuk:
        ozet = "dosya bozuk veya okunamıyor"
    else:
        ozet = "dosya işlenemedi"
    return f"**Hata — {ad}:** {ozet}.\n\nAyrıntı: {ayrinti}"


def ozet_metni(s: Sonuc) -> str:
    konusmaci = len({e for *_, e in s.segmentler})
    if not s.segmentler:
        return (f"**{s.dosya_id}**: konuşma bulunamadı (ses {s.ses_sn:.1f} sn). "
                f"İşlem süresi {s.islem_sn:.1f} sn.")
    return (f"**{s.dosya_id}**: ses {s.ses_sn:.1f} sn, {konusmaci} konuşmacı, {len(s.segmentler)} segment, "
            f"{len(cikti.degisim_noktalari(s.segmentler))} değişim noktası. İşlem süresi {s.islem_sn:.1f} sn.")


def _isle(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu=None, hat=None, cizelge=True):
    """Ortak işleyici: (Sonuc | None, hata Markdown, referans notu, kullanılan ses yolu). İstisna fırlatmaz.

    cizelge False ise PNG çizilmez (yeni arayüz plotly kullanır).

    sunucu_yolu doluysa yüklenen dosya yerine o kullanılır (ses_yolu_sec). Referans verilmemişse
    otomatik_referans ile data/sample/ RTTM'i aranır; kullanılan referans notta belirtilir.
    """
    try:
        ses_yolu = ses_yolu_sec(ses_yolu, sunucu_yolu)
    except (FileNotFoundError, ValueError) as hata:  # mesaj izinli klasör dışını göstermez
        return None, f"**Hata — sunucu yolu:** {hata}.", "", None
    if not ses_yolu:
        return None, "**Önce bir ses dosyası seçin** (WAV, MP3 veya FLAC) veya sunucudaki ses yolunu yazın.", "", None
    try:
        k, e = girdileri_denetle(konusmaci_sayisi, esik)
    except ValueError as hata:
        return None, f"**Hata — ayarlar:** {hata}.", "", ses_yolu
    ref_notu = ""
    if ref_rttm:
        ref_notu = f"Referans: {os.path.basename(ref_rttm)} (yüklenen)"
    else:
        ref_rttm = otomatik_referans(ses_yolu)
        if ref_rttm:
            ref_notu = f"Referans: {os.path.relpath(ref_rttm, KOK).replace(os.sep, '/')} (otomatik)"
    try:
        return (analiz(ses_yolu, yontem or ana.VARSAYILAN_YONTEM, k, e, ref_rttm, hat=hat, cizelge=cizelge),
                "", ref_notu, ses_yolu)
    except Exception as hata:  # arayüz çökmesin; mesaj kullanıcıya
        dosya = ref_rttm if ref_rttm and "referans RTTM" in str(hata) else ses_yolu
        return None, hata_mesaji(dosya, hata), ref_notu, ses_yolu


CIKTI_SAYISI = 7  # durum, çizelge, segmentler, toplamlar, değişimler, RTTM, metrikler


def arayuz_cikti(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu=None, hat=None) -> tuple:
    """Yedi düz çıktı (API ve testler için); hiçbir zaman istisna fırlatmaz.

    Sıra: (durum Markdown, çizelge PNG, segment tablosu, konuşmacı toplamları, değişim noktaları,
    RTTM yolu, metrik Markdown). Hata olursa durumda hata mesajı, diğerleri boş.
    """
    s, hata, ref_notu, _ = _isle(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu, hat)
    if s is None:
        return (hata, None, [], [], [], None, "")
    metrik = metrik_metni(s.metrikler)
    durum = ozet_metni(s)
    if ref_notu:
        metrik = f"{ref_notu}\n\n{metrik}"
        durum = f"{durum}\n\n{ref_notu}"
    return (durum, s.png_yolu, segment_tablosu(s.segmentler), konusmaci_toplamlari(s.segmentler),
            degisim_tablosu(s.segmentler), s.rttm_yolu, metrik)


# ---------------------------------------------------------------------------
# Görünüm (HTML parçaları; gradio gerektirmez)
# ---------------------------------------------------------------------------

def _md_html(metin: str) -> str:
    """Kısa Markdown'ı (yalnızca **kalın** ve satır sonu) güvenli HTML'e çevirir."""
    h = html.escape(metin)
    h = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", h)
    return h.replace("\n\n", "<br>").replace("\n", "<br>")


def uyari_html(mesaj: str) -> str:
    """Kırmızı uyarı kutusu (hata mesajı); mesaj boşsa boş dize."""
    if not mesaj:
        return ""
    return f'<div class="kt-uyari kt-hata" role="alert">{_md_html(mesaj)}</div>'


def _sure_yaz(sn: float) -> str:
    """Okunur süre: 59,4 sn / 12 dk 05 sn."""
    if sn < 60:
        return f"{sn:.1f} sn".replace(".", ",")
    dk, kalan = divmod(int(round(sn)), 60)
    return f"{dk} dk {kalan:02d} sn"


def kartlar_html(s: Sonuc, ref_notu: str = "") -> str:
    """Özet kartları: konuşmacı sayısı, ses süresi, değişim noktası sayısı, işlem süresi."""
    return _kartlar(s.dosya_id, s.segmentler, s.ses_sn, s.islem_sn, ref_notu)


def _kartlar(dosya: str, segmentler, ses_sn: float, islem_sn: float, ref_notu: str = "") -> str:
    kartlar = [
        (len({e for *_, e in segmentler}), "konuşmacı"),
        (_sure_yaz(ses_sn), "ses süresi"),
        (len(cikti.degisim_noktalari(segmentler)), "değişim noktası"),
        (_sure_yaz(islem_sn), "işlem süresi"),
    ]
    govde = "".join(f'<div class="kt-kart"><div class="kt-deger">{html.escape(str(d))}</div>'
                    f'<div class="kt-etiket">{e}</div></div>' for d, e in kartlar)
    alt = html.escape(dosya) + (f" · {html.escape(ref_notu)}" if ref_notu else "")
    if not segmentler:
        alt += " · konuşma bulunamadı"
    return f'<div class="kt-dosya">{alt}</div><div class="kt-kartlar">{govde}</div>'


def _nokta(etiket: str, renkler: dict, adlar: dict | None = None) -> str:
    ad = (adlar or {}).get(etiket, etiket)
    return (f'<span class="kt-konusmaci"><span class="kt-nokta" style="background:{renkler.get(etiket, "#888")}">'
            f'</span>{html.escape(ad)}</span>')


def _tablo(basliklar, satirlar, bos_metin: str) -> str:
    if not satirlar:
        return f'<div class="kt-bos">{bos_metin}</div>'
    bas = "".join(f"<th>{b}</th>" for b in basliklar)
    govde = "".join("<tr>" + "".join(f"<td>{h}</td>" for h in s) + "</tr>" for s in satirlar)
    return f'<div class="kt-tablo"><table><thead><tr>{bas}</tr></thead><tbody>{govde}</tbody></table></div>'


def segment_html(segmentler) -> str:
    """Segment tablosu; konuşmacı renkleri zaman çizelgesiyle aynı (gorsel.renk_haritasi)."""
    renk = gorsel.renk_haritasi(e for *_, e in segmentler)
    satirlar = [[_nokta(e, renk), cikti._zaman(a), cikti._zaman(b), f"{b - a:.2f} sn".replace(".", ",")]
                for a, b, e in segmentler]
    return _tablo(["Konuşmacı", "Başlangıç", "Bitiş", "Süre"], satirlar, "Konuşma bulunamadı.")


def sure_html(segmentler, renk: dict | None = None, adlar: dict | None = None) -> str:
    """Konuşmacı başına toplam süre, yüzde ve renkli çubuk (renk verilmezse gorsel.renk_haritasi)."""
    renk = renk or gorsel.renk_haritasi(e for *_, e in segmentler)
    satirlar = [[_nokta(e, renk, adlar), _sure_yaz(s), f"%{y:.1f}".replace(".", ","),
                 f'<div class="kt-cubuk" style="width:{y:.1f}%;background:{renk.get(e, "#888")}"></div>']
                for e, s, y in konusmaci_toplamlari(segmentler)]
    return _tablo(["Konuşmacı", "Toplam", "Pay", ""], satirlar, "Konuşma bulunamadı.")


def degisim_ciftleri(segmentler) -> list[tuple[float, str, str]]:
    """[(zaman sn, önceki konuşmacı, yeni konuşmacı)]; zamanlar cikti.degisim_noktalari ile aynı."""
    nov = cikti.tek_konusmaci_segmentleri([(cikti._ms(a), cikti._ms(b), e) for a, b, e in segmentler])
    return [(nov[i][0] / 1000, nov[i - 1][2], nov[i][2]) for i in range(1, len(nov))]


def degisim_html(segmentler, renk: dict | None = None, adlar: dict | None = None) -> str:
    """Değişim noktaları: sıra, zaman, önceki → yeni konuşmacı."""
    renk = renk or gorsel.renk_haritasi(e for *_, e in segmentler)
    satirlar = [[i, cikti._zaman(t), f"{_nokta(a, renk, adlar)} → {_nokta(b, renk, adlar)}"]
                for i, (t, a, b) in enumerate(degisim_ciftleri(segmentler), 1)]
    return _tablo(["#", "Zaman", "Değişim"], satirlar, "Değişim noktası yok.")


def dogruluk_metni(s: Sonuc, ref_notu: str = "") -> str:
    """Doğruluk sekmesi (Markdown); referans yoksa boş dize."""
    if not s.metrikler:
        return ""
    return f"{ref_notu}\n\n{metrik_metni(s.metrikler)}\n\n{METRIK_NOTU}".strip()


GORUNUM_SAYISI = 8  # uyarı, kartlar, çizelge, segmentler, süreler, değişimler, doğruluk, RTTM


def gorunum_cikti(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu=None, hat=None) -> tuple:
    """Arayüz düğmesinin sekiz çıktısı; hiçbir zaman istisna fırlatmaz.

    Sıra: (uyarı HTML, kartlar HTML, çizelge PNG, segment HTML, süre HTML, değişim HTML,
    doğruluk Markdown, RTTM yolu). Hata olursa yalnızca uyarı dolu; doğruluk referans yoksa boş.
    """
    s, hata, ref_notu, _ = _isle(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu, hat)
    if s is None:
        return (uyari_html(hata), "", None, "", "", "", "", None)
    return ("", kartlar_html(s, ref_notu), s.png_yolu, segment_html(s.segmentler), sure_html(s.segmentler),
            degisim_html(s.segmentler), dogruluk_metni(s, ref_notu), s.rttm_yolu)


# ---------------------------------------------------------------------------
# Yeni arayüz: veri sözlüğü, basit/gelişmiş görünüm, geçmiş, segment dinleme
# ---------------------------------------------------------------------------
MODLAR = ("Basit", "Gelişmiş")
SEGMENT_SUTUNLARI = ["#", "Konuşmacı", "Başlangıç", "Bitiş", "Süre (sn)"]
GECMIS_SUTUNLARI = ["Tarih", "Dosya", "Konuşmacı", "Süre", "Kayıt"]


def izinli_yollar() -> list[str]:
    """Gradio'nun dosya sunabileceği ek klasörler (launch(allowed_paths=...)): izinli ve geçmiş klasörü."""
    return [y for y in (izinli_dizin(), gecmis.gecmis_dizini()) if y]


def bilgi_html(mesaj: str) -> str:
    """Sarı bilgi kutusu (ör. geçmişe kaydedilemedi); mesaj boşsa boş dize."""
    return f'<div class="kt-uyari kt-bilgi" role="status">{_md_html(mesaj)}</div>' if mesaj else ""


def konusmaci_adlari(segmentler, basit: bool) -> dict[str, str]:
    """Görünen adlar: basitte toplam süreye göre "Konuşmacı 1, 2, …", gelişmişte etiketin kendisi (spkNN)."""
    etiketler = [e for e, *_ in konusmaci_toplamlari(segmentler)]
    return {e: (f"Konuşmacı {i}" if basit else e) for i, e in enumerate(etiketler, 1)}


def konusmaci_benzerligi(pencereler, X, segmentler) -> dict | None:
    """Konuşmacıların ortalama gömmeleri arasındaki kosinüs benzerliği.

    Her pencere, orta noktasını içeren sistem segmentinin etiketine atanır (segment dışındakiler atlanır).
    Dönüş: {"etiketler": [...], "matris": [[...]]}; vektör yoksa veya 2'den az konuşmacı varsa None.
    """
    if X is None or not len(pencereler) or not segmentler:
        return None
    import numpy as np

    sirali = sorted(segmentler)
    baslar = [a for a, _, _ in sirali]
    gruplar: dict[str, list] = {}
    for (a, b), v in zip(pencereler, np.asarray(X, dtype=float)):
        m = (a + b) / 2
        i = bisect_right(baslar, m) - 1
        if i >= 0 and sirali[i][0] <= m < sirali[i][1]:
            gruplar.setdefault(sirali[i][2], []).append(v)
    etiketler = sorted(gruplar)
    if len(etiketler) < 2:
        return None
    M = np.array([np.mean(gruplar[e], axis=0) for e in etiketler])
    n = np.linalg.norm(M, axis=1, keepdims=True)
    U = M / np.where(n == 0, 1, n)
    S = np.clip(U @ U.T, -1, 1)
    return {"etiketler": etiketler, "matris": [[round(float(x), 4) for x in r] for r in S]}


def veri_olustur(s: Sonuc, ref_notu: str = "", ayarlar: dict | None = None, dosya_adi: str | None = None) -> dict:
    """Sonucu JSON'a yazılabilir görünüm sözlüğüne çevirir (geçmişe de bu kaydedilir)."""
    a = s.ayrinti or {}
    rttm_metni = ""
    if s.rttm_yolu and os.path.isfile(s.rttm_yolu):
        with open(s.rttm_yolu, encoding="utf-8") as f:
            rttm_metni = f.read()
    return {
        "dosya_id": s.dosya_id, "dosya_adi": dosya_adi or s.dosya_id, "ayarlar": ayarlar or {},
        "ses_sn": float(s.ses_sn), "islem_sn": float(s.islem_sn),
        "segmentler": [[float(x), float(y), e] for x, y, e in s.segmentler],
        "referans": [[float(x), float(y), e] for x, y, e in s.referans] if s.referans is not None else None,
        "ref_notu": ref_notu,
        "metrikler": {k: (int(v) if isinstance(v, int) else float(v)) for k, v in s.metrikler.items()}
        if s.metrikler else None,
        "vad": [[float(x), float(y)] for x, y in a["bolgeler"]] if a.get("bolgeler") is not None else None,
        "sureler": {k: float(v) for k, v in a["sureler"].items()} if a.get("sureler") else None,
        "kullanilan_esik": a.get("esik"),
        "benzerlik": konusmaci_benzerligi(a.get("pencereler") or [], a.get("X"), s.segmentler) if a else None,
        "rttm_metni": rttm_metni, "rttm_yolu": s.rttm_yolu, "ses_yolu": None,
    }


def segment_sesi(ses, sr: int, bas: float, bit: float):
    """Bellekteki sesten [bas, bit) aralığını keser (numpy): (sr, int16 dizi); aralık sese göre kırpılır."""
    import numpy as np

    x = np.asarray(ses)
    n = len(x)
    i0 = max(0, min(n, int(round(bas * sr))))
    i1 = max(i0, min(n, int(round(bit * sr))))
    parca = x[i0:i1]
    if parca.dtype.kind == "f":
        parca = (np.clip(parca, -1.0, 1.0) * 32767).astype(np.int16)
    return int(sr), parca


def segment_secimi(veri: dict | None, ses_bilgi: dict | None, satir, yukle=None) -> tuple:
    """Segment tablosunda seçilen satırın sesi: (oynatıcı değeri | None, ses_bilgi, not Markdown).

    ses_bilgi: {"yol", "dizi", "sr"}; dizi yoksa yol bir kez okunur (yukle; varsayılan
    ses_girdi.ses_yukle, süre sınırı yok) ve ses_bilgi'de saklanır. İstisna fırlatmaz.
    """
    ses_bilgi = dict(ses_bilgi or {})
    if not veri or satir is None or not 0 <= int(satir) < len(veri["segmentler"]):
        return None, ses_bilgi, ""
    a, b, e = veri["segmentler"][int(satir)]
    if ses_bilgi.get("dizi") is None:
        yol = ses_bilgi.get("yol")
        if not yol:
            return None, ses_bilgi, "Bu analizin sesi yok (geçmişe kopyalanmamış)."
        try:
            dizi, sr = (yukle or (lambda y: ses_girdi.ses_yukle(y, None)))(yol)
        except Exception as hata:
            return None, ses_bilgi, hata_mesaji(yol, hata)
        ses_bilgi.update(dizi=dizi, sr=sr)
    return (segment_sesi(ses_bilgi["dizi"], ses_bilgi["sr"], a, b), ses_bilgi,
            f"Çalınan segment: **{e}**, {cikti._zaman(a)} – {cikti._zaman(b)}")


def _yuzde_virgul(x: float) -> str:
    return f"%{x:.1f}".replace(".", ",")


def ozet_cumlesi(segmentler, adlar: dict | None = None) -> str:
    """Basit görünümdeki Türkçe özet: kaç kişi, en çok konuşan ve payı, kaç değişim."""
    adlar = adlar or {}
    toplamlar = konusmaci_toplamlari(segmentler)
    if not toplamlar:
        return "Kayıtta konuşma bulunamadı."
    d = len(cikti.degisim_noktalari(segmentler))
    e, s, y = toplamlar[0]
    if len(toplamlar) == 1:
        return f"Kayıtta tek kişi konuşuyor ({adlar.get(e, e)}, {_sure_yaz(s)}); konuşmacı değişimi yok."
    return (f"Kayıtta {len(toplamlar)} kişi konuşuyor. En çok {adlar.get(e, e)} konuşuyor "
            f"({_yuzde_virgul(y)} pay, {_sure_yaz(s)}). Konuşmacı {d} kez değişiyor.")


def _yazi_rengi(renk: str) -> str:
    r, g, b = (int(renk[i:i + 2], 16) for i in (1, 3, 5))
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#ffffff"


def segment_tablosu_df(segmentler, renk: dict, adlar: dict):
    """Seçilebilir segment tablosu (pandas Styler; Konuşmacı hücresi konuşmacı renginde)."""
    import pandas as pd

    df = pd.DataFrame([[i, adlar.get(e, e), cikti._zaman(a), cikti._zaman(b), round(b - a, 2)]
                       for i, (a, b, e) in enumerate(segmentler, 1)], columns=SEGMENT_SUTUNLARI)
    etiketler = [e for *_, e in segmentler]

    def boya(sutun):
        if sutun.name != "Konuşmacı":
            return [""] * len(sutun)
        return [f"background-color: {renk.get(e, '#888888')}; color: {_yazi_rengi(renk.get(e, '#888888'))}"
                for e in etiketler]

    return df.style.apply(boya)


def bilesen_html(metrikler: dict | None) -> str:
    """DER/JER/F1 ve DER bileşenleri (sn ve referans konuşma süresinin yüzdesi)."""
    if not metrikler:
        return ""
    m = metrikler
    toplam = m["referans_sn"]
    yzd = (lambda x: _yuzde_virgul(100 * x / toplam) if toplam > 0 else "–")
    hata = m["yanlis_alarm_sn"] + m["kacirilan_sn"] + m["karisiklik_sn"]
    satirlar = [["Yanlış alarm", f"{m['yanlis_alarm_sn']:.2f}", yzd(m["yanlis_alarm_sn"])],
                ["Kaçırılan konuşma", f"{m['kacirilan_sn']:.2f}", yzd(m["kacirilan_sn"])],
                ["Konuşmacı karışıklığı", f"{m['karisiklik_sn']:.2f}", yzd(m["karisiklik_sn"])],
                ["<strong>Toplam (DER)</strong>", f"<strong>{hata:.2f}</strong>", f"<strong>{_yuzde(m['der'])}</strong>"],
                ["Referans konuşma süresi", f"{toplam:.2f}", ""],
                ["JER", "", _yuzde(m["jer"])],
                ["Değişim F1 / kesinlik / duyarlık", "",
                 f"{m['f1']:.3f} / {m['precision']:.3f} / {m['recall']:.3f}"]]
    return _tablo(["Ölçü", "sn", "Değer"], satirlar, "")


def eslesme_html(referans, segmentler, tolerans_sn: float = TOLERANS_SN) -> str:
    """Değişim noktası eşleşmeleri: doğru eşleşen, kaçırılan, fazladan (±tolerans_sn)."""
    ref_d, hyp_d = cikti.degisim_noktalari(referans), cikti.degisim_noktalari(segmentler)
    ciftler = degerlendirme._eslesen_ciftler(ref_d, hyp_d, tolerans_sn)
    ri, hi = {i for i, _ in ciftler}, {j for _, j in ciftler}
    satirlar = [(ref_d[i], "Doğru eşleşen", cikti._zaman(ref_d[i]), cikti._zaman(hyp_d[j]),
                 f"{hyp_d[j] - ref_d[i]:+.2f} sn") for i, j in ciftler]
    satirlar += [(t, "Kaçırılan", cikti._zaman(t), "–", "") for i, t in enumerate(ref_d) if i not in ri]
    satirlar += [(t, "Fazladan", "–", cikti._zaman(t), "") for j, t in enumerate(hyp_d) if j not in hi]
    ozet = (f"<div class='kt-not'>±{tolerans_sn:g} sn: {len(ciftler)} doğru eşleşen, {len(ref_d) - len(ciftler)} "
            f"kaçırılan, {len(hyp_d) - len(ciftler)} fazladan.</div>")
    return ozet + _tablo(["Tür", "Referans", "Sistem", "Fark"], [s[1:] for s in sorted(satirlar)],
                         "Değişim noktası yok.")


def asama_html(sureler: dict | None, islem_sn: float) -> str:
    """Aşama süreleri (okuma, VAD, öznitelik, kümeleme)."""
    if not sureler:
        return '<div class="kt-bos">Aşama süreleri bu analiz için yok.</div>'
    adlar = {"okuma": "Ses okuma", "vad": "VAD", "oznitelik": "Öznitelik", "kumeleme": "Kümeleme"}
    satirlar = [[adlar.get(k, k), f"{v:.2f} sn"] for k, v in sureler.items()]
    satirlar.append(["<strong>Toplam (işlem)</strong>", f"<strong>{islem_sn:.2f} sn</strong>"])
    return _tablo(["Aşama", "Süre"], satirlar, "")


def gorunum_verisi(veri: dict, mod: str = "Basit") -> dict:
    """Görünüm sözlüğünden ekranda gösterilecek değerler (saf; plotly ve pandas kullanır).

    Anahtarlar: ozet, kartlar, cizelge, sureler, segmentler, degisimler, dogruluk, bilesenler, eslesmeler,
    hata_cizelgesi, asamalar, benzerlik, rttm_metni, ses_yolu, rttm_yolu, gelismis, referans_var, renk, adlar.
    Basit: "Konuşmacı N" adları, tek (sistem) şeritli çizelge. Gelişmiş: spkNN; referans ve VAD şeritleri,
    doğruluk, eşleşme, hata çizelgesi, benzerlik ve aşama süreleri. Referans varsa sistem konuşmacıları
    referansa Macar algoritmasıyla eşlenir; renkler çizelge ve tablolarda aynıdır.
    """
    basit = mod != "Gelişmiş"
    seg = [tuple(x) for x in veri["segmentler"]]
    ref = [tuple(x) for x in veri["referans"]] if veri.get("referans") is not None else None
    esleme = gorsel_etkilesimli.konusmaci_esle(ref, seg) if ref else {}
    hyp_renk, ref_renk = gorsel_etkilesimli.renkler([e for *_, e in seg],
                                                    [e for *_, e in ref] if ref is not None else None, esleme)
    adlar = konusmaci_adlari(seg, basit)
    seritler = [{"ad": "Sistem", "segmentler": seg, "renk": hyp_renk, "adlar": adlar}]
    if not basit and ref is not None:
        seritler.insert(0, {"ad": "Referans", "segmentler": ref, "renk": ref_renk})
    if not basit and veri.get("vad"):
        seritler.append({"ad": "VAD", "segmentler": [(a, b, "konuşma") for a, b in veri["vad"]],
                         "renk": {"konuşma": "#7f7f7f"}, "adlar": {"konuşma": "konuşma bölgesi"}})
    referans_var = ref is not None and bool(veri.get("metrikler"))
    d = {
        "ozet": ozet_cumlesi(seg, adlar),
        "kartlar": _kartlar(veri.get("dosya_adi") or veri["dosya_id"], seg, veri["ses_sn"], veri["islem_sn"],
                            veri.get("ref_notu", "")),
        "cizelge": gorsel_etkilesimli.zaman_cizelgesi(seritler, cikti.degisim_noktalari(seg)),
        "sureler": sure_html(seg, hyp_renk, adlar),
        "segmentler": segment_tablosu_df(seg, hyp_renk, adlar),
        "degisimler": "", "dogruluk": "", "bilesenler": "", "eslesmeler": "", "hata_cizelgesi": None,
        "asamalar": "", "benzerlik": None, "rttm_metni": veri.get("rttm_metni", ""),
        "ses_yolu": veri.get("ses_yolu"), "rttm_yolu": veri.get("rttm_yolu"),
        "gelismis": not basit, "referans_var": referans_var, "renk": hyp_renk, "adlar": adlar,
    }
    if basit:
        return d
    d["degisimler"] = degisim_html(seg, hyp_renk, adlar)
    d["asamalar"] = asama_html(veri.get("sureler"), veri["islem_sn"])
    b = veri.get("benzerlik")
    if b:
        d["benzerlik"] = gorsel_etkilesimli.benzerlik_haritasi(b["etiketler"], b["matris"], adlar)
    if referans_var:
        gri = [e for e in hyp_renk if e not in esleme]
        not_gri = f"\n\nReferansla eşleşmeyen sistem konuşmacıları gri: {', '.join(gri)}." if gri else ""
        d["dogruluk"] = f"{veri.get('ref_notu', '')}\n\n{METRIK_NOTU}{not_gri}".strip()
        d["bilesenler"] = bilesen_html(veri["metrikler"])
        d["eslesmeler"] = eslesme_html(ref, seg)
        d["hata_cizelgesi"] = gorsel_etkilesimli.hata_cizelgesi(
            gorsel_etkilesimli.hata_bolgeleri(ref, seg, esleme), veri["ses_sn"])
    return d


def analiz_et(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu=None, hat=None,
              gecmis_kok: str | None = None) -> tuple:
    """Yeni arayüzün analizi: (veri | None, ses_bilgi | None, uyarı HTML). İstisna fırlatmaz.

    Başarılıysa ve gecmis_kok verildiyse analiz geçmişe kaydedilir (veri["gecmis_adi"]); kayıt
    hatası analizi bozmaz, sarı bilgi kutusunda gösterilir.
    """
    s, hata, ref_notu, yol = _isle(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu, hat,
                                   cizelge=False)
    if s is None:
        return None, None, uyari_html(hata)
    k, e = girdileri_denetle(konusmaci_sayisi, esik)
    ayarlar = {"yontem": yontem or ana.VARSAYILAN_YONTEM, "konusmaci_sayisi": k, "esik": e}
    veri = veri_olustur(s, ref_notu, ayarlar, os.path.basename(yol))
    veri["ses_yolu"] = yol
    a = s.ayrinti or {}
    ses_bilgi = {"yol": yol, "dizi": a.get("ses"), "sr": a.get("sr")}
    uyari = ""
    if gecmis_kok:
        try:
            veri["gecmis_adi"] = gecmis.kaydet(gecmis_kok, veri, yol, s.rttm_yolu)
        except OSError as h:
            uyari = bilgi_html(f"Analiz tamamlandı ama geçmişe kaydedilemedi ({type(h).__name__}).")
    return veri, ses_bilgi, uyari


def gecmis_tablosu(kok: str | None) -> list[list]:
    """Geçmiş sekmesinin satırları: [tarih, dosya, konuşmacı, süre, kayıt adı]."""
    return [[k["tarih"].replace("T", " ")[:16], k["dosya"], k["konusmaci"], _sure_yaz(float(k["ses_sn"])), k["ad"]]
            for k in gecmis.listele(kok)] if kok else []


def gecmis_acilis(kok: str | None, ad: str | None) -> tuple:
    """Geçmiş kaydını açar (yeniden hesaplamadan): (veri | None, ses_bilgi | None, uyarı HTML)."""
    if not kok or not ad:
        return None, None, uyari_html("**Önce geçmişten bir kayıt seçin.**")
    try:
        veri = gecmis.ac(kok, ad)
    except ValueError as h:
        return None, None, uyari_html(f"**Hata — geçmiş:** {h}.")
    return veri, {"yol": veri.get("ses_yolu"), "dizi": None, "sr": None}, ""


def gecmis_silme(kok: str | None, ad: str | None) -> tuple:
    """Kaydı siler: (yeni tablo, seçili kayıt = None, not Markdown)."""
    if not kok or not ad:
        return gecmis_tablosu(kok), None, "Önce silinecek kaydı seçin."
    try:
        gecmis.sil(kok, ad)
        notu = f"Silindi: {ad}"
    except (ValueError, OSError) as h:
        notu = f"Silinemedi: {h}"
    return gecmis_tablosu(kok), None, notu


CSS = """
.kt-ust h1 {margin: 0 0 4px; font-size: 1.9rem;}
.kt-ust p {margin: 0; color: var(--body-text-color-subdued);}
.kt-not {font-size: .85rem; color: var(--body-text-color-subdued);}
#kt-analiz {min-height: 52px; font-size: 1.1rem;}
#kt-mod {max-width: 360px;}
.kt-uyari {border-radius: 10px; padding: 12px 14px; margin: 4px 0;}
.kt-hata {background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b;}
.dark .kt-hata {background: #450a0a; border-color: #7f1d1d; color: #fecaca;}
.kt-bilgi {background: #fffbeb; border: 1px solid #fcd34d; color: #92400e;}
.dark .kt-bilgi {background: #451a03; border-color: #92400e; color: #fde68a;}
.kt-ozet {font-size: 1.05rem;}
.kt-dosya {font-size: .9rem; color: var(--body-text-color-subdued); margin: 2px 0 8px; word-break: break-all;}
.kt-kartlar {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px;}
.kt-kart {background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
          border-radius: 12px; padding: 14px 16px;}
.kt-deger {font-size: 1.7rem; font-weight: 700; color: var(--color-accent); line-height: 1.15;}
.kt-etiket {font-size: .85rem; color: var(--body-text-color-subdued); margin-top: 2px;}
.kt-tablo {max-height: 420px; overflow: auto;}
.kt-tablo table {width: 100%; border-collapse: collapse; font-size: .92rem;}
.kt-tablo th, .kt-tablo td {padding: 6px 10px; border-bottom: 1px solid var(--border-color-primary); text-align: left;}
.kt-tablo th {position: sticky; top: 0; background: var(--background-fill-secondary);}
.kt-konusmaci {white-space: nowrap;}
.kt-nokta {display: inline-block; width: .8em; height: .8em; border-radius: 50%; margin-right: 6px; vertical-align: middle;}
.kt-cubuk {height: 8px; border-radius: 4px; min-width: 2px;}
.kt-bos {padding: 12px; color: var(--body-text-color-subdued);}
@media (max-width: 640px) {.kt-ust h1 {font-size: 1.5rem;} .kt-deger {font-size: 1.35rem;}
                           .kt-tablo th, .kt-tablo td {padding: 5px 6px;}}
"""


def tema():
    """Arayüz teması (gradio 6'da launch(theme=...) ile verilir)."""
    import gradio as gr

    return gr.themes.Soft(primary_hue="indigo", secondary_hue="sky", neutral_hue="slate",
                          font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"])


def arayuz_olustur(hat=None):
    """Gradio Blocks arayüzünü kurar (başlatmaz). Başlatma: app.py; tema, CSS ve izinli yollar
    launch(theme=tema(), css=CSS, allowed_paths=izinli_yollar()) ile verilir.

    "Sunucudaki ses yolu" yalnızca KT_IZINLI_DIZIN tanımlıysa, "Geçmiş" sekmesi yalnızca
    KT_GECMIS_DIZIN tanımlıysa görünür (arayüz kurulurken okunur). hat: testler için sahte hat.
    """
    import gradio as gr
    import pandas as pd

    izinli = izinli_dizin()
    gecmis_kok = gecmis.gecmis_dizini()
    esik_ipucu = ", ".join(f"{y} {varsayilanlar.esik(y)}" for y in varsayilanlar.YONTEMLER)
    bos_df = pd.DataFrame(columns=SEGMENT_SUTUNLARI)

    with gr.Blocks(title=SAYFA_BASLIGI) as demo:
        gr.HTML(f'<div class="kt-ust"><h1>{BASLIK}</h1><p>{html.escape(ACIKLAMA)}</p></div>')
        mod = gr.Radio(choices=list(MODLAR), value=MODLAR[0], label="Görünüm", elem_id="kt-mod")
        veri_d = gr.State(None)
        ses_d = gr.State(None)
        with gr.Tabs(selected="analiz") as ust:
            with gr.Tab("Analiz", id="analiz"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=280):
                        ses = gr.File(label="Ses dosyası (WAV, MP3, FLAC)", type="filepath", height=200)
                    with gr.Column(scale=2, min_width=260):
                        sunucu = gr.Textbox(label="Sunucudaki ses yolu (isteğe bağlı)", visible=izinli is not None,
                                            placeholder="Hafta2/ornek_ses/kayit.wav",
                                            info="İzinli klasöre göre yol; doluysa yüklenen dosya yerine kullanılır.")
                        with gr.Accordion("Gelişmiş ayarlar", open=False):
                            yontem = gr.Radio(choices=list(YONTEMLER), value=ana.VARSAYILAN_YONTEM, label="Yöntem")
                            konusmaci = gr.Number(value=0, precision=0, minimum=0,
                                                  label="Konuşmacı sayısı (0 = otomatik)")
                            esik = gr.Number(value=0, minimum=0, step=0.01,
                                             label=f"Eşik (0 = otomatik: {esik_ipucu})")
                            ref = gr.File(label="Referans RTTM (isteğe bağlı)", type="filepath")
                        dugme = gr.Button("Analiz et", variant="primary", size="lg", elem_id="kt-analiz")
                        gr.Markdown(YUKLEME_NOTU, elem_classes="kt-not")
                uyari = gr.HTML()
                with gr.Column(visible=False) as sonuc_alani:
                    ozet = gr.Markdown(elem_classes="kt-ozet")
                    kartlar = gr.HTML()
                    oynatici = gr.Audio(label="Ses", type="filepath", interactive=False)
                    cizelge = gr.Plot(label="Konuşmacı zaman çizelgesi")
                    with gr.Tabs():
                        with gr.Tab("Konuşmacı süreleri"):
                            sureler = gr.HTML()
                        with gr.Tab("Segmentler"):
                            gr.Markdown("Dinlemek için bir satır seçin.", elem_classes="kt-not")
                            segment_df = gr.Dataframe(value=bos_df, interactive=False, max_height=420)
                            segment_ses = gr.Audio(label="Seçilen segment", type="numpy", interactive=False,
                                                   autoplay=True)
                            segment_notu = gr.Markdown()
                        with gr.Tab("Değişim noktaları", visible=False) as sk_degisim:
                            degisimler = gr.HTML()
                        with gr.Tab("Doğruluk", visible=False) as sk_dogruluk:
                            bilesenler = gr.HTML()
                            dogruluk = gr.Markdown()
                            hata_cizelgesi = gr.Plot(label="Hata bölgeleri (collar'sız)")
                            eslesmeler = gr.HTML()
                        with gr.Tab("Hat ayrıntıları", visible=False) as sk_ayrinti:
                            asamalar = gr.HTML()
                            benzerlik = gr.Plot(label="Konuşmacı gömmeleri: kosinüs benzerliği")
                        with gr.Tab("RTTM", visible=False) as sk_rttm:
                            rttm_metni = gr.Code(label="RTTM", interactive=False)
                    indir = gr.DownloadButton("RTTM indir", visible=False)
            with gr.Tab("Geçmiş", id="gecmis", visible=gecmis_kok is not None) as sk_gecmis:
                gr.Markdown("Bir kayıt seçip **Aç** ile sonuç ekranını yeniden hesaplamadan açın; "
                            "**Sil** kaydı kalıcı olarak siler.", elem_classes="kt-not")
                gecmis_df = gr.Dataframe(headers=GECMIS_SUTUNLARI, interactive=False, max_height=420)
                with gr.Row():
                    yenile = gr.Button("Yenile")
                    ac = gr.Button("Aç", variant="primary")
                    sil = gr.Button("Sil", variant="stop")
                secili = gr.State(None)
                gecmis_notu = gr.Markdown()

        gorunum = [sonuc_alani, ozet, kartlar, oynatici, cizelge, sureler, segment_df, degisimler, sk_degisim,
                   sk_dogruluk, bilesenler, dogruluk, hata_cizelgesi, eslesmeler, sk_ayrinti, asamalar, benzerlik,
                   sk_rttm, rttm_metni, indir, segment_ses, segment_notu]

        def guncelle(veri, mod_):
            if not veri:
                return (gr.Column(visible=False), "", "", None, None, "", bos_df, "", gr.Tab(visible=False),
                        gr.Tab(visible=False), "", "", None, "", gr.Tab(visible=False), "", None,
                        gr.Tab(visible=False), "", gr.DownloadButton(visible=False), None, "")
            g = gorunum_verisi(veri, mod_)
            ileri, rv = g["gelismis"], g["referans_var"]
            return (gr.Column(visible=True), g["ozet"], g["kartlar"], g["ses_yolu"], g["cizelge"], g["sureler"],
                    g["segmentler"], g["degisimler"], gr.Tab(visible=ileri), gr.Tab(visible=ileri and rv),
                    g["bilesenler"], g["dogruluk"], g["hata_cizelgesi"], g["eslesmeler"], gr.Tab(visible=ileri),
                    g["asamalar"], g["benzerlik"], gr.Tab(visible=ileri), g["rttm_metni"],
                    gr.DownloadButton(value=g["rttm_yolu"], visible=bool(g["rttm_yolu"])), None, "")

        def calistir(ses_yolu, sunucu_yolu, yontem_, konusmaci_, esik_, ref_, mod_, progress=gr.Progress()):
            progress(0, desc="Analiz ediliyor…")
            veri, ses_bilgi, u = analiz_et(ses_yolu, yontem_, konusmaci_, esik_, ref_, sunucu_yolu, hat=hat,
                                           gecmis_kok=gecmis_kok)
            return (u, veri, ses_bilgi, *guncelle(veri, mod_))

        def segment_cal(veri, ses_bilgi, evt: gr.SelectData):
            satir = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            oyna, ses_bilgi, notu = segment_secimi(veri, ses_bilgi, satir)
            return oyna, ses_bilgi, notu

        def gecmis_sec(evt: gr.SelectData):
            ad = evt.row_value[-1] if evt.row_value else None
            return ad, f"Seçili kayıt: {ad}" if ad else ""

        def gecmis_ac(ad, mod_):
            veri, ses_bilgi, u = gecmis_acilis(gecmis_kok, ad)
            return (u, veri, ses_bilgi, *guncelle(veri, mod_), gr.Tabs(selected="analiz" if veri else "gecmis"))

        dugme.click(calistir, inputs=[ses, sunucu, yontem, konusmaci, esik, ref, mod],
                    outputs=[uyari, veri_d, ses_d, *gorunum], api_name="analiz", show_progress="full")
        mod.change(guncelle, inputs=[veri_d, mod], outputs=gorunum, api_name=False)
        segment_df.select(segment_cal, inputs=[veri_d, ses_d], outputs=[segment_ses, ses_d, segment_notu],
                          api_name=False)
        tablo_yenile = lambda: gecmis_tablosu(gecmis_kok)  # noqa: E731
        sk_gecmis.select(tablo_yenile, outputs=gecmis_df, api_name=False)
        yenile.click(tablo_yenile, outputs=gecmis_df, api_name=False)
        gecmis_df.select(gecmis_sec, outputs=[secili, gecmis_notu], api_name=False)
        ac.click(gecmis_ac, inputs=[secili, mod], outputs=[uyari, veri_d, ses_d, *gorunum, ust], api_name=False)
        sil.click(lambda ad: gecmis_silme(gecmis_kok, ad), inputs=secili, outputs=[gecmis_df, secili, gecmis_notu],
                  api_name=False)
    return demo