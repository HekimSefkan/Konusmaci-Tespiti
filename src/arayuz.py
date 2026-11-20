"""Web arayüzü mantığı (Gradio). Sözleşme: docs/arayuz.md (3.15). Başlatma: python app.py [--share]

gradio yalnızca arayuz_olustur içinde import edilir; diğer fonksiyonlar gradio ve model olmadan
çalışır ve test edilir (hat parametresiyle sahte hat verilebilir).
Modeller (ECAPA, Silero VAD) ilk istekte bir kez yüklenir ve süreç boyunca tutulur
(oznitelik.ecapa_modeli, vad._silero_modeli); GPU varsa ECAPA onu kullanır.
"""
import html
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass

from . import cikti, degerlendirme, gorsel, main as ana, ses_girdi, varsayilanlar

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


def ses_yolu_sec(yuklenen: str | None, sunucu_yolu: str | None) -> str | None:
    """Kullanılacak ses yolu: sunucu_yolu doluysa o (yüklenen dosya yok sayılır), değilse yüklenen.

    Hata: FileNotFoundError (yol yok veya klasör), ValueError (dosya okunamıyor); mesajda dosya adı ve yol.
    """
    if _bos_mu(sunucu_yolu):
        return yuklenen or None
    yol = os.path.expanduser(sunucu_yolu.strip().strip('"').strip("'"))
    ad = os.path.basename(yol) or yol
    if not os.path.exists(yol):
        raise FileNotFoundError(f"{ad}: sunucuda bu yol yok ({yol})")
    if not os.path.isfile(yol):
        raise FileNotFoundError(f"{ad}: bir klasör, ses dosyası değil ({yol})")
    try:
        with open(yol, "rb") as f:
            f.read(1)
    except OSError as e:
        raise ValueError(f"{ad}: okunamadı ({e})") from e
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
           hat=None) -> Sonuc:
    """Sesi hattan geçirir; RTTM ve zaman çizelgesi PNG'si yazar, referans varsa ölçer.

    hat: (girdi, yontem, konusmaci_sayisi, esik) -> (segmentler, ses_sn); varsayılan
    main.segmentleri_hesapla (eşik None ise varsayilanlar.json). Referans RTTM, hat
    çalışmadan önce okunur (bozuksa hemen hata). cikti_dizin verilmezse geçici klasör.
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
    hat = hat or ana.segmentleri_hesapla
    t0 = time.perf_counter()
    segmentler, ses_sn = hat(ses_yolu, yontem, konusmaci_sayisi, esik)
    islem_sn = time.perf_counter() - t0
    dosya_id = os.path.splitext(os.path.basename(ses_yolu))[0]
    cikti_dizin = cikti_dizin or tempfile.mkdtemp(prefix="kt_arayuz_")
    os.makedirs(cikti_dizin, exist_ok=True)
    rttm_yolu = os.path.join(cikti_dizin, f"{dosya_id}.rttm")
    cikti.rttm_yaz(segmentler, dosya_id, rttm_yolu)
    seritler = {"Referans": referans, "Sistem": segmentler} if referans is not None else {"Sistem": segmentler}
    png_yolu = gorsel.zaman_seritleri(seritler, os.path.join(cikti_dizin, f"{dosya_id}.png"), dosya_id)
    metrikler = None
    if referans is not None:
        metrikler = degerlendirme._segmentleri_degerlendir(referans, segmentler, COLLAR, TOLERANS_SN)
    return Sonuc(dosya_id, segmentler, ses_sn, islem_sn, rttm_yolu, png_yolu, referans, metrikler)


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


def _isle(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu=None, hat=None):
    """Ortak işleyici: (Sonuc | None, hata Markdown, referans notu). İstisna fırlatmaz.

    sunucu_yolu doluysa yüklenen dosya yerine o kullanılır (ses_yolu_sec). Referans verilmemişse
    otomatik_referans ile data/sample/ RTTM'i aranır; kullanılan referans notta belirtilir.
    """
    try:
        ses_yolu = ses_yolu_sec(ses_yolu, sunucu_yolu)
    except (FileNotFoundError, ValueError) as hata:
        return None, hata_mesaji(str(sunucu_yolu).strip(), hata), ""
    if not ses_yolu:
        return None, "**Önce bir ses dosyası seçin** (WAV, MP3 veya FLAC) veya sunucudaki ses yolunu yazın.", ""
    try:
        k, e = girdileri_denetle(konusmaci_sayisi, esik)
    except ValueError as hata:
        return None, f"**Hata — ayarlar:** {hata}.", ""
    ref_notu = ""
    if ref_rttm:
        ref_notu = f"Referans: {os.path.basename(ref_rttm)} (yüklenen)"
    else:
        ref_rttm = otomatik_referans(ses_yolu)
        if ref_rttm:
            ref_notu = f"Referans: {os.path.relpath(ref_rttm, KOK).replace(os.sep, '/')} (otomatik)"
    try:
        return analiz(ses_yolu, yontem or ana.VARSAYILAN_YONTEM, k, e, ref_rttm, hat=hat), "", ref_notu
    except Exception as hata:  # arayüz çökmesin; mesaj kullanıcıya
        dosya = ref_rttm if ref_rttm and "referans RTTM" in str(hata) else ses_yolu
        return None, hata_mesaji(dosya, hata), ref_notu


CIKTI_SAYISI = 7  # durum, çizelge, segmentler, toplamlar, değişimler, RTTM, metrikler


def arayuz_cikti(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu=None, hat=None) -> tuple:
    """Yedi düz çıktı (API ve testler için); hiçbir zaman istisna fırlatmaz.

    Sıra: (durum Markdown, çizelge PNG, segment tablosu, konuşmacı toplamları, değişim noktaları,
    RTTM yolu, metrik Markdown). Hata olursa durumda hata mesajı, diğerleri boş.
    """
    s, hata, ref_notu = _isle(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu, hat)
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
    kartlar = [
        (len({e for *_, e in s.segmentler}), "konuşmacı"),
        (_sure_yaz(s.ses_sn), "ses süresi"),
        (len(cikti.degisim_noktalari(s.segmentler)), "değişim noktası"),
        (_sure_yaz(s.islem_sn), "işlem süresi"),
    ]
    govde = "".join(f'<div class="kt-kart"><div class="kt-deger">{html.escape(str(d))}</div>'
                    f'<div class="kt-etiket">{e}</div></div>' for d, e in kartlar)
    alt = html.escape(s.dosya_id) + (f" · {html.escape(ref_notu)}" if ref_notu else "")
    if not s.segmentler:
        alt += " · konuşma bulunamadı"
    return f'<div class="kt-dosya">{alt}</div><div class="kt-kartlar">{govde}</div>'


def _nokta(etiket: str, renkler: dict) -> str:
    return (f'<span class="kt-konusmaci"><span class="kt-nokta" style="background:{renkler.get(etiket, "#888")}">'
            f'</span>{html.escape(etiket)}</span>')


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


def sure_html(segmentler) -> str:
    """Konuşmacı başına toplam süre, yüzde ve renkli çubuk."""
    renk = gorsel.renk_haritasi(e for *_, e in segmentler)
    satirlar = [[_nokta(e, renk), _sure_yaz(s), f"%{y:.1f}".replace(".", ","),
                 f'<div class="kt-cubuk" style="width:{y:.1f}%;background:{renk[e]}"></div>']
                for e, s, y in konusmaci_toplamlari(segmentler)]
    return _tablo(["Konuşmacı", "Toplam", "Pay", ""], satirlar, "Konuşma bulunamadı.")


def degisim_ciftleri(segmentler) -> list[tuple[float, str, str]]:
    """[(zaman sn, önceki konuşmacı, yeni konuşmacı)]; zamanlar cikti.degisim_noktalari ile aynı."""
    nov = cikti.tek_konusmaci_segmentleri([(cikti._ms(a), cikti._ms(b), e) for a, b, e in segmentler])
    return [(nov[i][0] / 1000, nov[i - 1][2], nov[i][2]) for i in range(1, len(nov))]


def degisim_html(segmentler) -> str:
    """Değişim noktaları: sıra, zaman, önceki → yeni konuşmacı."""
    renk = gorsel.renk_haritasi(e for *_, e in segmentler)
    satirlar = [[i, cikti._zaman(t), f"{_nokta(a, renk)} → {_nokta(b, renk)}"]
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
    s, hata, ref_notu = _isle(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, sunucu_yolu, hat)
    if s is None:
        return (uyari_html(hata), "", None, "", "", "", "", None)
    return ("", kartlar_html(s, ref_notu), s.png_yolu, segment_html(s.segmentler), sure_html(s.segmentler),
            degisim_html(s.segmentler), dogruluk_metni(s, ref_notu), s.rttm_yolu)


CSS = """
.kt-ust h1 {margin: 0 0 4px; font-size: 1.9rem;}
.kt-ust p {margin: 0; color: var(--body-text-color-subdued);}
.kt-not {font-size: .85rem; color: var(--body-text-color-subdued);}
#kt-analiz {min-height: 52px; font-size: 1.1rem;}
.kt-uyari {border-radius: 10px; padding: 12px 14px; margin: 4px 0;}
.kt-hata {background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b;}
.dark .kt-hata {background: #450a0a; border-color: #7f1d1d; color: #fecaca;}
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
    """Gradio Blocks arayüzünü kurar (başlatmaz; tema ve CSS: launch(theme=tema(), css=CSS)).

    hat: testler için sahte hat.
    """
    import gradio as gr

    esik_ipucu = ", ".join(f"{y} {varsayilanlar.esik(y)}" for y in varsayilanlar.YONTEMLER)
    with gr.Blocks(title=SAYFA_BASLIGI) as demo:
        gr.HTML(f'<div class="kt-ust"><h1>{BASLIK}</h1><p>{html.escape(ACIKLAMA)}</p></div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=280):
                ses = gr.File(label="Ses dosyası (WAV, MP3, FLAC)", type="filepath", height=200)
            with gr.Column(scale=2, min_width=260):
                sunucu = gr.Textbox(label="Sunucudaki ses yolu (isteğe bağlı)",
                                    placeholder="/content/drive/MyDrive/.../kayit.wav",
                                    info="Doluysa yüklenen dosya yerine bu yol kullanılır.")
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
        kartlar = gr.HTML()
        cizelge = gr.Image(label="Konuşmacı zaman çizelgesi", type="filepath", interactive=False)
        with gr.Tabs():
            with gr.Tab("Segmentler"):
                segment = gr.HTML()
            with gr.Tab("Konuşmacı süreleri"):
                sure = gr.HTML()
            with gr.Tab("Değişim noktaları"):
                degisim = gr.HTML()
            with gr.Tab("Doğruluk", visible=False) as dogruluk_sekme:
                dogruluk = gr.Markdown()
        indir = gr.DownloadButton("RTTM indir", visible=False)

        def calistir(ses_yolu, sunucu_yolu, yontem_, konusmaci_, esik_, ref_, progress=gr.Progress()):
            progress(0, desc="Analiz ediliyor…")
            u, k, png, seg, sur, deg, dog, rttm = gorunum_cikti(ses_yolu, yontem_, konusmaci_, esik_, ref_,
                                                                 sunucu_yolu, hat=hat)
            return (u, k, png, seg, sur, deg, dog, gr.Tab(visible=bool(dog)),
                    gr.DownloadButton(value=rttm, visible=bool(rttm)))

        dugme.click(calistir, inputs=[ses, sunucu, yontem, konusmaci, esik, ref],
                    outputs=[uyari, kartlar, cizelge, segment, sure, degisim, dogruluk, dogruluk_sekme, indir],
                    api_name="analiz", show_progress="full")
    return demo


