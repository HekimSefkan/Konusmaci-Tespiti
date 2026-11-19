"""Web arayüzü mantığı (Gradio). Sözleşme: docs/arayuz.md (3.15). Başlatma: python app.py [--share]

gradio yalnızca arayuz_olustur içinde import edilir; diğer fonksiyonlar gradio ve model olmadan
çalışır ve test edilir (hat parametresiyle sahte hat verilebilir).
Modeller (ECAPA, Silero VAD) ilk istekte bir kez yüklenir ve süreç boyunca tutulur
(oznitelik.ecapa_modeli, vad._silero_modeli); GPU varsa ECAPA onu kullanır.
"""
import math
import os
import tempfile
import time
from dataclasses import dataclass

from . import cikti, degerlendirme, gorsel, main as ana, ses_girdi, varsayilanlar

BASLIK = "Ses Kayıtlarında Konuşmacı Değişimlerinin Tespiti — Grup_68"
ACIKLAMA = (
    "Bir ses kaydı yükleyin; sistem kimin ne zaman konuştuğunu bulur, konuşmacı değişim noktalarını "
    "listeler ve sonucu RTTM dosyası olarak verir. Desteklenen biçimler: WAV, MP3, FLAC "
    f"(en fazla {ses_girdi.MAKS_SURE_SN} sn). Referans RTTM verirseniz sonuç DER, JER ve değişim F1 ile ölçülür."
)
METRIK_NOTU = (
    "**DER** (diarization error rate): yanlış alarm + kaçırılan konuşma + konuşmacı karışıklığı süresinin "
    "referans konuşma süresine oranı; düşük olan iyidir. **JER**: konuşmacı başına hata oranlarının "
    "ortalaması. **Değişim F1**: referanstaki konuşmacı değişim noktalarının ±0,5 sn içinde bulunma "
    "başarısı (0–1); yüksek olan iyidir. DER/JER ±0,25 sn collar ile, örtüşen konuşma dahil hesaplanır."
)
YONTEMLER = (("ECAPA (önerilen)", "ecapa"), ("MFCC", "mfcc"))
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
    """Arayüzden gelen isteğe bağlı sayıları denetler; boş değer None olur.

    Hata: ValueError (konuşmacı sayısı tam sayı ve >= 1 değil; eşik > 0 değil).
    """
    k = None
    if not _bos_mu(konusmaci_sayisi):
        k_f = float(konusmaci_sayisi)
        if k_f != int(k_f) or k_f < 1:
            raise ValueError(f"konuşmacı sayısı 1 veya daha büyük bir tam sayı olmalı (girilen: {konusmaci_sayisi})")
        k = int(k_f)
    e = None
    if not _bos_mu(esik):
        e = float(esik)
        if not e > 0:
            raise ValueError(f"eşik 0'dan büyük olmalı (girilen: {esik})")
    return k, e


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


CIKTI_SAYISI = 7  # durum, çizelge, segmentler, toplamlar, değişimler, RTTM, metrikler


def arayuz_cikti(ses_yolu, yontem, konusmaci_sayisi, esik, ref_rttm, hat=None) -> tuple:
    """Düğme işleyicisi: arayüzün yedi çıktısını döndürür, hiçbir zaman istisna fırlatmaz.

    Sıra: (durum Markdown, çizelge PNG, segment tablosu, konuşmacı toplamları, değişim noktaları,
    RTTM yolu, metrik Markdown). Hata olursa durumda hata_mesaji, diğerleri boş.
    """
    bos = (None, [], [], [], None, "")
    if not ses_yolu:
        return ("**Önce bir ses dosyası seçin** (WAV, MP3 veya FLAC).", *bos)
    try:
        k, e = girdileri_denetle(konusmaci_sayisi, esik)
    except ValueError as hata:
        return (f"**Hata — ayarlar:** {hata}.", *bos)
    try:
        s = analiz(ses_yolu, yontem or ana.VARSAYILAN_YONTEM, k, e, ref_rttm or None, hat=hat)
    except Exception as hata:  # arayüz çökmesin; mesaj kullanıcıya
        dosya = ses_yolu
        if ref_rttm and "referans RTTM" in str(hata):
            dosya = ref_rttm
        return (hata_mesaji(dosya, hata), *bos)
    return (ozet_metni(s), s.png_yolu, segment_tablosu(s.segmentler), konusmaci_toplamlari(s.segmentler),
            degisim_tablosu(s.segmentler), s.rttm_yolu, metrik_metni(s.metrikler))


def arayuz_olustur(hat=None):
    """Gradio Blocks arayüzünü kurar (başlatmaz). hat: testler için sahte hat."""
    import gradio as gr

    esik_ipucu = ", ".join(f"{y} {varsayilanlar.esik(y)}" for y in varsayilanlar.YONTEMLER)
    with gr.Blocks(title=BASLIK) as demo:
        gr.Markdown(f"# {BASLIK}\n{ACIKLAMA}")
        with gr.Row():
            with gr.Column():
                ses = gr.File(label="Ses dosyası (WAV, MP3, FLAC)", type="filepath")
                yontem = gr.Radio(choices=list(YONTEMLER), value=ana.VARSAYILAN_YONTEM, label="Yöntem")
                konusmaci = gr.Number(value=None, precision=0, label="Konuşmacı sayısı (boş = otomatik)")
                esik = gr.Number(value=None, label=f"Eşik (boş = varsayılan: {esik_ipucu})")
                ref = gr.File(label="Referans RTTM (isteğe bağlı)", type="filepath")
                dugme = gr.Button("Analiz et", variant="primary")
                gr.Markdown(METRIK_NOTU)
            with gr.Column():
                durum = gr.Markdown()
                cizelge = gr.Image(label="Konuşmacı zaman çizelgesi", type="filepath")
                metrik = gr.Markdown()
                rttm = gr.File(label="RTTM (indir)")
        segment = gr.Dataframe(headers=["Konuşmacı", "Başlangıç (sn)", "Bitiş (sn)", "Süre (sn)"],
                               label="Segmentler", interactive=False)
        with gr.Row():
            toplam = gr.Dataframe(headers=["Konuşmacı", "Toplam (sn)", "Yüzde"],
                                  label="Konuşmacı başına toplam süre", interactive=False)
            degisim = gr.Dataframe(headers=["#", "Zaman (sn)", "dk:ss.ms"],
                                   label="Değişim noktaları", interactive=False)
        dugme.click(lambda *g: arayuz_cikti(*g, hat=hat), inputs=[ses, yontem, konusmaci, esik, ref],
                    outputs=[durum, cizelge, segment, toplam, degisim, rttm, metrik], api_name="analiz")
    return demo
