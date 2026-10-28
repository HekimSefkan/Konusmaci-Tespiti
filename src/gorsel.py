"""Görselleştirme: zaman çizelgesi ve deney sonuç grafiği. Sözleşme: docs/arayuz.md (3.13)

matplotlib fonksiyon içinde import edilir ve pyplot KULLANILMAZ: çizimler doğrudan Agg
tuvaline (FigureCanvasAgg) yapılır, PNG'ye yazılır. Ekranda pencere açılmaz ve defterdeki
satır içi (inline) çizim arka ucu değişmez.
"""
import os

from . import cikti


def _yeni_sekil(genislik: float, yukseklik: float, **alt_grafik):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sekil = Figure(figsize=(genislik, yukseklik))
    FigureCanvasAgg(sekil)
    return sekil, sekil.subplots(**alt_grafik)


def _kaydet(sekil, png_yolu: str) -> str:
    klasor = os.path.dirname(png_yolu)
    if klasor:
        os.makedirs(klasor, exist_ok=True)
    sekil.savefig(png_yolu, dpi=150, bbox_inches="tight")
    return png_yolu


def _renkler(n: int):
    import matplotlib

    harita = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return [harita(i % harita.N) for i in range(n)]


def zaman_cizelgesi(ref_rttm: str, hyp_rttm: str, png_yolu: str, baslik: str | None = None) -> str:
    """Aynı kaydın referans ve sistem RTTM'ini alt alta iki şeritte çizer, PNG'ye yazar.

    Her şeritte konuşmacı başına bir satır vardır; renkler her şeritte kendi içinde atanır
    (referans ve sistem etiketleri eşleştirilmez). Zaman ekseni saniye, iki şeritte ortaktır.
    Konuşma olmayan şeritte "konuşma yok" yazar.
    Dönüş: png_yolu. Hata: FileNotFoundError, ValueError (RTTM bozuk).
    """
    seritler = [("Referans", cikti.rttm_oku(ref_rttm)), ("Sistem", cikti.rttm_oku(hyp_rttm))]
    bitis = max((b for _, segler in seritler for _, b, _ in segler), default=1.0)
    satir_sayilari = [max(1, len({e for *_, e in segler})) for _, segler in seritler]
    sekil, eksenler = _yeni_sekil(12, 1.2 + 0.32 * sum(satir_sayilari) + 0.6, nrows=2, ncols=1, sharex=True,
                                  gridspec_kw={"height_ratios": satir_sayilari})
    for ax, (ad, segler) in zip(eksenler, seritler):
        etiketler = sorted({e for *_, e in segler})
        renk = _renkler(len(etiketler))
        for i, e in enumerate(etiketler):
            ax.broken_barh([(a, b - a) for a, b, l in segler if l == e], (i - 0.4, 0.8), color=renk[i])
        ax.set_yticks(range(len(etiketler)), etiketler)
        ax.set_ylim(max(len(etiketler), 1) - 0.5, -0.5)  # ilk konuşmacı üstte
        ax.set_ylabel(f"{ad}\n({len(etiketler)} konuşmacı)")
        ax.grid(axis="x", alpha=0.3)
        if not segler:
            ax.text(0.5, 0.5, "konuşma yok", transform=ax.transAxes, ha="center", va="center")
    eksenler[-1].set_xlim(0, bitis)
    eksenler[-1].set_xlabel("zaman (sn)")
    if baslik is None:
        baslik = os.path.splitext(os.path.basename(ref_rttm))[0]
    eksenler[0].set_title(baslik)
    return _kaydet(sekil, png_yolu)


_DENEY_SUTUNLARI = ("deney", "kayit", "der", "jer", "f1")  # deney CSV'lerinin sabit ilk beş sütunu
_OZET_SUTUNLARI = ("deney", "der", "jer_ort", "mikro_f1")


def _sonuclari_oku(csv_yolu) -> list[tuple[str, float, float, float]]:
    """[(deney, DER, JER, F1)]: ozet.csv satırları doğrudan, deney CSV'leri deney başına toplanarak."""
    import csv
    import math

    from .toplu import ozet_hesapla

    yollar = [csv_yolu] if isinstance(csv_yolu, (str, os.PathLike)) else list(csv_yolu)
    if not yollar:
        raise ValueError("en az bir CSV gerekli")
    sonuc: dict[str, tuple[float, float, float]] = {}
    for yol in yollar:
        with open(yol, encoding="utf-8", newline="") as f:
            satirlar = list(csv.DictReader(f))
        if not satirlar:
            raise ValueError(f"{yol}: CSV boş")
        sutunlar = set(satirlar[0])
        if set(_OZET_SUTUNLARI) <= sutunlar:
            for r in satirlar:
                sonuc[r["deney"]] = tuple(float(r[k]) if r[k] not in ("", None) else math.nan
                                          for k in ("der", "jer_ort", "mikro_f1"))
        elif set(_DENEY_SUTUNLARI) <= sutunlar:
            for deney in sorted({r["deney"] for r in satirlar}):
                o = ozet_hesapla([r for r in satirlar if r["deney"] == deney])
                sonuc[deney] = (o["der"], o["jer_ort"], o["mikro_f1"])
        else:
            raise ValueError(f"{yol}: ozet.csv veya deney CSV'si değil (sütunlar: {sorted(sutunlar)})")
    return [(d, *sonuc[d]) for d in sorted(sonuc)]


def sonuc_grafigi(csv_yolu, png_yolu: str) -> str:
    """Deney başına DER, JER ve değişim F1'ini gruplu çubuk grafikle çizer, PNG'ye yazar.

    csv_yolu: ozet.csv (der, jer_ort, mikro_f1), tek bir deney CSV'si (ilk beş sütun
    deney, kayit, der, jer, f1) veya bunların listesi. Deney CSV'leri toplu.ozet_hesapla ile
    toplanır (DER toplam saniyelerden, JER NaN'sız ortalama, F1 mikro).
    Dönüş: png_yolu. Hata: FileNotFoundError, ValueError (boş veya tanınmayan CSV).
    """
    import numpy as np

    veriler = _sonuclari_oku(csv_yolu)
    deneyler = [d for d, *_ in veriler]
    sekil, ax = _yeni_sekil(max(6, 1.4 * len(deneyler) + 2), 4)
    x = np.arange(len(deneyler))
    genislik = 0.27
    renk = _renkler(3)
    for i, ad in enumerate(("DER", "JER", "Değişim F1")):
        degerler = [v[i + 1] for v in veriler]
        cubuklar = ax.bar(x + (i - 1) * genislik, degerler, genislik, label=ad, color=renk[i])
        ax.bar_label(cubuklar, labels=[f"{v:.3f}" if v == v else "–" for v in degerler], fontsize=7, padding=2)
    ax.set_xticks(x, deneyler)
    tepe = max([v for _, *d in veriler for v in d if v == v] + [1.0])
    ax.set_ylim(0, tepe * 1.12)
    ax.set_ylabel("değer (0-1)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Deney başına DER, JER ve değişim F1")
    return _kaydet(sekil, png_yolu)
