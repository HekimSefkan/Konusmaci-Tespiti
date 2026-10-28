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
