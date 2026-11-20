"""Etkileşimli (plotly) görseller: web arayüzü için. Sözleşme: docs/arayuz.md (3.16)

Raporlardaki PNG çizimleri src/gorsel.py'de (matplotlib) kalır. plotly ve scipy fonksiyon içinde
import edilir; konusmaci_esle, renkler ve hata_bolgeleri saf hesaplardır (renkler ve hata_bolgeleri
yalnızca standart kütüphane).
Arka planlar saydam, yazı ve ızgara gri: açık ve koyu temada okunur.
"""
from . import cikti
from .gorsel import TAB10, TAB20

GRILER = ("#9e9e9e", "#757575", "#bdbdbd", "#616161", "#e0e0e0", "#424242")  # eşleşmeyen konuşmacılar
HATA_RENKLERI = {"kacirilan": "#d62728", "yanlis_alarm": "#ff7f0e", "karisiklik": "#9467bd"}
HATA_ADLARI = {"kacirilan": "Kaçırılan", "yanlis_alarm": "Yanlış alarm", "karisiklik": "Karışıklık"}
YAZI = "#8a8f98"
IZGARA = "rgba(128,128,128,0.25)"


def _palet(n: int) -> tuple:
    return TAB10 if n <= 10 else TAB20


def konusmaci_esle(ref, hyp) -> dict[str, str]:
    """Sistem konuşmacılarını referansa Macar algoritmasıyla (scipy) eşler: {sistem etiketi: referans etiketi}.

    Maliyet = örtüşme süresi (sn, en büyüklenir); örtüşmesi 0 olan çiftler eşleşme sayılmaz.
    """
    ref_et = sorted({e for *_, e in ref})
    hyp_et = sorted({e for *_, e in hyp})
    if not ref_et or not hyp_et:
        return {}
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    ortusme = np.zeros((len(ref_et), len(hyp_et)))
    ri = {e: i for i, e in enumerate(ref_et)}
    hi = {e: j for j, e in enumerate(hyp_et)}
    for a, b, r in ref:
        for c, d, h in hyp:
            k = min(b, d) - max(a, c)
            if k > 0:
                ortusme[ri[r], hi[h]] += k
    satir, sutun = linear_sum_assignment(ortusme, maximize=True)
    return {hyp_et[j]: ref_et[i] for i, j in zip(satir, sutun) if ortusme[i, j] > 0}


def renkler(hyp_etiketler, ref_etiketler=None, esleme: dict | None = None) -> tuple[dict, dict]:
    """(sistem renkleri, referans renkleri) {etiket: "#rrggbb"}.

    Referans yoksa sistem etiketleri sıralanıp tab10/tab20'den renk alır (gorsel.renk_haritasi ile aynı).
    Referans varsa referans etiketleri paletten renk alır; eşleşen sistem konuşmacısı eşinin rengini,
    eşleşmeyenler gri tonlarını alır.
    """
    hyp = sorted(set(hyp_etiketler))
    if ref_etiketler is None:
        palet = _palet(len(hyp))
        return {e: palet[i % len(palet)] for i, e in enumerate(hyp)}, {}
    ref = sorted(set(ref_etiketler))
    palet = _palet(len(ref))
    ref_renk = {e: palet[i % len(palet)] for i, e in enumerate(ref)}
    esleme = esleme or {}
    hyp_renk, g = {}, 0
    for e in hyp:
        if esleme.get(e) in ref_renk:
            hyp_renk[e] = ref_renk[esleme[e]]
        else:
            hyp_renk[e] = GRILER[g % len(GRILER)]
            g += 1
    return hyp_renk, ref_renk


def hata_bolgeleri(ref, hyp, esleme: dict) -> list[tuple[float, float, str]]:
    """Kaçırılan, yanlış alarm ve karışıklık bölgeleri [(bas, bit, tür)] (collar'sız, ms çözünürlük).

    Her temel aralıkta R = etkin referans, H = etkin sistem konuşmacıları (esleme ile referans adına
    çevrilir): kaçırılan |R| > |H|, yanlış alarm |H| > |R|, karışıklık min(|R|, |H|) > |R ∩ H|.
    Bir aralık birden çok türde olabilir. Aynı türün bitişik aralıkları birleşir.
    """
    ms = [(cikti._ms(a), cikti._ms(b), ("R", e)) for a, b, e in ref]
    ms += [(cikti._ms(a), cikti._ms(b), ("H", esleme.get(e, ("?", e)))) for a, b, e in hyp]
    sonuc: list[list] = []
    for a, b, kume in cikti.temel_araliklar(ms):
        r = {e for t, e in kume if t == "R"}
        h = {e for t, e in kume if t == "H"}
        turler = []
        if len(r) > len(h):
            turler.append("kacirilan")
        if len(h) > len(r):
            turler.append("yanlis_alarm")
        if min(len(r), len(h)) > len(r & h):
            turler.append("karisiklik")
        for tur in turler:
            onceki = next((s for s in reversed(sonuc) if s[2] == tur), None)
            if onceki and onceki[1] == a:
                onceki[1] = b
            else:
                sonuc.append([a, b, tur])
    return [(a / 1000, b / 1000, t) for a, b, t in sorted(sonuc)]


def _duzen(fig, yukseklik: int, x_baslik: str | None = "zaman (sn)"):
    fig.update_layout(height=yukseklik, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=YAZI), showlegend=False,
                      hoverlabel=dict(font_size=13), dragmode="zoom")
    fig.update_xaxes(gridcolor=IZGARA, zeroline=False, title_text=x_baslik)
    fig.update_yaxes(gridcolor=IZGARA, zeroline=False)
    return fig


def zaman_cizelgesi(seritler: list[dict], degisimler=(), baslik: str | None = None):
    """Etkileşimli zaman çizelgesi (plotly Figure).

    seritler: [{"ad": "Sistem", "segmentler": [(bas, bit, etiket)], "renk": {etiket: renk},
               "adlar": {etiket: görünen ad}}] (üstten alta; "renk"/"adlar" isteğe bağlı).
    Her şeritte konuşmacı başına bir satır; üzerine gelince konuşmacı, başlangıç, bitiş, süre görünür.
    degisimler: dikey kesikli çizgi olarak çizilen değişim noktaları (sn). x ekseni yakınlaştırılabilir.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    satirlar = []
    coklu = len(seritler) > 1
    for s in seritler:
        renk = s.get("renk") or renkler(e for *_, e in s["segmentler"])[0]
        adlar = s.get("adlar") or {}
        for e in sorted({e for *_, e in s["segmentler"]}):
            ad = adlar.get(e, e)
            satir = f"{s['ad']} · {ad}" if coklu else ad
            satirlar.append(satir)
            segler = [(a, b) for a, b, l in s["segmentler"] if l == e]
            fig.add_trace(go.Bar(
                x=[b - a for a, b in segler], base=[a for a, _ in segler], y=[satir] * len(segler),
                orientation="h", marker=dict(color=renk.get(e, GRILER[0]), line=dict(width=0)), name=ad,
                customdata=[[ad, cikti._zaman(a), cikti._zaman(b), b - a, s["ad"]] for a, b in segler],
                hovertemplate=("<b>%{customdata[0]}</b><br>Başlangıç: %{customdata[1]}<br>Bitiş: %{customdata[2]}"
                               "<br>Süre: %{customdata[3]:.2f} sn<extra>%{customdata[4]}</extra>")))
        if not s["segmentler"]:
            satirlar.append(f"{s['ad']} · konuşma yok" if coklu else "konuşma yok")
            fig.add_trace(go.Bar(x=[0], base=[0], y=[satirlar[-1]], orientation="h", hoverinfo="skip"))
    for t in degisimler:
        fig.add_vline(x=t, line_dash="dash", line_width=1, line_color="rgba(128,128,128,0.75)")
    fig.update_layout(barmode="overlay", bargap=0.3, title=dict(text=baslik or "", x=0.01, font_size=14))
    fig.update_yaxes(categoryorder="array", categoryarray=satirlar, autorange="reversed", fixedrange=True)
    fig.update_xaxes(rangemode="tozero")
    return _duzen(fig, 70 + 30 * max(len(satirlar), 1) + (24 if baslik else 0))


def hata_cizelgesi(bolgeler, sure_sn: float | None = None):
    """hata_bolgeleri çıktısını tür başına bir satırda renkli bantlarla çizer (plotly Figure)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    satirlar = [HATA_ADLARI[t] for t in HATA_RENKLERI]
    for tur, renk in HATA_RENKLERI.items():
        segler = [(a, b) for a, b, t in bolgeler if t == tur]
        fig.add_trace(go.Bar(
            x=[b - a for a, b in segler] or [0], base=[a for a, _ in segler] or [0],
            y=[HATA_ADLARI[tur]] * max(len(segler), 1), orientation="h", marker=dict(color=renk, line=dict(width=0)),
            customdata=[[HATA_ADLARI[tur], cikti._zaman(a), cikti._zaman(b), b - a] for a, b in segler] or None,
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]} – %{customdata[2]}"
                           "<br>%{customdata[3]:.2f} sn<extra></extra>") if segler else None,
            hoverinfo=None if segler else "skip"))
    fig.update_layout(barmode="overlay", bargap=0.3)
    fig.update_yaxes(categoryorder="array", categoryarray=satirlar, autorange="reversed", fixedrange=True)
    fig.update_xaxes(rangemode="tozero", range=[0, sure_sn] if sure_sn else None)
    return _duzen(fig, 70 + 30 * len(satirlar))


def benzerlik_haritasi(etiketler, matris, adlar: dict | None = None):
    """Konuşmacıların ortalama gömmeleri arasındaki kosinüs benzerliği (plotly ısı haritası, -1..1)."""
    import plotly.graph_objects as go

    adlar = adlar or {}
    gorunen = [adlar.get(e, e) for e in etiketler]
    fig = go.Figure(go.Heatmap(z=matris, x=gorunen, y=gorunen, zmin=-1, zmax=1, colorscale="RdBu",
                               texttemplate="%{z:.2f}", hovertemplate="%{y} – %{x}: %{z:.3f}<extra></extra>",
                               colorbar=dict(thickness=12)))
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    fig.update_xaxes(fixedrange=True)
    return _duzen(fig, min(140 + 44 * len(etiketler), 640), x_baslik=None)
