"""Testler için sentetik kayıt üretici: ses yok, yalnızca RTTM ve rastgele vektörler.

Her konuşmacının rastgele bir ECAPA (192) ve MFCC (26) merkezi vardır; pencere
vektörü, penceredeki konuşmacıların kapsama oranıyla ağırlıklı merkez + gürültüdür.
"""
import numpy as np

from src import cikti, oznitelik, toplu


def kayit_uret(tohum: int, konusmacilar=("A", "B", "C"), sure_sn: float = 60.0, sessizlik=(30.0, 31.0),
               gurultu: float = 0.15) -> dict:
    rng = np.random.default_rng(tohum)
    merkez_e = {k: rng.normal(size=192) for k in konusmacilar}
    merkez_m = {k: rng.normal(size=26) for k in konusmacilar}

    # Referans: rastgele uzunlukta (4-10 sn) konuşma sıraları, ortada bir sessizlik
    ref, t, onceki = [], 0.0, None
    while t < sure_sn - 1e-9:
        aday = [k for k in konusmacilar if k != onceki] or list(konusmacilar)
        k = aday[int(rng.integers(len(aday)))]
        b = min(round(t + float(rng.uniform(4, 10)), 3), sure_sn)
        ref.append((round(t, 3), b, k))
        t, onceki = b, k
    if sessizlik:
        s0, s1 = sessizlik
        kesik = []
        for a, b, k in ref:
            if b <= s0 or a >= s1:
                kesik.append((a, b, k))
            else:
                if a < s0:
                    kesik.append((a, s0, k))
                if b > s1:
                    kesik.append((s1, b, k))
        ref = kesik
    bolgeler = []
    for a, b, _ in ref:
        if bolgeler and a <= bolgeler[-1][1] + 1e-9:
            bolgeler[-1] = (bolgeler[-1][0], b)
        else:
            bolgeler.append((a, b))
    pencereler = oznitelik.pencerele(bolgeler)

    Xe = np.zeros((len(pencereler), 192))
    Xm = np.zeros((len(pencereler), 26))
    for i, (s, e) in enumerate(pencereler):
        for a, b, k in ref:
            kapsama = max(0.0, min(e, b) - max(s, a)) / (e - s)
            Xe[i] += kapsama * merkez_e[k]
            Xm[i] += kapsama * merkez_m[k]
        Xe[i] += gurultu * rng.normal(size=192) * np.linalg.norm(Xe[i] + 1e-9) / np.sqrt(192)
        Xm[i] += gurultu * rng.normal(size=26) * np.linalg.norm(Xm[i] + 1e-9) / np.sqrt(26)
    return {"ref": ref, "bolgeler": bolgeler, "pencereler": pencereler,
            "X_ecapa": Xe.astype(np.float32), "X_mfcc": Xm.astype(np.float32), "ses_sn": sure_sn}


def ortam_kur(kok, kayitlar: dict[str, dict]) -> tuple[str, str]:
    """tmp klasöründe referans RTTM'leri ve dolu önbelleği yazar; (rttm_dizin, onbellek) döner."""
    rttm_dizin = kok / "rttm"
    rttm_dizin.mkdir(parents=True, exist_ok=True)
    onbellek = str(kok / "onbellek")
    for kayit, v in kayitlar.items():
        cikti.rttm_yaz(v["ref"], kayit, str(rttm_dizin / f"{kayit}.rttm"))
        P = np.array(v["pencereler"], dtype=np.float64).reshape(-1, 2)
        toplu._npz_yaz(toplu.onbellek_yolu(onbellek, "vad", kayit), "vad",
                       bolgeler=np.array(v["bolgeler"]).reshape(-1, 2), ses_sn=np.float64(v["ses_sn"]),
                       sure_sn=np.float64(0.5))
        toplu._npz_yaz(toplu.onbellek_yolu(onbellek, "mfcc", kayit), "mfcc",
                       pencereler=P, X=v["X_mfcc"], sure_sn=np.float64(0.2))
        toplu._npz_yaz(toplu.onbellek_yolu(onbellek, "ecapa", kayit), "ecapa",
                       pencereler=P, X=v["X_ecapa"], sure_sn=np.float64(1.0))
    return str(rttm_dizin), onbellek
