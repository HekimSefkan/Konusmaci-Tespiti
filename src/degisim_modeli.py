"""Gözetimli konuşmacı değişim modeli (S-7) ve E5 hattı. Sözleşme: docs/arayuz.md

Ardışık iki pencere (i, i+1) arasında konuşmacı değişimi olasılığını tahmin eder.
scikit-learn ve joblib fonksiyon içinde import edilir.

Etiket tanımı (eğitim):
  - Her pencerenin BASKIN konuşmacısı, referans RTTM'de pencereyle en çok
    örtüşen konuşmacıdır (aynı konuşmacının segmentleri birleştirilerek ölçülür).
  - Baskın konuşmacı pencerenin yarısından azını kaplıyorsa pencere BELİRSİZDİR
    (sessizlik, yoğun örtüşme, kısa konuşmacı geçişleri).
  - Çift etiketi: iki pencerenin baskın konuşmacısı farklıysa 1, aynıysa 0.
    Pencerelerden biri belirsizse çift eğitimden çıkarılır.
"""
import datetime
import math

import numpy as np

from . import cikti, kumeleme
from . import main as ana
from .degerlendirme import _degisim_eslestir

OZELLIK_SURUMU = "v1"
OZELLIKLER = (
    "ecapa_cos",           # cos(e_i, e_i+1)
    "ecapa_baglam_cos",    # cos(ort(e_i-k+1..e_i), ort(e_i+1..e_i+k))
    "ecapa_atlamali_cos",  # cos(e_i-1, e_i+2)
    "mfcc_cos",            # cos(m_i, m_i+1)
    "bosluk_sn",           # s_i+1 - b_i (negatif = örtüşme)
    "ayni_bolge",          # iki pencere aynı VAD bölgesinde mi (0/1)
)
BAGLAM_K = 3
BASKIN_ORAN = 0.5
TOLERANS_SN = 0.5
OLASILIK_IZGARASI = tuple(round(0.10 + 0.05 * i, 2) for i in range(17))  # 0,10 .. 0,90
MIN_ARALIK_IZGARASI = (0.5, 1.0, 1.5, 2.0)
SEED = 68

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Özellikler ve etiketler
# ---------------------------------------------------------------------------

def _normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), _EPS)


def _cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.sum(_normalize(A) * _normalize(B), axis=1)


def _bolge_indisleri(pencereler, bolgeler) -> np.ndarray:
    """Her pencerenin merkezini içeren VAD bölgesinin indisi (yoksa -1)."""
    sonuc = np.full(len(pencereler), -1, dtype=int)
    if not bolgeler:
        return sonuc
    baslar = np.array([a for a, _ in bolgeler])
    bitisler = np.array([b for _, b in bolgeler])
    merkez = np.array([(s + e) / 2 for s, e in pencereler])
    j = np.searchsorted(baslar, merkez, side="right") - 1
    gecerli = (j >= 0) & (merkez <= bitisler[np.clip(j, 0, None)])
    sonuc[gecerli] = j[gecerli]
    return sonuc


def ozellikler(pencereler, X_ecapa, X_mfcc, bolgeler, k: int = BAGLAM_K) -> np.ndarray:
    """Ardışık pencere çiftlerinin özellik matrisi, şekil (N-1, len(OZELLIKLER)), float32.

    Sütun sırası OZELLIKLER ile aynıdır. N < 2 ise (0, F).
    """
    n = len(pencereler)
    F = len(OZELLIKLER)
    if n < 2:
        return np.zeros((0, F), dtype=np.float32)
    E = _normalize(X_ecapa)
    i = np.arange(n - 1)

    # Bağlam: kümülatif toplamla sol [i-k+1..i] ve sağ [i+1..i+k] ortalamaları
    kum = np.vstack([np.zeros((1, E.shape[1])), np.cumsum(E, axis=0)])
    sol_bas = np.maximum(i - k + 1, 0)
    sag_bit = np.minimum(i + 1 + k, n)
    sol = (kum[i + 1] - kum[sol_bas]) / (i + 1 - sol_bas)[:, None]
    sag = (kum[sag_bit] - kum[i + 1]) / (sag_bit - (i + 1))[:, None]

    bas = np.array([s for s, _ in pencereler])
    bit = np.array([e for _, e in pencereler])
    bolge = _bolge_indisleri(pencereler, bolgeler)

    M = np.zeros((n - 1, F), dtype=np.float64)
    M[:, 0] = np.sum(E[i] * E[i + 1], axis=1)
    M[:, 1] = _cos(sol, sag)
    M[:, 2] = np.sum(E[np.maximum(i - 1, 0)] * E[np.minimum(i + 2, n - 1)], axis=1)
    M[:, 3] = _cos(np.asarray(X_mfcc)[i], np.asarray(X_mfcc)[i + 1])
    M[:, 4] = bas[i + 1] - bit[i]
    M[:, 5] = ((bolge[i] == bolge[i + 1]) & (bolge[i] >= 0)).astype(float)
    return M.astype(np.float32)


def cift_zamanlari(pencereler) -> np.ndarray:
    """Her çiftin olası değişim zamanı: örtüşme varsa örtüşmenin, boşluk varsa boşluğun ortası."""
    if len(pencereler) < 2:
        return np.zeros(0)
    bas = np.array([s for s, _ in pencereler])
    bit = np.array([e for _, e in pencereler])
    return (bit[:-1] + bas[1:]) / 2


def baskin_konusmacilar(pencereler, ref_segmentler, oran: float = BASKIN_ORAN) -> list:
    """Her pencerenin baskın referans konuşmacısı; belirsiz pencere için None.

    Baskın konuşmacı = pencereyle en çok örtüşen konuşmacı (eşitlikte alfabetik ilk).
    Kapladığı süre pencere uzunluğunun `oran` katından azsa pencere belirsizdir.
    """
    n = len(pencereler)
    if n == 0:
        return []
    ws = np.array([s for s, _ in pencereler], dtype=float)
    we = np.array([e for _, e in pencereler], dtype=float)
    uzunluk = we - ws

    # Aynı konuşmacının üst üste binen segmentleri tek sayılsın: konuşmacı başına birleştir.
    konusmaci: dict[str, list[list[float]]] = {}
    for a, b, e in sorted(ref_segmentler):
        araliklar = konusmaci.setdefault(e, [])
        if araliklar and a <= araliklar[-1][1]:
            araliklar[-1][1] = max(araliklar[-1][1], b)
        else:
            araliklar.append([a, b])

    adlar = sorted(konusmaci)
    if not adlar:
        return [None] * n
    kapsama = np.zeros((n, len(adlar)))
    for j, ad in enumerate(adlar):
        ar = np.array(konusmaci[ad])
        ortusme = np.minimum(we[:, None], ar[None, :, 1]) - np.maximum(ws[:, None], ar[None, :, 0])
        kapsama[:, j] = np.clip(ortusme, 0, None).sum(axis=1)
    en_iyi = np.argmax(kapsama, axis=1)  # eşitlikte ilk indis = alfabetik ilk
    sonuc = []
    for i in range(n):
        if uzunluk[i] > 0 and kapsama[i, en_iyi[i]] >= oran * uzunluk[i] - 1e-9:
            sonuc.append(adlar[en_iyi[i]])
        else:
            sonuc.append(None)
    return sonuc


def cift_etiketleri(baskin: list) -> tuple[np.ndarray, np.ndarray]:
    """(y, maske): y[i] = baskın(i) != baskın(i+1); maske[i] = iki pencere de belirli."""
    if len(baskin) < 2:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=bool)
    a, b = baskin[:-1], baskin[1:]
    maske = np.array([x is not None and y is not None for x, y in zip(a, b)])
    y = np.array([int(x != y) for x, y in zip(a, b)])
    return y, maske


def tepeler(olasiliklar, zamanlar, esik: float, min_aralik_sn: float) -> list[float]:
    """Değişim noktaları: eşiği geçen yerel tepeler; en yüksek olasılık önce seçilir,
    seçilene min_aralik_sn'den yakın tepeler bastırılır. Dönüş: artan zamanlar (sn)."""
    p = np.asarray(olasiliklar, dtype=float)
    z = np.asarray(zamanlar, dtype=float)
    n = len(p)
    if n == 0:
        return []
    sol = np.concatenate([[-np.inf], p[:-1]])
    sag = np.concatenate([p[1:], [-np.inf]])
    adaylar = np.where((p >= esik) & (p >= sol) & (p >= sag))[0]
    secilen: list[float] = []
    for i in sorted(adaylar, key=lambda i: (-p[i], z[i])):
        if all(abs(z[i] - t) >= min_aralik_sn - 1e-9 for t in secilen):
            secilen.append(float(z[i]))
    return sorted(secilen)


def kayit_verisi(kayit, pencereler, X_ecapa, X_mfcc, bolgeler, ref_segmentler, k: int = BAGLAM_K) -> dict:
    """Bir kaydın eğitim verisi: F (özellikler), y, maske, zamanlar, referans değişimleri."""
    y, maske = cift_etiketleri(baskin_konusmacilar(pencereler, ref_segmentler))
    return {
        "kayit": kayit,
        "F": ozellikler(pencereler, X_ecapa, X_mfcc, bolgeler, k),
        "y": y,
        "maske": maske,
        "zamanlar": cift_zamanlari(pencereler),
        "ref_degisim": cikti.degisim_noktalari(ref_segmentler),
    }


# ---------------------------------------------------------------------------
# Eğitim
# ---------------------------------------------------------------------------

def _adaylar(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    adaylar = []
    for C in (0.1, 1.0, 10.0):
        adaylar.append((f"lojistik_C{C:g}", lambda C=C: make_pipeline(
            StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))))
    for gizli in ((32,), (64, 32)):
        for alpha in (1e-4, 1e-3):
            ad = f"mlp_{'x'.join(map(str, gizli))}_a{alpha:g}"
            adaylar.append((ad, lambda g=gizli, a=alpha: make_pipeline(
                StandardScaler(), MLPClassifier(hidden_layer_sizes=g, alpha=a, early_stopping=True,
                                                max_iter=300, random_state=seed))))
    return adaylar


def _izgara_uyarilari(olasilik_esigi: float, min_aralik_sn: float) -> list[str]:
    """Seçilen tepe ayarı ızgaranın kenarındaysa uyarı metinleri (eşik taramasındaki mantık)."""
    uyarilar = []
    if olasilik_esigi in (OLASILIK_IZGARASI[0], OLASILIK_IZGARASI[-1]):
        uyarilar.append(f"olasilik_esigi: en iyi değer ızgaranın kenarında ({olasilik_esigi})")
    if min_aralik_sn in (MIN_ARALIK_IZGARASI[0], MIN_ARALIK_IZGARASI[-1]):
        uyarilar.append(f"min_aralik_sn: en iyi değer ızgaranın kenarında ({min_aralik_sn})")
    return uyarilar


def _mikro_f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def egit(veriler: list[dict], seed: int = SEED, n_kat: int = 5, log=print) -> dict:
    """Modeli eğitir, kayıt bazında gruplu çapraz doğrulamayla seçer ve tepe ayarlarını belirler.

    veriler: kayit_verisi çıktıları (yalnızca eğitim yarısı).
    Seçim ölçütü: katlar üzerinde ortalama kesinlik (average precision).
    Tepe ayarları: seçilen modelin kat dışı (out-of-fold) olasılıklarıyla, ±TOLERANS_SN
    toleranslı mikro değişim F1'ini en yüksek yapan (olasılık eşiği, min aralık).
    Dönüş: kaydedilecek paket (sözlük).
    Hata: ValueError (yetersiz veri: < 2 kayıt veya tek sınıf).
    """
    import warnings

    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold

    kullanilan = [v for v in veriler if v["maske"].any()]
    if len(kullanilan) < 2:
        raise ValueError("eğitim için etiketli çifti olan en az 2 kayıt gerekir")
    X = np.vstack([v["F"][v["maske"]] for v in kullanilan])
    Y = np.concatenate([v["y"][v["maske"]] for v in kullanilan])
    G = np.concatenate([np.full(int(v["maske"].sum()), i) for i, v in enumerate(kullanilan)])
    if len(set(Y.tolist())) < 2:
        raise ValueError("eğitim verisinde yalnızca tek sınıf var")
    kat = min(n_kat, len(kullanilan))
    bolucu = list(GroupKFold(n_splits=kat).split(X, Y, G))
    log(f"Eğitim: {len(kullanilan)} kayıt, {len(Y)} çift (değişim oranı {Y.mean():.3f}), {kat} kat")

    cv: dict[str, dict] = {}
    oof: dict[str, dict[str, np.ndarray]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for ad, kur in _adaylar(seed):
            skorlar = []
            oof[ad] = {}
            for egitim_idx, test_idx in bolucu:
                model = kur().fit(X[egitim_idx], Y[egitim_idx])
                if len(set(Y[test_idx].tolist())) == 2:
                    skorlar.append(float(average_precision_score(
                        Y[test_idx], model.predict_proba(X[test_idx])[:, 1])))
                for g in sorted(set(G[test_idx].tolist())):  # bu kattaki kayıtların TÜM çiftleri
                    v = kullanilan[g]
                    oof[ad][v["kayit"]] = model.predict_proba(v["F"])[:, 1]
            cv[ad] = {"ap_ort": float(np.mean(skorlar)) if skorlar else math.nan,
                      "ap_std": float(np.std(skorlar)) if skorlar else math.nan,
                      "katlar": skorlar}
            log(f"  {ad:<22} AP = {cv[ad]['ap_ort']:.4f} ± {cv[ad]['ap_std']:.4f}")

        secilen = max(cv, key=lambda a: (np.nan_to_num(cv[a]["ap_ort"], nan=-1), -list(cv).index(a)))
        kur = dict(_adaylar(seed))[secilen]
        model = kur().fit(X, Y)
    log(f"Seçilen model: {secilen}")

    # Tepe ayarları: kat dışı olasılıklar (eğitimde görülmemiş kayıtlar gibi)
    olas = {}
    for v in veriler:
        if len(v["F"]) == 0:
            continue
        olas[v["kayit"]] = oof[secilen].get(v["kayit"])
        if olas[v["kayit"]] is None:  # etiketli çifti olmayan kayıt: son model
            olas[v["kayit"]] = model.predict_proba(v["F"])[:, 1]
    tarama = []
    for esik in OLASILIK_IZGARASI:
        for aralik in MIN_ARALIK_IZGARASI:
            tp = fp = fn = 0
            for v in veriler:
                hyp = tepeler(olas[v["kayit"]], v["zamanlar"], esik, aralik) if v["kayit"] in olas else []
                a, b, c = _degisim_eslestir(v["ref_degisim"], hyp, TOLERANS_SN)
                tp, fp, fn = tp + a, fp + b, fn + c
            tarama.append({"olasilik_esigi": esik, "min_aralik_sn": aralik,
                           "tp": tp, "fp": fp, "fn": fn, "mikro_f1": _mikro_f1(tp, fp, fn)})
    en_iyi = max(tarama, key=lambda r: (r["mikro_f1"], r["olasilik_esigi"], r["min_aralik_sn"]))
    log(f"Tepe ayarı: olasılık eşiği {en_iyi['olasilik_esigi']:.2f}, min aralık "
        f"{en_iyi['min_aralik_sn']:.1f} sn -> eğitim yarısı mikro F1 {en_iyi['mikro_f1']:.4f}")
    uyarilar = _izgara_uyarilari(en_iyi["olasilik_esigi"], en_iyi["min_aralik_sn"])
    for u in uyarilar:
        log(f"UYARI: {u}")

    return {
        "model": model,
        "secilen_model": secilen,
        "ozellik_surumu": OZELLIK_SURUMU,
        "ozellikler": list(OZELLIKLER),
        "baglam_k": BAGLAM_K,
        "baskin_oran": BASKIN_ORAN,
        "olasilik_esigi": en_iyi["olasilik_esigi"],
        "min_aralik_sn": en_iyi["min_aralik_sn"],
        "tolerans_sn": TOLERANS_SN,
        "egitim_f1": en_iyi["mikro_f1"],
        "tepe_tarama": tarama,
        "uyarilar": uyarilar,
        "cv": cv,
        "egitim_tarihi": datetime.datetime.now().isoformat(timespec="seconds"),
        "egitim_kayit_sayisi": len(veriler),
        "egitim_cift_sayisi": int(len(Y)),
        "degisim_orani": float(Y.mean()),
        "sklearn_surumu": sklearn.__version__,
        "seed": seed,
    }


def kaydet(paket: dict, yol: str) -> None:
    """Paketi joblib ile tek dosyaya yazar (klasör yoksa oluşturur)."""
    import os

    import joblib

    klasor = os.path.dirname(yol)
    if klasor:
        os.makedirs(klasor, exist_ok=True)
    gecici = yol + ".yaziliyor"
    joblib.dump(paket, gecici)
    os.replace(gecici, yol)


def yukle(yol: str) -> dict:
    """Paketi yükler. Hata: FileNotFoundError, ValueError (özellik sürümü uyuşmuyor)."""
    import joblib

    paket = joblib.load(yol)
    if not isinstance(paket, dict) or paket.get("ozellik_surumu") != OZELLIK_SURUMU:
        surum = paket.get("ozellik_surumu") if isinstance(paket, dict) else None
        raise ValueError(f"{yol}: özellik sürümü {surum!r}, beklenen {OZELLIK_SURUMU!r}")
    return paket


# ---------------------------------------------------------------------------
# Tahmin ve E5 hattı
# ---------------------------------------------------------------------------

def degisim_tahmin(paket: dict, pencereler, X_ecapa, X_mfcc, bolgeler) -> list[float]:
    """Modelin değişim noktaları (sn, artan)."""
    if len(pencereler) < 2:
        return []
    F = ozellikler(pencereler, X_ecapa, X_mfcc, bolgeler, paket["baglam_k"])
    p = paket["model"].predict_proba(F)[:, 1]
    return tepeler(p, cift_zamanlari(pencereler), paket["olasilik_esigi"], paket["min_aralik_sn"])


def bolgeleri_kes(bolgeler, degisimler) -> list[tuple[float, float]]:
    """VAD bölgelerini, içlerine düşen değişim noktalarından parçalara böler (sıralı)."""
    noktalar = sorted(degisimler)
    parcalar = []
    for a, b in sorted(bolgeler):
        kesimler = [t for t in noktalar if a + 1e-6 < t < b - 1e-6]
        sinirlar = [a] + kesimler + [b]
        parcalar += [(sinirlar[j], sinirlar[j + 1]) for j in range(len(sinirlar) - 1)]
    return parcalar


def e5_segmentleri(paket: dict, bolgeler, pencereler, X_ecapa, X_mfcc, esik: float | None = None):
    """E5: değişim noktalarından segmentler -> segment ECAPA ortalaması -> AHC (eşikle).

    Her parçanın vektörü, merkezi parçanın içine düşen pencerelerin (L2 normalize)
    ECAPA ortalamasıdır; hiç pencere yoksa merkeze en yakın pencere kullanılır.
    Kısa konuşma ve sessizlik kuralları main.hat_segmentleri ile aynıdır.
    """
    if not bolgeler or ana.kisa_konusma(bolgeler) or len(pencereler) == 0:
        return ana.hat_segmentleri(bolgeler, [], None)
    parcalar = bolgeleri_kes(bolgeler, degisim_tahmin(paket, pencereler, X_ecapa, X_mfcc, bolgeler))
    E = _normalize(X_ecapa)
    merkez = np.array([(s + e) / 2 for s, e in pencereler])
    V = np.zeros((len(parcalar), E.shape[1]), dtype=np.float32)
    for j, (a, b) in enumerate(parcalar):
        icerde = np.where((merkez >= a) & (merkez < b))[0]
        if len(icerde) == 0:
            icerde = [int(np.argmin(np.abs(merkez - (a + b) / 2)))]
        V[j] = E[icerde].mean(axis=0)
    etiketler = kumeleme.kumele(V, None, esik)
    return cikti.segmentleri_birlestir(parcalar, etiketler)
