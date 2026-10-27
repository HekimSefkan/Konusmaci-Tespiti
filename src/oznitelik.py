"""Segmentleme ve öznitelik/gömme çıkarımı. Sahibi: Abdullah. Sözleşme: docs/arayuz.md

librosa / torch / speechbrain fonksiyon içinde import edilir; pencerele saf Python'dur.
"""
import os

import numpy as np

BOYUT = {"mfcc": 26, "ecapa": 192}

# MFCC ayarları (16 kHz'de 32 ms FFT, 25 ms pencere, 10 ms adım)
N_MFCC = 13
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160

# ECAPA ayarları
ECAPA_KAYNAK = "speechbrain/spkrec-ecapa-voxceleb"
ONBELLEK_ORTAM_DEGISKENI = "KT_MODEL_ONBELLEK"
VARSAYILAN_ONBELLEK = os.path.join(os.path.expanduser("~"), ".cache", "konusmaci-tespiti", "ecapa")
ECAPA_TOPLU = 32  # bir seferde işlenen pencere sayısı

_ecapa = None


def pencerele(
    bolgeler: list[tuple[float, float]],
    pencere_sn: float = 1.5,
    kaydirma_sn: float = 0.75,
) -> list[tuple[float, float]]:
    """Konuşma bölgelerini örtüşen sabit uzunlukta pencerelere böler.

    Girdi: bolgeler [(baslangic_sn, bitis_sn)]; pencere_sn ve kaydirma_sn saniye.
    Çıktı: [(baslangic_sn, bitis_sn)], zamana göre sıralı. Pencereler bölge
           sınırını aşmaz; pencere_sn'den kısa bölge tek pencere olur.
           Bölgenin sonu açıkta kalırsa son pencere bölge sonuna hizalanır.
           Boş girdi için boş liste.
    Hata: ValueError (pencere_sn <= 0 veya kaydirma_sn <= 0).
    """
    if pencere_sn <= 0 or kaydirma_sn <= 0:
        raise ValueError(f"pencere_sn ve kaydirma_sn pozitif olmalı ({pencere_sn}, {kaydirma_sn})")
    eps = 1e-9
    pencereler: list[tuple[float, float]] = []
    for a, b in sorted(bolgeler):
        a, b = float(a), float(b)
        if b <= a:
            continue
        if b - a <= pencere_sn + eps:
            pencereler.append((round(a, 6), round(b, 6)))
            continue
        k = 0
        while True:
            s = a + k * kaydirma_sn
            if s + pencere_sn > b + eps:
                break
            pencereler.append((round(s, 6), round(s + pencere_sn, 6)))
            k += 1
        if pencereler[-1][1] < b - 1e-6:  # kalan kuyruk: sona hizalı son pencere
            pencereler.append((round(b - pencere_sn, 6), round(b, 6)))
    return pencereler


def ecapa_modeli(onbellek: str | None = None):
    """SpeechBrain ECAPA modelini bir kez yükleyip saklar ve döndürür.

    onbellek: model dosyalarının klasörü. Verilmezse KT_MODEL_ONBELLEK ortam
    değişkeni, o da yoksa ~/.cache/konusmaci-tespiti/ecapa. GPU varsa CUDA
    kullanılır. Model zaten yüklüyse onbellek yok sayılır.
    """
    global _ecapa
    if _ecapa is None:
        import torch

        from .uyumluluk import speechbrain_hazirla

        speechbrain_hazirla()  # torchaudio >= 2.9 ile speechbrain 1.0.3 içe aktarılabilsin
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        klasor = onbellek or os.environ.get(ONBELLEK_ORTAM_DEGISKENI) or VARSAYILAN_ONBELLEK
        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        _ecapa = EncoderClassifier.from_hparams(
            source=ECAPA_KAYNAK,
            savedir=klasor,
            run_opts={"device": cihaz},
            local_strategy=LocalStrategy.COPY,  # Windows'ta sembolik bağ izni gerekmez
        )
        _ecapa.eval()
    return _ecapa


def _parcalar(ses: np.ndarray, sr: int, pencereler) -> list[np.ndarray]:
    n = len(ses)
    sonuc = []
    for a, b in pencereler:
        i0 = min(max(int(round(a * sr)), 0), n)
        i1 = min(max(int(round(b * sr)), i0), n)
        sonuc.append(ses[i0:i1])
    return sonuc


def _mfcc(parcalar: list[np.ndarray], sr: int) -> np.ndarray:
    import librosa

    X = np.zeros((len(parcalar), BOYUT["mfcc"]), dtype=np.float32)
    for i, p in enumerate(parcalar):
        if p.size == 0:
            continue
        m = librosa.feature.mfcc(
            y=np.asarray(p, dtype=np.float32), sr=sr, n_mfcc=N_MFCC + 1,
            n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        )[1:]  # c0 (enerji) atılır: c1..c13
        X[i, :N_MFCC] = m.mean(axis=1)
        X[i, N_MFCC:] = m.std(axis=1)
    return X


def _ecapa_gomme(parcalar: list[np.ndarray]) -> np.ndarray:
    import torch

    model = ecapa_modeli()
    cihaz = next(model.mods.parameters()).device
    X = np.zeros((len(parcalar), BOYUT["ecapa"]), dtype=np.float32)
    # Benzer uzunlukları bir araya getir: daha az dolgu.
    sira = sorted(range(len(parcalar)), key=lambda i: len(parcalar[i]))
    for k in range(0, len(sira), ECAPA_TOPLU):
        idx = [i for i in sira[k:k + ECAPA_TOPLU] if parcalar[i].size > 0]
        if not idx:
            continue
        en_uzun = max(len(parcalar[i]) for i in idx)
        dalga = torch.zeros(len(idx), en_uzun)
        uzunluk = torch.zeros(len(idx))
        for j, i in enumerate(idx):
            p = torch.from_numpy(np.asarray(parcalar[i], dtype=np.float32))
            dalga[j, :len(p)] = p
            uzunluk[j] = len(p) / en_uzun  # SpeechBrain göreli uzunluk ister
        with torch.no_grad():
            g = model.encode_batch(dalga.to(cihaz), uzunluk.to(cihaz))  # (B, 1, 192)
        X[idx] = g.squeeze(1).cpu().numpy().astype(np.float32)
    return X


def olcek_hesapla(X_listesi) -> dict:
    """Genel (tüm kayıtlar üzerinden) boyut başına ortalama ve standart sapma.

    X_listesi: (N_i, D) matrisleri (ör. eğitim yarısının tüm MFCC pencereleri).
    Dönüş: {"ortalama": [D], "std": [D], "pencere_sayisi": int}; std alt sınırı 1e-8.
    Kayıt başına ortalama çıkarılmaz (tek konuşmacılı kayıtları bölebilir).
    Hata: ValueError (hiç pencere yok).
    """
    dolu = [np.asarray(X, dtype=np.float64) for X in X_listesi if len(X)]
    if not dolu:
        raise ValueError("ölçek için hiç pencere yok")
    T = np.vstack(dolu)
    return {"ortalama": T.mean(axis=0).tolist(),
            "std": np.maximum(T.std(axis=0), 1e-8).tolist(),
            "pencere_sayisi": int(T.shape[0])}


def olcekle(X: np.ndarray, olcek: dict | None) -> np.ndarray:
    """X'i olcek_hesapla istatistikleriyle standartlaştırır (olcek None ise X değişmeden döner)."""
    X = np.asarray(X, dtype=np.float32)
    if olcek is None or len(X) == 0:
        return X
    return ((X - np.asarray(olcek["ortalama"], dtype=np.float32))
            / np.asarray(olcek["std"], dtype=np.float32)).astype(np.float32)


def vektorler(
    ses: np.ndarray,
    sr: int,
    pencereler: list[tuple[float, float]],
    yontem: str = "mfcc",
) -> np.ndarray:
    """Her pencere için bir vektör üretir.

    Girdi: ses (float32, (n_ornek,)); sr (16000); pencereler [(bas_sn, bit_sn)];
           yontem "mfcc" (c1-c13 ortalama+std, c0 atılır, D=26) veya
           "ecapa" (SpeechBrain ECAPA-TDNN, D=192).
    Çıktı: np.ndarray, şekil (N, D), float32; N = len(pencereler).
           Boş pencere listesinde şekil (0, D).
    Hata: ValueError (bilinmeyen yontem).
    """
    if yontem not in BOYUT:
        raise ValueError(f"bilinmeyen yontem '{yontem}' (mfcc veya ecapa)")
    if not pencereler:
        return np.zeros((0, BOYUT[yontem]), dtype=np.float32)
    parcalar = _parcalar(np.asarray(ses, dtype=np.float32), sr, pencereler)
    if yontem == "mfcc":
        return _mfcc(parcalar, sr)
    return _ecapa_gomme(parcalar)
