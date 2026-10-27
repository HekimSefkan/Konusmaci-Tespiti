"""Konuşma etkinliği tespiti (VAD). Sahibi: Abdullah. Sözleşme: docs/arayuz.md

silero-vad ve torch fonksiyon içinde import edilir; model bir kez yüklenir.
"""
import numpy as np

MIN_KONUSMA_MS = 250
MIN_SESSIZLIK_MS = 300

_model = None


def _silero_modeli():
    """Silero VAD modelini (ONNX) bir kez yükleyip saklar.

    onnx=True: Windows'ta ASCII dışı karakterli yollarda varsayılan model açılamıyor.
    """
    global _model
    if _model is None:
        from silero_vad import load_silero_vad

        _model = load_silero_vad(onnx=True)
    return _model


def konusma_bolgeleri(ses: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Konuşma olan bölgeleri bulur (Silero VAD; min. konuşma 250 ms, min. sessizlik 300 ms).

    Girdi: ses (float32, (n_ornek,), [-1, 1]); sr (Hz, 16000).
    Çıktı: [(baslangic_sn, bitis_sn), ...] saniye cinsinden, zamana göre sıralı,
           birbiriyle örtüşmeyen. Konuşma yoksa boş liste.
    """
    ses = np.ascontiguousarray(ses, dtype=np.float32)
    if ses.size == 0:
        return []

    import torch
    from silero_vad import get_speech_timestamps

    zamanlar = get_speech_timestamps(
        torch.from_numpy(ses),
        _silero_modeli(),
        sampling_rate=sr,
        min_speech_duration_ms=MIN_KONUSMA_MS,
        min_silence_duration_ms=MIN_SESSIZLIK_MS,
    )  # örnek indisleri; saniyeye kendimiz çeviriyoruz (yuvarlama kaybı olmasın)

    bolgeler: list[tuple[float, float]] = []
    for t in sorted(zamanlar, key=lambda z: z["start"]):
        a, b = round(t["start"] / sr, 3), round(t["end"] / sr, 3)
        if b <= a:
            continue
        if bolgeler and a <= bolgeler[-1][1]:  # örtüşme/bitişiklik: birleştir
            bolgeler[-1] = (bolgeler[-1][0], max(bolgeler[-1][1], b))
        else:
            bolgeler.append((a, b))
    return bolgeler
