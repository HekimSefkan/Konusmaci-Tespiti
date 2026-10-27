"""Kurulum kontrolu (YG-10): python tests/kurulum_kontrol.py [--hizli]

Importlari ve kucuk duman testlerini calistirir; sonunda "KURULUM: OK" yazar.
--hizli: SpeechBrain ECAPA modelinin indirilmesini ve calistirilmasini atlar.
Ek paket gerektirmez (yalnizca requirements.txt).
"""
import importlib
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows konsolunda Turkce karakter (FOG-13)

SR = 16000
MODULLER = ["numpy", "scipy", "sklearn", "librosa", "soundfile", "torch", "torchaudio",
            "speechbrain", "silero_vad", "webrtcvad", "noisereduce", "pyannote.metrics",
            "matplotlib"]


def sinyal(sure, genlik=0.1, seed=0):
    import numpy as np
    return (np.random.default_rng(seed).standard_normal(int(SR * sure)) * genlik).astype("float32")


def import_testi():
    for ad in MODULLER:
        importlib.import_module(ad)


def mfcc_testi():
    import librosa
    m = librosa.feature.mfcc(y=sinyal(1.0), sr=SR, n_mfcc=13, n_fft=512, win_length=400, hop_length=160)
    assert m.shape == (13, 101), m.shape


def silero_testi():
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    # onnx=True: varsayilan TorchScript modeli Windows'ta ASCII disi (ör. "Sınıf") yollarda acilmiyor.
    ts = get_speech_timestamps(torch.zeros(SR * 2), load_silero_vad(onnx=True), sampling_rate=SR)
    assert ts == [], ts  # sessiz sinyalde konusma bulunmamali


def webrtc_testi():
    import webrtcvad
    cerceve = bytes(2 * SR * 30 // 1000)  # 30 ms sessizlik, 16 bit
    assert webrtcvad.Vad(2).is_speech(cerceve, SR) is False


def ecapa_testi():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    onbellek = Path.home() / ".cache" / "konusmaci-tespiti" / "ecapa"
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=str(onbellek), run_opts={"device": "cpu"},
                                           local_strategy=LocalStrategy.COPY)  # Windows'ta symlink izni gerekmez
    t0 = time.time()
    with torch.no_grad():
        gomme = model.encode_batch(torch.from_numpy(sinyal(1.5, seed=1))[None, :])
    assert gomme.numel() == 192, gomme.shape
    return f"gomme boyutu 192, tek pencere {time.time() - t0:.2f} sn"


def der_testi():
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref, hip = Annotation(), Annotation()
    ref[Segment(0, 5)], ref[Segment(5, 10)] = "A", "B"
    hip[Segment(0, 5)], hip[Segment(5, 8)] = "x", "y"
    der = DiarizationErrorRate()(ref, hip)
    assert abs(der - 0.2) < 1e-6, der  # 10 sn referansin 2 sn'si kacirildi


def main():
    testler = [("import", import_testi), ("MFCC", mfcc_testi), ("Silero VAD", silero_testi),
               ("WebRTC VAD", webrtc_testi), ("DER (pyannote.metrics)", der_testi)]
    if "--hizli" not in sys.argv:
        testler.append(("SpeechBrain ECAPA", ecapa_testi))
    print(f"Python {sys.version.split()[0]}")
    for ad, fonk in testler:
        t0 = time.time()
        ek = fonk() or ""
        print(f"  [OK] {ad} ({time.time() - t0:.1f} sn) {ek}")
    print("KURULUM: OK")


if __name__ == "__main__":
    main()
