# -*- coding: utf-8 -*-
"""
gurultu_ekle.py - Temiz bir WAV dosyasına belirli SNR değerlerinde beyaz
gürültü ekleyerek "yapay gürültü eklenmiş test dosyası" üretir.

Grup_68 - "Ses Kayıtlarında Konuşmacı Değişimlerinin Tespiti" projesi.
Yalnızca Python standart kütüphanesini kullanır (Python 3.8+).

Not: Üretilen dosyalar GERÇEK gürültülü ortam kaydı değildir; gürültü
sonradan yapay olarak eklenmiştir.

Yöntem:
  - Girdi: 16-bit PCM, mono WAV.
  - Sinyal gücü Ps, dosyadaki TÜM örneklerin ortalama karesidir (sessiz
    kısımlar dahil).
  - Hedef SNR (dB) için gürültü gücü Pn = Ps / 10^(SNR/10); gürültü,
    ortalaması 0 ve standart sapması sqrt(Pn) olan Gauss (beyaz) gürültüdür.
  - Sabit seed kullanıldığı için aynı girdiyle her çalıştırmada aynı çıktı
    elde edilir. Taşan örnekler 16-bit aralığına kırpılır.
  - Konuşma zamanları değişmediğinden kaynak RTTM aynen geçerlidir;
    --rttm verilirse 2. alanı (dosya adı) yeni adla değiştirilmiş bir
    kopyası da yazılır.

Kullanım örneği:
  python gurultu_ekle.py --wav <ikgcq_028.0-118.0.wav> --rttm <ikgcq_028.0-118.0.rttm>
      --snr 10 5 --wav-cikti <ses_klasoru> --rttm-cikti <rttm_klasoru>
"""

import argparse
import array
import math
import os
import random
import sys
import wave


def bos_ad(klasor, ad, uzanti):
    """Var olan dosyanın üzerine yazmamak için boş bir dosya adı bulur."""
    yol = os.path.join(klasor, ad + uzanti)
    n = 2
    while os.path.exists(yol):
        yol = os.path.join(klasor, f"{ad}_v{n}{uzanti}")
        n += 1
    return yol


def wav_oku(yol):
    with wave.open(yol, "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise SystemExit("Yalnızca 16-bit PCM mono WAV desteklenir.")
        ornekler = array.array("h", w.readframes(w.getnframes()))
        if sys.byteorder == "big":
            ornekler.byteswap()
        return ornekler, w.getframerate()


def wav_yaz(yol, ornekler, hiz):
    veri = array.array("h", ornekler)
    if sys.byteorder == "big":
        veri.byteswap()
    with wave.open(yol, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(hiz)
        w.writeframes(veri.tobytes())


def gurultu_ekle(ornekler, snr_db, seed):
    ps = sum(x * x for x in ornekler) / len(ornekler)
    sigma = math.sqrt(ps / (10 ** (snr_db / 10)))
    rng = random.Random(seed)
    return [max(-32768, min(32767, int(round(x + rng.gauss(0.0, sigma)))))
            for x in ornekler]


def main():
    ap = argparse.ArgumentParser(description="WAV dosyasına belirli SNR'da beyaz gürültü ekler.")
    ap.add_argument("--wav", required=True, help="Temiz kaynak WAV (16-bit mono)")
    ap.add_argument("--rttm", help="Kaynak RTTM (isteğe bağlı; yeni adla kopyalanır)")
    ap.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0], help="SNR (dB) listesi")
    ap.add_argument("--seed", type=int, default=68, help="Rastgele sayı tohumu (varsayılan 68)")
    ap.add_argument("--wav-cikti", required=True, help="Çıktı WAV klasörü")
    ap.add_argument("--rttm-cikti", help="Çıktı RTTM klasörü")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ornekler, hiz = wav_oku(args.wav)
    kok = os.path.splitext(os.path.basename(args.wav))[0]
    os.makedirs(args.wav_cikti, exist_ok=True)

    for snr in args.snr:
        wav_yolu = bos_ad(args.wav_cikti, f"{kok}_yapay_gurultu_snr{snr:g}dB", ".wav")
        ad = os.path.splitext(os.path.basename(wav_yolu))[0]
        wav_yaz(wav_yolu, gurultu_ekle(ornekler, snr, args.seed), hiz)
        print(f"WAV : {wav_yolu} (SNR {snr:g} dB, seed {args.seed})")
        if args.rttm and args.rttm_cikti:
            os.makedirs(args.rttm_cikti, exist_ok=True)
            rttm_yolu = bos_ad(args.rttm_cikti, ad, ".rttm")
            with open(args.rttm, encoding="utf-8") as f, \
                    open(rttm_yolu, "w", encoding="utf-8", newline="\n") as g:
                for satir in f:
                    al = satir.split()
                    if al and al[0] == "SPEAKER":
                        al[1] = ad
                        g.write(" ".join(al) + "\n")
            print(f"RTTM: {rttm_yolu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
