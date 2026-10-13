# -*- coding: utf-8 -*-
"""
edge_uret.py - Kenar durum (edge case) test dosyalarını üretir.

Grup_68 - "Ses Kayıtlarında Konuşmacı Değişimlerinin Tespiti" projesi.
Yalnızca Python standart kütüphanesini kullanır (Python 3.8+).

Üretilen dosyalar:
  (a) bozuk_yarim.wav        : Geçerli bir WAV'ın ilk yarısı. Başlık (header)
                               sağlamdır ama başlıkta yazan veri uzunluğu
                               dosyada yoktur (dosya yarıda kesilmiştir).
  (b) sessiz_5sn.wav         : 5 sn, 16 kHz mono, tamamen sıfır örnek.
      sessiz_5sn.rttm        : BOŞ referans RTTM (hiç konuşma yok).
  (c) kisa_1sn.wav           : Kaynak WAV'dan --kisa-bas noktasından alınmış
                               1 sn'lik parça.
      kisa_1sn.rttm          : Aynı 1 sn'ye göre kırpılmış, 0'a kaydırılmış RTTM.
  (d) desteklenmeyen_bicim.xyz: Ses olmayan, desteklenmeyen uzantılı küçük
                               metin dosyası.

Hiçbir var olan dosyanın üzerine yazılmaz; aynı ad varsa _v2, _v3 ... eklenir.

Kullanım örneği:
  python edge_uret.py --kaynak-wav <ikgcq_028.0-118.0.wav>
      --kaynak-rttm <ikgcq_028.0-118.0.rttm> --kisa-bas 10
      --wav-cikti <edge_ses_klasoru> --rttm-cikti <edge_rttm_klasoru>
"""

import argparse
import os
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


def rttm_kirp(rttm_yolu, bas_ms, bit_ms, yeni_ad):
    """RTTM'i [bas_ms, bit_ms) aralığına kırpar, sınırdaki segmentleri keser,
    zamanları 0'a kaydırır ve 2. alanı yeni adla değiştirir."""
    satirlar = []
    with open(rttm_yolu, encoding="utf-8") as f:
        for satir in f:
            al = satir.split()
            if not al or al[0] != "SPEAKER":
                continue
            s = int(round(float(al[3]) * 1000))
            e = s + int(round(float(al[4]) * 1000))
            s2, e2 = max(s, bas_ms), min(e, bit_ms)
            if e2 <= s2:
                continue
            al[1] = yeni_ad
            al[3] = f"{(s2 - bas_ms) / 1000:.3f}"
            al[4] = f"{(e2 - s2) / 1000:.3f}"
            satirlar.append((s2, " ".join(al)))
    return [s for _, s in sorted(satirlar)]


def main():
    ap = argparse.ArgumentParser(description="Kenar durum test dosyalarını üretir.")
    ap.add_argument("--kaynak-wav", required=True, help="Kaynak 16-bit mono WAV")
    ap.add_argument("--kaynak-rttm", required=True, help="Kaynak WAV'ın RTTM'i")
    ap.add_argument("--kisa-bas", type=float, default=10.0,
                    help="1 sn'lik parçanın kaynak içindeki başlangıcı (sn)")
    ap.add_argument("--wav-cikti", required=True, help="Ses dosyalarının klasörü")
    ap.add_argument("--rttm-cikti", required=True, help="RTTM dosyalarının klasörü")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.wav_cikti, exist_ok=True)
    os.makedirs(args.rttm_cikti, exist_ok=True)

    with wave.open(args.kaynak_wav, "rb") as w:
        params = w.getparams()
        hiz = w.getframerate()
        tum = w.readframes(w.getnframes())
        w.setpos(int(args.kisa_bas * hiz))
        kisa = w.readframes(hiz)  # 1 sn

    # (a) Yarıda kesilmiş (bozuk) WAV: kaynak dosyanın ilk yarısı.
    with open(args.kaynak_wav, "rb") as f:
        ham = f.read()
    yol = bos_ad(args.wav_cikti, "bozuk_yarim", ".wav")
    with open(yol, "wb") as f:
        f.write(ham[: len(ham) // 2])
    print(f"(a) {yol} ({len(ham) // 2} bayt; başlıkta {len(tum)} bayt veri yazıyor)")

    # (b) 5 sn sessiz WAV ve boş RTTM.
    yol = bos_ad(args.wav_cikti, "sessiz_5sn", ".wav")
    with wave.open(yol, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 5)
    ad = os.path.splitext(os.path.basename(yol))[0]
    rttm = bos_ad(args.rttm_cikti, ad, ".rttm")
    open(rttm, "w", encoding="utf-8").close()
    print(f"(b) {yol} ve boş RTTM {rttm}")

    # (c) 1 sn'lik kısa WAV ve kırpılmış RTTM.
    yol = bos_ad(args.wav_cikti, "kisa_1sn", ".wav")
    with wave.open(yol, "wb") as w:
        w.setparams(params)
        w.writeframes(kisa)
    ad = os.path.splitext(os.path.basename(yol))[0]
    bas_ms = int(round(args.kisa_bas * 1000))
    satirlar = rttm_kirp(args.kaynak_rttm, bas_ms, bas_ms + 1000, ad)
    rttm = bos_ad(args.rttm_cikti, ad, ".rttm")
    with open(rttm, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(s + "\n" for s in satirlar))
    print(f"(c) {yol} ve {rttm} ({len(satirlar)} segment)")

    # (d) Desteklenmeyen biçim: ses olmayan küçük metin dosyası.
    yol = bos_ad(args.wav_cikti, "desteklenmeyen_bicim", ".xyz")
    with open(yol, "w", encoding="utf-8") as f:
        f.write("Bu dosya ses dosyasi degildir; desteklenmeyen bicim testi icindir.\n")
    print(f"(d) {yol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
