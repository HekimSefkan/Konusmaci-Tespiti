# -*- coding: utf-8 -*-
"""
ornek_kirp.py - VoxConverse zip arşivinden tek bir kaydı çıkarır, belirtilen
zaman aralığını 16 kHz mono PCM WAV olarak kırpar ve eşleşen RTTM'i aynı
aralığa göre kırpar.

Grup_68 - "Ses Kayıtlarında Konuşmacı Değişimlerinin Tespiti" projesi.
Gereksinimler: Python 3.8+ (yalnızca standart kütüphane) ve PATH'te ffmpeg.

Yapılanlar:
  1) Zip'in TAMAMI açılmaz; yalnızca istenen kayıt (audio/<kayit>.wav)
     --gecici klasörüne çıkarılır. zipfile, okuma sonunda CRC-32 kontrolü
     yapar; bozuk giriş varsa hata verir ve betik durur.
  2) ffmpeg ile [--bas, --bas + --sure) aralığı 16 kHz, mono, 16-bit PCM
     olarak kırpılır. --sure verilmezse kaydın sonuna kadar alınır
     (kırpma yok, yalnızca biçim dönüşümü).
  3) RTTM kırpılır:
       - Aralıkla kesişmeyen segmentler atılır.
       - Aralık sınırında bölünen segment sınırda kesilir (yalnızca aralık
         içindeki kısmı kalır).
       - Tüm zamanlar aralık başına göre 0'a kaydırılır.
       - 2. alan (dosya adı) yeni çıktı adıyla değiştirilir.
     Sınırda kesilen segmentler ekrana yazdırılır.
  4) Hiçbir var olan dosyanın üzerine yazılmaz; aynı ad varsa _v2, _v3 ...
     eklenir.

Kullanım örneği:
  python ornek_kirp.py --zip <voxconverse_dev_wav.zip> --rttm-klasor <voxconverse/dev>
      --kayit ikgcq --bas 28 --sure 90 --wav-cikti <ses_klasoru>
      --rttm-cikti <data/sample> --gecici <gecici_klasor>
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile


def bos_ad(klasor, ad, uzanti):
    """Var olan dosyanın üzerine yazmamak için boş bir dosya adı bulur."""
    yol = os.path.join(klasor, ad + uzanti)
    n = 2
    while os.path.exists(yol):
        yol = os.path.join(klasor, f"{ad}_v{n}{uzanti}")
        n += 1
    return yol


def zipten_cikar(zip_yolu, kayit, gecici):
    """audio/<kayit>.wav girişini gecici klasöre çıkarır (CRC kontrollü)."""
    os.makedirs(gecici, exist_ok=True)
    hedef = os.path.join(gecici, kayit + ".wav")
    if os.path.exists(hedef):
        return hedef
    with zipfile.ZipFile(zip_yolu) as z:
        adlar = [a for a in z.namelist() if a.rsplit("/", 1)[-1] == kayit + ".wav"]
        if len(adlar) != 1:
            raise SystemExit(f"Zip içinde {kayit}.wav bulunamadı ya da birden fazla.")
        gecici_hedef = hedef + ".part"
        with z.open(adlar[0]) as kaynak, open(gecici_hedef, "wb") as cikti:
            shutil.copyfileobj(kaynak, cikti, 1024 * 1024)  # CRC sonda kontrol edilir
        os.replace(gecici_hedef, hedef)
    return hedef


def ffmpeg_kirp(girdi, cikti, bas, sure):
    """ffmpeg ile 16 kHz mono 16-bit PCM WAV üretir (-n: üzerine yazma yok)."""
    komut = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", girdi]
    if bas:
        komut += ["-ss", f"{bas:.3f}"]
    if sure:
        komut += ["-t", f"{sure:.3f}"]
    komut += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", cikti]
    subprocess.run(komut, check=True)


def rttm_kirp(rttm_yolu, kayit, bas, bit, yeni_ad):
    """RTTM'i [bas, bit) aralığına göre kırpar. bit None ise sona kadar.
    (satirlar, kesilen_segmentler) döndürür. Zamanlar milisaniye ile işlenir."""
    a = int(round(bas * 1000))
    b = None if bit is None else int(round(bit * 1000))
    satirlar, kesilenler = [], []
    with open(rttm_yolu, encoding="utf-8") as f:
        for satir in f:
            al = satir.split()
            if not al or al[0] != "SPEAKER" or al[1] != kayit:
                continue
            s = int(round(float(al[3]) * 1000))
            e = s + int(round(float(al[4]) * 1000))
            s2 = max(s, a)
            e2 = e if b is None else min(e, b)
            if e2 <= s2:
                continue  # aralık dışında
            if s2 != s or e2 != e:
                kesilenler.append((al[7], s / 1000, e / 1000, s2 / 1000, e2 / 1000))
            al[1] = yeni_ad
            al[3] = f"{(s2 - a) / 1000:.3f}"
            al[4] = f"{(e2 - s2) / 1000:.3f}"
            satirlar.append((s2, " ".join(al)))
    satirlar.sort()
    return [s for _, s in satirlar], kesilenler


def main():
    ap = argparse.ArgumentParser(description="Zip'ten tek kayıt çıkar, WAV ve RTTM'i kırp.")
    ap.add_argument("--zip", required=True, help="voxconverse_dev_wav.zip yolu")
    ap.add_argument("--rttm-klasor", required=True, help="VoxConverse dev RTTM klasörü")
    ap.add_argument("--kayit", required=True, help="Kayıt adı, örn. ikgcq")
    ap.add_argument("--bas", type=float, default=0.0, help="Başlangıç (sn)")
    ap.add_argument("--sure", type=float, help="Süre (sn); verilmezse sona kadar")
    ap.add_argument("--wav-cikti", required=True, help="Kırpılmış WAV klasörü")
    ap.add_argument("--rttm-cikti", required=True, help="Kırpılmış RTTM klasörü")
    ap.add_argument("--gecici", required=True, help="Zip'ten çıkarılan WAV için geçici klasör")
    ap.add_argument("--ad", help="Çıktı adı (varsayılan: <kayit>_<bas>-<bit> veya <kayit>_tam)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg bulunamadı (PATH'e ekleyin).")

    bit = None if args.sure is None else args.bas + args.sure
    ad = args.ad or (f"{args.kayit}_tam" if bit is None
                     else f"{args.kayit}_{args.bas:05.1f}-{bit:05.1f}")
    os.makedirs(args.wav_cikti, exist_ok=True)
    os.makedirs(args.rttm_cikti, exist_ok=True)
    wav_yolu = bos_ad(args.wav_cikti, ad, ".wav")
    rttm_yolu = bos_ad(args.rttm_cikti, ad, ".rttm")
    ad = os.path.splitext(os.path.basename(wav_yolu))[0]

    kaynak = zipten_cikar(args.zip, args.kayit, args.gecici)
    print(f"Çıkarıldı (CRC tamam): {os.path.basename(kaynak)}")
    ffmpeg_kirp(kaynak, wav_yolu, args.bas, args.sure)

    satirlar, kesilenler = rttm_kirp(
        os.path.join(args.rttm_klasor, args.kayit + ".rttm"), args.kayit, args.bas, bit, ad)
    with open(rttm_yolu, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(s + "\n" for s in satirlar))

    print(f"WAV : {wav_yolu}")
    print(f"RTTM: {rttm_yolu} ({len(satirlar)} segment)")
    for k, s, e, s2, e2 in kesilenler:
        print(f"Sınırda kesilen segment: {k} [{s:.3f}-{e:.3f}] -> [{s2:.3f}-{e2:.3f}] (orijinal zaman)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
