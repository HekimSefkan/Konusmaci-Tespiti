"""VoxConverse dev kayıtlarını eğitim / değerlendirme yarılarına böler. Sözleşme: docs/arayuz.md

Konuşmacı sayısına göre tabakalı (1, 2, 3-4, 5+), tekrarlanabilir (seed=68).
Yalnızca standart kütüphane kullanır.

Komut satırı:
    python -m src.bolme --rttm-dizin <dev_rttm_klasoru> --cikti-dizin <klasor> [--seed 68]
"""
import argparse
import os
import random
import sys

from . import cikti

GRUPLAR = ("1", "2", "3-4", "5+")
VARSAYILAN_SEED = 68


def grup(sayi: int) -> str:
    """Konuşmacı sayısının tabakası: "1", "2", "3-4" veya "5+" (0 konuşmacı "1" sayılır)."""
    if sayi <= 1:
        return "1"
    if sayi == 2:
        return "2"
    if sayi <= 4:
        return "3-4"
    return "5+"


def konusmaci_sayilari(rttm_dizin: str) -> dict[str, int]:
    """Klasördeki her <kayit>.rttm için farklı konuşmacı sayısı. Kayıt kimliği dosya adıdır."""
    if not os.path.isdir(rttm_dizin):
        raise FileNotFoundError(f"RTTM klasörü yok: {rttm_dizin}")
    sayilar = {}
    for ad in sorted(os.listdir(rttm_dizin)):
        if ad.lower().endswith(".rttm"):
            segler = cikti.rttm_oku(os.path.join(rttm_dizin, ad))
            sayilar[os.path.splitext(ad)[0]] = len({e for _, _, e in segler})
    return sayilar


def bol(sayilar: dict[str, int], seed: int = VARSAYILAN_SEED) -> tuple[list[str], list[str]]:
    """Kayıtları tabakalı olarak ikiye böler; (egitim, degerlendirme), ikisi de sıralı.

    Her tabaka sıralanıp random.Random(seed) ile karıştırılır ve ikiye ayrılır.
    Tek sayılı tabakalarda artan kayıt sırayla bir eğitime, bir değerlendirmeye
    verilir; böylece toplamlar en fazla 1 farklı olur. Aynı girdi her zaman aynı
    bölmeyi verir.
    """
    rng = random.Random(seed)
    egitim, degerlendirme = [], []
    fazla_egitime = True
    for g in GRUPLAR:
        kayitlar = sorted(k for k, n in sayilar.items() if grup(n) == g)
        rng.shuffle(kayitlar)
        n_egitim = len(kayitlar) // 2
        if len(kayitlar) % 2:
            n_egitim += 1 if fazla_egitime else 0
            fazla_egitime = not fazla_egitime
        egitim += kayitlar[:n_egitim]
        degerlendirme += kayitlar[n_egitim:]
    return sorted(egitim), sorted(degerlendirme)


def liste_yaz(kayitlar: list[str], yol: str) -> None:
    """Kayıt kimliklerini satır satır yazar."""
    klasor = os.path.dirname(yol)
    if klasor:
        os.makedirs(klasor, exist_ok=True)
    with open(yol, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{k}\n" for k in kayitlar)


def liste_oku(yol: str) -> list[str]:
    """Liste dosyasını okur; boş satırlar ve '#' ile başlayanlar atlanır."""
    with open(yol, encoding="utf-8") as f:
        return [s.strip() for s in f if s.strip() and not s.lstrip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.bolme", description="Tabakalı eğitim/değerlendirme bölmesi")
    ap.add_argument("--rttm-dizin", required=True, help="VoxConverse dev RTTM klasörü")
    ap.add_argument("--cikti-dizin", required=True, help="egitim.txt ve degerlendirme.txt klasörü")
    ap.add_argument("--seed", type=int, default=VARSAYILAN_SEED)
    a = ap.parse_args(argv)
    try:
        sayilar = konusmaci_sayilari(a.rttm_dizin)
    except (FileNotFoundError, ValueError) as e:
        print(f"Girdi hatası: {e}", file=sys.stderr)
        return 1
    egitim, degerlendirme = bol(sayilar, a.seed)
    liste_yaz(egitim, os.path.join(a.cikti_dizin, "egitim.txt"))
    liste_yaz(degerlendirme, os.path.join(a.cikti_dizin, "degerlendirme.txt"))
    print(f"Kayıt: {len(sayilar)}  eğitim: {len(egitim)}  değerlendirme: {len(degerlendirme)}  (seed={a.seed})")
    print(f"{'grup':<6}{'eğitim':>8}{'değ.':>8}")
    e_set = set(egitim)
    for g in GRUPLAR:
        uyeler = [k for k, n in sayilar.items() if grup(n) == g]
        n_e = sum(k in e_set for k in uyeler)
        print(f"{g:<6}{n_e:>8}{len(uyeler) - n_e:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
