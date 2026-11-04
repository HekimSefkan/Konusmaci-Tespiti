"""RTTM karşılaştırması ve deney grafikleri. Kullanım: python -m src.hafta5_gorsel --help."""
import argparse
import csv
from pathlib import Path

from .cikti import rttm_oku


def zaman_cizelgesi(dosyalar, cikti):
    """dosyalar: {başlık: RTTM yolu}. Renkler her satırda yerel etiketleri gösterir."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(dosyalar), 1, sharex=True,
                             figsize=(14, 2 * len(dosyalar)), squeeze=False)
    for ax, (ad, yol) in zip(axes[:, 0], dosyalar.items()):
        segler = rttm_oku(str(yol))
        etiketler = sorted({e for _, _, e in segler})
        for a, b, e in segler:
            i = etiketler.index(e)
            ax.broken_barh([(a, b-a)], (i-.35, .7), color=plt.cm.tab20(i % 20))
        ax.set_yticks(range(len(etiketler)), etiketler)
        ax.set_title(ad)
        ax.grid(axis='x', alpha=.2)
    axes[-1, 0].set_xlabel('Saniye (renkler satırlar arasında kişi eşleşmesi göstermez)')
    Path(cikti).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cikti, dpi=150)
    plt.close(fig)


def deney_grafigi(csv_yol, cikti):
    import matplotlib.pyplot as plt
    with open(csv_yol, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (key, label) in enumerate([('der', 'DER'), ('jer_ort', 'JER ort.'),
                                     ('mikro_f1', 'Değişim F1 mikro')]):
        ax.bar([i+(j-1)*.25 for i in range(len(rows))],
               [float(r[key]) for r in rows], width=.25, label=label)
    ax.set_xticks(range(len(rows)), [r['deney'] for r in rows])
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    Path(cikti).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cikti, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rttm', action='append', help='Başlık=yol (tekrarlanabilir)')
    ap.add_argument('--csv', help='ozet.csv')
    ap.add_argument('-o', '--cikti', required=True)
    a = ap.parse_args(argv)
    if bool(a.rttm) == bool(a.csv):
        ap.error('--rttm veya --csv seçeneklerinden birini verin')
    if a.csv:
        deney_grafigi(a.csv, a.cikti)
    else:
        zaman_cizelgesi(dict(s.split('=', 1) for s in a.rttm), a.cikti)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
