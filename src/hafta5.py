"""Hafta5 teknik ölçümleri. Rapor üretmez. python -m src.hafta5 --help."""
import argparse
import csv
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
from . import cikti, degerlendirme, degisim_modeli, hafta5_gorsel, main as ana, toplu


def girdi_ozeti(kok, referans=None):
    """Kod, model, liste, öznitelik ve kullanılan referansları içerik hashleriyle tanımlar."""
    paths = list(Path(__file__).resolve().parent.glob('*.py'))
    for dizin in ('Hafta4/modeller', 'Hafta4/bolme', 'Hafta4/onbellek',
                  'Hafta2/ornek_ses', 'Hafta4/dinleme'):
        paths.extend(p for p in (kok/dizin).rglob('*') if p.is_file())
    if referans is not None:
        paths.extend(referans.glob('*.rttm'))
    ozet = {}
    for p in sorted(set(paths)):
        sha = hashlib.sha256()
        with p.open('rb') as f:
            for parca in iter(lambda: f.read(1024 * 1024), b''):
                sha.update(parca)
        ozet[str(p.resolve())] = sha.hexdigest()
    return ozet


def kenar_beklentisi(ad, kod, mevcut, segmentler):
    """Bilinen kenar dosyaları için çıkış kodu ve çıktı içeriğini birlikte doğrular."""
    if ad in ('bozuk_yarim.wav', 'desteklenmeyen_bicim.xyz'):
        return kod == 1 and not mevcut
    if ad == 'sessiz_5sn.wav':
        return kod == 0 and mevcut and not segmentler
    if ad == 'kisa_1sn.wav':
        return kod == 0 and mevcut and len({s[2] for s in segmentler}) <= 1
    raise ValueError(f'Tanımsız kenar durumu: {ad}')


def yaz_csv(yol, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'Boş sonuç: {yol}')
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    with yol.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def json_yaz(yol, value):
    Path(yol).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def onbellek(kok, kayit):
    veri = {}
    for tur, anahtar in toplu.onbellek_anahtarlari().items():
        p = kok / 'Hafta4/onbellek' / tur / anahtar / f'{kayit}.npz'
        with np.load(p) as d:
            veri[tur] = {k: d[k] for k in d.files}
    if not np.array_equal(veri['ecapa']['pencereler'], veri['mfcc']['pencereler']):
        raise ValueError(f'Pencere uyuşmazlığı: {kayit}')
    return veri


def e5(paket, d, esik):
    return degisim_modeli.e5_segmentleri(paket, d['vad']['bolgeler'].tolist(),
        d['ecapa']['pencereler'].tolist(), d['ecapa']['X'], d['mfcc']['X'], esik)


def deneyler(kok, ref, out):
    """Yalnız eğitim kayıtlarında aday seçer, seçileni değerlendirmede bir kez ölçer."""
    egitim = (kok/'Hafta4/bolme/egitim.txt').read_text().split()
    test = (kok/'Hafta4/bolme/degerlendirme.txt').read_text().split()
    if set(egitim) & set(test):
        raise ValueError('Eğitim ve değerlendirme kesişiyor')
    paket = degisim_modeli.yukle(str(kok/'Hafta4/modeller/degisim_modeli.joblib'))
    # Tek değişken: E5 için AHC eşiği. Diğer model ayarları korunur.
    esikler = sorted(set([.55, .65, .70, .75, .79346, .85, .90, .95]))
    adaylar = []
    for esik in esikler:
        rows = []
        for k in egitim:
            bas = time.perf_counter()
            d = onbellek(kok, k)
            s = degerlendirme._segmentleri_degerlendir(cikti.rttm_oku(str(ref/f'{k}.rttm')),
                                                     e5(paket, d, esik))
            rows.append({'kayit': k, 'ses_sn': float(d['vad']['ses_sn']),
                         'sure_sn': time.perf_counter()-bas, **s})
        yaz_csv(out/f'egitim/esik_{esik}.csv', rows)
        adaylar.append({'esik': esik, **toplu.ozet_hesapla(rows)})
        print('E5 eğitim', esik, adaylar[-1]['der'], flush=True)
    yaz_csv(out/'egitim/esik_tarama.csv', adaylar)
    secilen = min(adaylar, key=lambda r: (r['der'], r['esik']))
    json_yaz(out/'secilen_ayar.json', {'secim_kumesi': 'egitim', 'olcut': 'toplam DER',
                                     'esik': secilen['esik'], 'adaylar': esikler})
    rows = []
    for k in test:
        bas = time.perf_counter()
        d = onbellek(kok, k)
        seg = e5(paket, d, secilen['esik'])
        hyp = out/'E5_ayarli'/f'{k}.rttm'
        hyp.parent.mkdir(parents=True, exist_ok=True)
        cikti.rttm_yaz(seg, k, str(hyp))
        rows.append({'kayit': k, 'ses_sn': float(d['vad']['ses_sn']),
                     'sure_sn': time.perf_counter()-bas,
                     **degerlendirme.degerlendir(str(ref/f'{k}.rttm'), str(hyp))})
    yaz_csv(out/'E5_ayarli.csv', rows)
    json_yaz(out/'E5_ayarli_ozet.json', toplu.ozet_hesapla(rows))


def analiz(kok, ref, out):
    rows = []
    for p in sorted((kok/'Hafta4/sonuclar').glob('E[1-5].csv')):
        with p.open() as f:
            data = list(csv.DictReader(f))
        for r in data:
            for k in ['der', 'jer', 'f1', 'tp', 'fp', 'fn', 'referans_sn',
                      'yanlis_alarm_sn', 'kacirilan_sn', 'karisiklik_sn']:
                r[k] = float(r[k])
        degisen = [r for r in data if r['tp']+r['fn'] > 0]
        rows.append({'deney': p.stem, 'kayit_sayisi': len(data),
                     'degisimli_kayit': len(degisen),
                     'makro_f1_degisimli': statistics.mean(r['f1'] for r in degisen),
                     **toplu.ozet_hesapla(data)})
    yaz_csv(out/'metrik_dogrulama.csv', rows)
    zor = ['tucrg', 'ngyrk', 'mpvoh', 'sqkup', 'wjhgf', 'jsmbi', 'nxgad', 'praxo']
    hata = []
    for k in zor:
        dosyalar = {'Referans': ref/f'{k}.rttm'}
        for e in ['E4', 'E5']:
            p = kok/'Hafta4/sonuclar'/e/f'{k}.rttm'
            dosyalar[e] = p
            seg = cikti.rttm_oku(str(p))
            s = degerlendirme.degerlendir(str(ref/f'{k}.rttm'), str(p))
            hata.append({'kayit': k, 'deney': e, 'konusmaci_sayisi': len({x[2] for x in seg}),
                         'segment_sayisi': len(seg),
                         'kisa_segment_sayisi': sum(b-a < 1.5 for a,b,_ in seg), **s})
        ayarli = kok/'Hafta5/deney/E5_ayarli'/f'{k}.rttm'
        if ayarli.exists():
            dosyalar['E5 eğitimde ayarlanmış eşik'] = ayarli
        hafta5_gorsel.zaman_cizelgesi(dosyalar, out/'gorseller'/f'{k}.png')
    yaz_csv(out/'hata_analizi.csv', hata)
    hafta5_gorsel.deney_grafigi(kok/'Hafta4/sonuclar/ozet.csv', out/'gorseller/deneyler.png')
    karsilastirma = []
    for k in (kok/'Hafta4/bolme/degerlendirme.txt').read_text().split():
        row = {'kayit': k, 'referans_konusmaci': len({s[2] for s in cikti.rttm_oku(str(ref/f'{k}.rttm'))})}
        for ad, klasor in [('E4', kok/'Hafta4/sonuclar/E4'),
                           ('E5', kok/'Hafta4/sonuclar/E5'),
                           ('E5_ayarli', kok/'Hafta5/deney/E5_ayarli')]:
            p = klasor/f'{k}.rttm'
            if not p.exists():
                continue
            row[ad+'_konusmaci'] = len({s[2] for s in cikti.rttm_oku(str(p))})
            row[ad+'_der'] = degerlendirme.degerlendir(str(ref/f'{k}.rttm'), str(p))['der']
        karsilastirma.append(row)
    yaz_csv(out/'kayit_karsilastirma.csv', karsilastirma)


def kabul(kok, out, benchmark=False):
    sesler = sorted((kok/'Hafta2/ornek_ses').glob('*.wav'))
    if len(sesler) != 5:
        raise ValueError(f'Kabul ölçümü beş örnek WAV gerektiriyor; bulunan {len(sesler)}')
    refs = Path(__file__).resolve().parents[1]/'data/sample'
    rows = []
    for ses in sesler:
        for yontem in ['mfcc', 'ecapa']:
            hyp = out/'kabul_rttm'/f'{ses.stem}_{yontem}.rttm'
            hyp.parent.mkdir(parents=True, exist_ok=True)
            t = time.perf_counter()
            kod = ana.main([str(ses), '-o', str(hyp), '--yontem', yontem])
            if kod:
                raise RuntimeError(f'{ses}: çıkış kodu {kod}')
            ref = refs/f'{ses.stem}.rttm'
            s = degerlendirme.degerlendir(str(ref), str(hyp))
            rows.append({'kayit': ses.stem, 'yontem': yontem, 'kod': kod,
                         'sure_sn': time.perf_counter()-t,
                         'konusmaci_sayisi': len({s[2] for s in cikti.rttm_oku(str(hyp))}), **s})
    yaz_csv(out/'kabul_sonuclari.csv', rows)
    kabul_ozeti = []
    for yontem in ['mfcc', 'ecapa']:
        rs = [r for r in rows if r['yontem']==yontem]
        degisen = [r for r in rs if r['tp']+r['fn']>0]
        der = statistics.mean(r['der'] for r in rs)
        f1 = statistics.mean(r['f1'] for r in degisen)
        kabul_ozeti.append({'yontem': yontem, 'der_ortalama': der,
            'tg2_gecti': der <= .35, 'degisimli_makro_f1': f1, 'tg5_gecti': f1 >= .4,
            'collar_tam_sn': .5, 'degisim_tolerans_sn': .5})
    yaz_csv(out/'kabul_ozeti.csv', kabul_ozeti)
    edge = []
    for ses in sorted((kok/'Hafta2/ornek_ses/edge').glob('*')):
        if 'snr' in ses.name:
            continue
        hyp = out/'kabul_rttm'/f'{ses.stem}.rttm'
        if hyp.exists():
            hyp.unlink()  # önceki başarılı çalışmanın çıktısı hatayı gizlemesin
        kod = ana.main([str(ses), '-o', str(hyp)])
        seg = cikti.rttm_oku(str(hyp)) if hyp.exists() else []
        edge.append({'kayit': ses.name, 'kod': kod, 'segment_sayisi': len(seg),
                     'konusmaci_sayisi': len({s[2] for s in seg}),
                     'gecti': kenar_beklentisi(ses.name, kod, hyp.exists(), seg)})
    yaz_csv(out/'kenar_durumlar.csv', edge)
    if not all(r['gecti'] for r in edge):
        raise RuntimeError('Kenar durum testi başarısız; kenar_durumlar.csv dosyasını inceleyin')
    if benchmark:
        import torch
        torch.set_num_threads(1)
        ses = next(s for s in sesler if s.name.startswith('ikgcq'))
        from .ses_girdi import wav_denetle
        ses_sn = wav_denetle(str(ses))
        if abs(ses_sn - 90) > .01:
            raise ValueError(f'CPU kabul ölçümü 90 sn örnek gerektiriyor; gerçek süre {ses_sn}')
        zamanlar = []
        for i in range(5):
            t = time.perf_counter()
            kod = ana.main([str(ses), '-o', str(out/'benchmark.rttm')])
            if kod:
                raise RuntimeError('CPU ölçümü başarısız')
            zamanlar.append(time.perf_counter()-t)
        json_yaz(out/'cpu_benchmark.json', {'cihaz': platform.processor(), 'platform': platform.platform(),
            'torch_thread': 1, 'ses_sn': ses_sn, 'tekrar_sn': zamanlar,
            'medyan_sn': statistics.median(zamanlar), 'rtf': statistics.median(zamanlar)/ses_sn,
            'gecti': statistics.median(zamanlar)<=45,
            'model_yukleme': 'önceden yüklenmiş; ses okuma ve öznitelik çıkarımı her tekrarda yeniden'})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kok', type=Path, required=True, help='Hafta2/Hafta4 klasörlerinin kökü')
    ap.add_argument('--referans', type=Path, help='VoxConverse dev RTTM klasörü')
    ap.add_argument('--is', dest='islem', choices=['analiz', 'deney', 'kabul'], required=True)
    ap.add_argument('--benchmark', action='store_true')
    a = ap.parse_args(argv)
    kok = a.kok.resolve()
    out = kok/'Hafta5'/a.islem
    out.mkdir(parents=True, exist_ok=True)
    os.environ['KT_MODEL_ONBELLEK'] = str(kok/'Hafta4/modeller/ecapa')
    import torch
    if torch.cuda.is_available() and a.benchmark:
        raise RuntimeError('CPU ölçümü için CUDA_VISIBLE_DEVICES="" ile çalıştırın')
    env = {'python': platform.python_version(), 'platform': platform.platform(),
           'paketler': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
           'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'git_durumu': subprocess.check_output(['git', 'status', '--short'], text=True),
           'is': a.islem, 'referans': str(a.referans),
           'cuda_kullanilabilir': torch.cuda.is_available()}
    json_yaz(out/'ortam.json', env)
    json_yaz(out/'girdi_sha256.json', girdi_ozeti(kok, a.referans))
    if a.islem == 'kabul':
        kabul(kok, out, a.benchmark)
    elif a.referans is None:
        ap.error('--referans gerekli')
    elif a.islem == 'analiz':
        analiz(kok, a.referans, out)
    else:
        deneyler(kok, a.referans, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
