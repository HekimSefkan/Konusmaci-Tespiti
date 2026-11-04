"""Örnek ve gerçek gürültülü kesitlerde Silero/WebRTC ölçümü; rapor üretmez."""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
from . import cikti, ses_girdi, vad
from .hafta5 import yaz_csv


def webrtc_bolgeleri(ses, sr, mod):
    import webrtcvad
    model = webrtcvad.Vad(mod)
    n = sr * 30 // 1000
    pcm = (np.clip(ses, -1, 1)*32767).astype('<i2')
    return [(i/sr, (i+n)/sr) for i in range(0, len(ses)-n+1, n)
            if model.is_speech(pcm[i:i+n].tobytes(), sr)]


def hata(ref, hyp, sure, adim=.01):
    """10 ms merkez ızgarasında konuşma var/yok hataları; örtüşme birleşimdir."""
    t = np.arange(adim/2, sure, adim)
    def maske(segler):
        m = np.zeros(len(t), dtype=bool)
        for a,b,*_ in segler:
            m |= (t>=a)&(t<b)
        return m
    r,h = maske(ref),maske(hyp)
    return {'referans_sn': float(r.sum()*adim),
            'kacirilan_sn': float((r&~h).sum()*adim),
            'yanlis_alarm_sn': float((~r&h).sum()*adim)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kok', type=Path, required=True)
    ap.add_argument('--referans', type=Path, required=True)
    a = ap.parse_args(argv)
    kok = a.kok.resolve()
    out = kok/'Hafta5/vad'
    out.mkdir(parents=True, exist_ok=True)
    refs = Path(__file__).resolve().parents[1]/'data/sample'
    girdiler = [(p, cikti.rttm_oku(str(refs/f'{p.stem}.rttm')), 'ornek')
                for p in sorted((kok/'Hafta2/ornek_ses').glob('*.wav'))]
    with (kok/'Hafta4/sonuclar/dinleme_listesi.csv').open() as f:
        for r in csv.DictReader(f):
            bas,bit = float(r['bas_sn']),float(r['bit_sn'])
            ref = cikti.rttm_oku(str(a.referans/f"{r['kayit']}.rttm"))
            kesit = [(max(x,bas)-bas,min(y,bit)-bas,e) for x,y,e in ref if y>bas and x<bit]
            girdiler.append((kok/'Hafta4/dinleme'/r['dosya'], kesit, 'gercek_gurultulu'))
    rows = []
    for p,ref,tur in girdiler:
        x,sr = ses_girdi.ses_yukle(str(p))
        for ad,fn in [('silero',lambda:vad.konusma_bolgeleri(x,sr)),
                      ('webrtc2',lambda:webrtc_bolgeleri(x,sr,2)),
                      ('webrtc3',lambda:webrtc_bolgeleri(x,sr,3))]:
            t = time.perf_counter()
            hyp = fn()
            rows.append({'kayit':p.stem,'tur':tur,'vad':ad,'ses_sn':len(x)/sr,
                         'sure_sn':time.perf_counter()-t,**hata(ref,hyp,len(x)/sr)})
        cikti.rttm_yaz(ref,p.stem,str(out/f'{p.stem}_referans.rttm'))
    yaz_csv(out/'kayitlar.csv',rows)
    yaz_csv(out/'ozet.csv', [
        {'tur':tur,'vad':ad,'kayit_sayisi':len(rs),
         **{k:sum(r[k] for r in rs) for k in ['ses_sn','sure_sn','referans_sn','kacirilan_sn','yanlis_alarm_sn']}}
        for tur in ['ornek','gercek_gurultulu'] for ad in ['silero','webrtc2','webrtc3']
        if (rs := [r for r in rows if r['tur']==tur and r['vad']==ad])])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
