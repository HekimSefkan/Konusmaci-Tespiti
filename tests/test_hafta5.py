import numpy as np
import pytest
from src import hafta5, toplu


def test_onbellek_farkli_pencereyi_reddeder(tmp_path):
    for tur, anahtar in toplu.onbellek_anahtarlari().items():
        p = tmp_path/'Hafta4/onbellek'/tur/anahtar/'kayit.npz'
        p.parent.mkdir(parents=True)
        np.savez(p, pencereler=np.array([[0, 1 if tur == 'mfcc' else 2]]))
    with pytest.raises(ValueError, match='Pencere uyuşmazlığı'):
        hafta5.onbellek(tmp_path, 'kayit')


def test_egitim_degerlendirme_kesisimi_model_yuklemeden_reddedilir(tmp_path):
    p = tmp_path/'Hafta4/bolme'
    p.mkdir(parents=True)
    (p/'egitim.txt').write_text('ortak\n')
    (p/'degerlendirme.txt').write_text('ortak\n')
    with pytest.raises(ValueError, match='kesişiyor'):
        hafta5.deneyler(tmp_path, tmp_path/'referans', tmp_path/'cikti')


def test_bos_sonuc_basari_gibi_kaydedilmez(tmp_path):
    with pytest.raises(ValueError, match='Boş sonuç'):
        hafta5.yaz_csv(tmp_path/'sonuc.csv', [])
    assert not (tmp_path/'sonuc.csv').exists()


@pytest.mark.parametrize('ad,kod,mevcut,seg,gecti', [
    ('sessiz_5sn.wav', 0, True, [], True),
    ('sessiz_5sn.wav', 2, False, [], False),
    ('kisa_1sn.wav', 0, True, [(0, 1, 'a')], True),
    ('kisa_1sn.wav', 0, True, [(0, .5, 'a'), (.5, 1, 'b')], False),
    ('kisa_1sn.wav', 0, False, [], False),
    ('bozuk_yarim.wav', 1, False, [], True),
    ('desteklenmeyen_bicim.xyz', 1, True, [], False),
])
def test_kenar_durum_beklentisi(ad, kod, mevcut, seg, gecti):
    assert hafta5.kenar_beklentisi(ad, kod, mevcut, seg) is gecti


def test_girdi_hashi_model_degisince_degisir(tmp_path):
    model = tmp_path/'Hafta4/modeller/model.joblib'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'ilk')
    once = hafta5.girdi_ozeti(tmp_path)[str(model)]
    model.write_bytes(b'ikinci')
    assert hafta5.girdi_ozeti(tmp_path)[str(model)] != once
