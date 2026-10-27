"""Kütüphane sürüm uyumluluğu yamaları. Sözleşme: docs/arayuz.md (3.11)

speechbrain 1.0.3, içe aktarılırken (speechbrain/dataio/dataio.py modül düzeyinde
check_torchaudio_backend() -> speechbrain/utils/torch_audio_backend.py)
torchaudio.list_audio_backends() çağırır. Bu fonksiyon torchaudio 2.9'da
kaldırıldı; Colab'daki torch/torchaudio 2.11 ile `import speechbrain...`
AttributeError verir. Aynı dosya torchaudio < 2.1 için set_audio_backend() de
çağırır. Eksik olanları zararsız yer tutucularla tamamlıyoruz.

Hattımız torchaudio ile dosya okumaz (ses soundfile/librosa ile okunur,
ECAPA'ya tensör verilir), bu yüzden yer tutucuların davranışı sonucu etkilemez.
Bu modül yalnızca standart kütüphaneyi import eder; torchaudio fonksiyon içinde alınır.
"""


def speechbrain_hazirla() -> list[str]:
    """speechbrain içe aktarılmadan ÖNCE çağrılır; eksik torchaudio fonksiyonlarını tamamlar.

    - torchaudio.list_audio_backends yoksa ["soundfile"] döndüren bir fonksiyon atanır
      (boş olmayan liste: speechbrain "backend bulunamadı" uyarısı vermesin).
    - torchaudio.set_audio_backend yoksa hiçbir şey yapmayan bir fonksiyon atanır.
    Var olan fonksiyonlara dokunulmaz; tekrar çağrılabilir.
    Dönüş: bu çağrıda eklenen fonksiyon adları.
    Hata: ImportError (torchaudio kurulu değil).
    """
    import torchaudio

    eklenen = []
    if not hasattr(torchaudio, "list_audio_backends"):
        def list_audio_backends():
            return ["soundfile"]

        torchaudio.list_audio_backends = list_audio_backends
        eklenen.append("list_audio_backends")
    if not hasattr(torchaudio, "set_audio_backend"):
        def set_audio_backend(backend=None):
            return None

        torchaudio.set_audio_backend = set_audio_backend
        eklenen.append("set_audio_backend")
    return eklenen
