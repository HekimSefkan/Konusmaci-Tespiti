"""src/uyumluluk.py testleri: sahte torchaudio modülüyle (yerelde torch gerekmez)."""
import sys
import types

from src import uyumluluk


def test_eksik_fonksiyonlar_eklenir(monkeypatch):
    sahte = types.ModuleType("torchaudio")  # torchaudio >= 2.9 gibi: iki fonksiyon da yok
    monkeypatch.setitem(sys.modules, "torchaudio", sahte)
    assert uyumluluk.speechbrain_hazirla() == ["list_audio_backends", "set_audio_backend"]
    assert sahte.list_audio_backends() == ["soundfile"]
    assert sahte.set_audio_backend("soundfile") is None
    assert uyumluluk.speechbrain_hazirla() == []  # tekrar çağrılabilir


def test_var_olan_fonksiyonlara_dokunulmaz(monkeypatch):
    sahte = types.ModuleType("torchaudio")  # eski torchaudio gibi: fonksiyonlar mevcut
    asil_liste = lambda: ["ffmpeg", "sox"]  # noqa: E731
    asil_ayar = lambda backend: "ayarlandi"  # noqa: E731
    sahte.list_audio_backends, sahte.set_audio_backend = asil_liste, asil_ayar
    monkeypatch.setitem(sys.modules, "torchaudio", sahte)
    assert uyumluluk.speechbrain_hazirla() == []
    assert sahte.list_audio_backends is asil_liste and sahte.set_audio_backend is asil_ayar
