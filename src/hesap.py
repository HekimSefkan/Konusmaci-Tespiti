"""Uygulama içi hesaplar: kayıt, giriş, oturum. Sözleşme: docs/arayuz.md (3.18)

Kullanıcılar KT_KULLANICI_DB (JSON) dosyasında durur. Şifreler hashlib.scrypt ve kullanıcıya özel
rastgele tuzla saklanır; dosyaya yazma süreç içi kilit + dosya kilidi (<db>.lock, O_EXCL) altında ve
atomiktir (geçici dosya + os.replace). Aynı e-posta ikinci kez kaydolamaz. Hatalı girişte genel
mesaj verilir; aynı e-postaya MAKS_DENEME hatalı denemeden sonra KILIT_SN boyunca giriş kapanır
(e-posta kayıtlı olmasa da). Oturumlar yalnızca bellekte, süre sınırlı rastgele anahtarlardır;
sunucu yeniden başlarsa yeniden giriş gerekir. Yalnızca standart kütüphane.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

DB_ORTAM = "KT_KULLANICI_DB"
KAYIT_ORTAM = "KT_KAYIT_ACIK"
MIN_SIFRE = 8
MAKS_DENEME = 5
KILIT_SN = 300
OTURUM_SN = 12 * 3600
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}
GENEL_HATA = "E-posta veya şifre hatalı."
KILIT_HATASI = "Çok fazla hatalı deneme. Lütfen 5 dakika sonra tekrar deneyin."
_EPOSTA = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class HesapHatasi(ValueError):
    """Kullanıcıya olduğu gibi gösterilebilecek hesap hatası."""


def kayit_acik() -> bool:
    """KT_KAYIT_ACIK "0" (veya false/hayır/no) değilse kayıt açıktır."""
    return os.environ.get(KAYIT_ORTAM, "1").strip().lower() not in ("0", "false", "hayir", "hayır", "no")


def db_yolu() -> str | None:
    deger = os.environ.get(DB_ORTAM, "").strip()
    return os.path.expanduser(deger) if deger else None


def _ozet(sifre: str, tuz: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(sifre.encode("utf-8"), salt=tuz, n=n, r=r, p=p, dklen=dklen, maxmem=64 * 1024 * 1024)


def _eposta(e: str) -> str:
    return (e or "").strip().lower()


class HesapDeposu:
    """Kullanıcı dosyası, hatalı deneme sayaçları ve bellekteki oturumlar."""

    def __init__(self, yol: str, oturum_sn: int = OTURUM_SN, saat=time.time):
        self.yol = yol
        self.oturum_sn = oturum_sn
        self._saat = saat
        self._kilit = threading.RLock()
        self._oturumlar: dict[str, dict] = {}
        self._denemeler: dict[str, dict] = {}
        self._sahte_tuz = os.urandom(16)  # kayıtlı olmayan e-postada da aynı süre harcansın

    # --- dosya --------------------------------------------------------------------------
    @contextmanager
    def _dosya_kilidi(self, bekleme_sn: float = 10.0, bayat_sn: float = 30.0):
        kilit = self.yol + ".lock"
        os.makedirs(os.path.dirname(os.path.abspath(self.yol)), exist_ok=True)
        bitis = time.time() + bekleme_sn
        while True:
            try:
                fd = os.open(kilit, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(kilit) > bayat_sn:  # çöken süreçten kalan kilit
                        os.remove(kilit)
                        continue
                except OSError:
                    pass
                if time.time() > bitis:
                    raise HesapHatasi("Kullanıcı veritabanı meşgul; lütfen tekrar deneyin.")
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(fd)
            try:
                os.remove(kilit)
            except OSError:
                pass

    def _oku(self) -> dict:
        if not os.path.isfile(self.yol):
            return {"surum": 1, "kullanicilar": {}}
        with open(self.yol, encoding="utf-8") as f:
            veri = json.load(f)
        veri.setdefault("kullanicilar", {})
        return veri

    def _yaz(self, veri: dict) -> None:
        klasor = os.path.dirname(os.path.abspath(self.yol))
        fd, gecici = tempfile.mkstemp(dir=klasor, prefix=".kullanicilar-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(veri, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(gecici, self.yol)
        except BaseException:
            if os.path.exists(gecici):
                os.remove(gecici)
            raise

    # --- kayıt ve giriş -------------------------------------------------------------------
    def kayit(self, ad: str, eposta: str, sifre: str, tekrar: str) -> dict:
        """Yeni kullanıcı; dönüş {"id", "ad", "eposta"}. Hata: HesapHatasi (geçersiz alan, e-posta kayıtlı)."""
        ad, e = (ad or "").strip(), _eposta(eposta)
        if not ad or len(ad) > 60:
            raise HesapHatasi("Ad 1–60 karakter olmalı.")
        if not _EPOSTA.match(e) or len(e) > 120:
            raise HesapHatasi("Geçerli bir e-posta adresi girin.")
        if len(sifre or "") < MIN_SIFRE:
            raise HesapHatasi(f"Şifre en az {MIN_SIFRE} karakter olmalı.")
        if sifre != tekrar:
            raise HesapHatasi("Şifreler aynı değil.")
        tuz = os.urandom(16)
        kayit = {"id": uuid.uuid4().hex, "ad": ad, "eposta": e, "tuz": tuz.hex(),
                 "ozet": _ozet(sifre, tuz, **SCRYPT).hex(), **{f"scrypt_{k}": v for k, v in SCRYPT.items()},
                 "olusturma": datetime.now().isoformat(timespec="seconds")}
        with self._kilit, self._dosya_kilidi():
            veri = self._oku()
            if e in veri["kullanicilar"]:
                raise HesapHatasi("Bu e-posta ile kayıtlı bir hesap var.")
            veri["kullanicilar"][e] = kayit
            self._yaz(veri)
        return {"id": kayit["id"], "ad": ad, "eposta": e}

    def _dogrula(self, kayit: dict | None, sifre: str) -> bool:
        if kayit is None:
            _ozet(sifre or "", self._sahte_tuz, **SCRYPT)
            return False
        p = {k: kayit[f"scrypt_{k}"] for k in SCRYPT}
        return hmac.compare_digest(_ozet(sifre or "", bytes.fromhex(kayit["tuz"]), **p), bytes.fromhex(kayit["ozet"]))

    def giris(self, eposta: str, sifre: str) -> str:
        """Başarılıysa oturum anahtarı. Hata: HesapHatasi (GENEL_HATA veya KILIT_HATASI)."""
        e = _eposta(eposta)
        with self._kilit:
            d = self._denemeler.get(e)
            if d and d.get("kilit", 0) > self._saat():
                raise HesapHatasi(KILIT_HATASI)
        kayit = self._oku()["kullanicilar"].get(e) if e else None
        if not self._dogrula(kayit, sifre):
            with self._kilit:
                d = self._denemeler.setdefault(e, {"sayi": 0, "kilit": 0})
                d["sayi"] += 1
                if d["sayi"] >= MAKS_DENEME:
                    d.update(sayi=0, kilit=self._saat() + KILIT_SN)
                    raise HesapHatasi(KILIT_HATASI)
            raise HesapHatasi(GENEL_HATA)
        with self._kilit:
            self._denemeler.pop(e, None)
        return self.oturum_ac({"id": kayit["id"], "ad": kayit["ad"], "eposta": e})

    # --- oturum ---------------------------------------------------------------------------
    def oturum_ac(self, kullanici: dict) -> str:
        anahtar = secrets.token_urlsafe(32)
        with self._kilit:
            simdi = self._saat()
            for k in [k for k, o in self._oturumlar.items() if o["bitis"] < simdi]:
                del self._oturumlar[k]
            self._oturumlar[anahtar] = {**kullanici, "bitis": simdi + self.oturum_sn}
        return anahtar

    def kullanici(self, anahtar) -> dict | None:
        """Geçerli oturumun kullanıcısı {"id", "ad", "eposta"}; geçersiz veya süresi dolmuşsa None."""
        if not isinstance(anahtar, str) or not anahtar:
            return None
        with self._kilit:
            o = self._oturumlar.get(anahtar)
            if o is None:
                return None
            if o["bitis"] < self._saat():
                del self._oturumlar[anahtar]
                return None
            return {k: o[k] for k in ("id", "ad", "eposta")}

    def cikis(self, anahtar) -> None:
        with self._kilit:
            self._oturumlar.pop(anahtar, None)
