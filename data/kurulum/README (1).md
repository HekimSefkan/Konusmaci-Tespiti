# Konuşmacı Tanıma – Ön İşleme ve Öznitelik Çıkarımı

Ses kaydından konuşma bölümlerini bulur (VAD), MFCC ve SpeechBrain ECAPA-TDNN özniteliklerini çıkarır ve konuşmayı 1,5 sn'lik pencerelerle (0,75 sn kaydırma) vektörlere dönüştürür.

## Proje özeti

| Konu | Karar |
|---|---|
| Girdi formatları | wav, mp3, flac |
| İç örnekleme hızı | 16 kHz, mono (stereo → kanal ortalaması) |
| Maksimum kayıt süresi | 300 sn (5 dk) |
| Ön işleme | RMS ≈ −20 dBFS normalizasyon, isteğe bağlı noisereduce |
| VAD | Silero VAD (ana), WebRTC VAD (karşılaştırma); min. konuşma 250 ms, min. sessizlik 300 ms |
| MFCC | 13 katsayı, 25 ms pencere, 10 ms kaydırma |
| Gömme modeli | SpeechBrain ECAPA-TDNN (192 boyut) |
| Segment | 1,5 sn pencere, 0,75 sn kaydırma |
| Python | 3.10 |
| Donanım | Yalnızca CPU yeterli, GPU gerekmez |

## Kurulum

### Ön koşullar
- **Python 3.10** (3.11 ve üzeri için `requirements.txt` sürümleri test edilmemiştir)
- **Git**
- **ffmpeg** (mp3 dosyalarını okumak için)
- Yaklaşık 3 GB boş disk alanı (torch ve model dosyaları için)

### Windows

1. **Python 3.10'u kurun.** python.org'dan Python 3.10 yükleyicisini indirin. Kurulum ekranında **"Add Python to PATH"** kutusunu işaretleyin.
2. **Sürümü doğrulayın.** PowerShell açın:
   ```powershell
   py -3.10 --version
   ```
   Çıktı `Python 3.10.x` olmalıdır.
3. **ffmpeg'i kurun.**
   ```powershell
   winget install Gyan.FFmpeg
   ```
   Kurulumdan sonra PowerShell'i kapatıp yeniden açın ve `ffmpeg -version` ile doğrulayın.
4. **Depoyu indirin.**
   ```powershell
   git clone <REPO_URL>
   cd <REPO_ADI>
   ```
5. **Sanal ortam oluşturun.**
   ```powershell
   py -3.10 -m venv venv
   ```
6. **Sanal ortamı etkinleştirin.**
   ```powershell
   venv\Scripts\activate
   ```
   "Bu sistemde betik çalıştırma devre dışı" hatası alırsanız bir kez şunu çalıştırın ve tekrar deneyin:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
   Etkinleşince satır başında `(venv)` görünür.
7. **Bağımlılıkları kurun.**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Kurulum birkaç dakika sürebilir (torch büyük bir pakettir).
8. **Kurulumu doğrulayın.**
   ```powershell
   python -c "import numpy, librosa, torch, torchaudio, speechbrain, webrtcvad, noisereduce; print('OK')"
   ```
   Çıktı `OK` olmalıdır.

### Google Colab

1. [colab.research.google.com](https://colab.research.google.com) adresinde **Yeni not defteri** açın. GPU gerekmez; varsayılan CPU çalışma zamanı yeterlidir.
2. **Python sürümünü kontrol edin.**
   ```python
   !python --version
   ```
   Sürüm 3.10 değilse sürüm uyarıları çıkabilir; bu durumda aşağıdaki Notlar bölümüne bakın.
3. **Depoyu indirin.**
   ```python
   !git clone <REPO_URL>
   %cd <REPO_ADI>
   ```
4. **ffmpeg'i kurun.**
   ```python
   !apt-get install -y ffmpeg
   ```
5. **Bağımlılıkları kurun.**
   ```python
   !pip install -r requirements.txt
   ```
6. **Çalışma zamanını yeniden başlatın.** Kurulum bittikten sonra Colab önceden yüklü numpy gibi paketleri değiştirmiş olabilir: *Çalışma zamanı → Oturumu yeniden başlat*. Yeniden başlattıktan sonra 2–5. adımları tekrarlamanız gerekmez.
7. **Kurulumu doğrulayın.**
   ```python
   import numpy, librosa, torch, torchaudio, speechbrain, webrtcvad, noisereduce
   print("OK")
   ```

### Notlar ve sorun giderme

- **`webrtcvad` Windows'ta derleme hatası veriyor:** "Microsoft C++ Build Tools" kurun (Visual Studio Build Tools → "C++ ile masaüstü geliştirme") ve `pip install -r requirements.txt` komutunu tekrar çalıştırın.
- **mp3 açılmıyor:** `ffmpeg -version` komutunun çalıştığını ve ffmpeg'in PATH'te olduğunu doğrulayın.
- **Colab'da sürüm çakışması uyarısı:** Colab önceden yüklü paketlerle çakışma bildirebilir. Hata mesajındaki paketin sürümünü not edin, `requirements.txt` dosyasını buna göre güncelleyin ve değişikliği bu README'ye yansıtın.
- **`ModuleNotFoundError`:** Sanal ortamın etkin olduğunu (satır başında `(venv)`) ve kurulumun hatasız bittiğini kontrol edin.
- **SpeechBrain modeli ilk çalıştırmada indirilir.** İnternet bağlantısı gerekir; model dosyaları önbelleğe alınır.
- **Sürüm sabitleme:** `requirements.txt` içindeki tüm sürümler `==` ile sabittir; sürümleri değiştirmeden önce kurulum testlerini yeniden çalıştırın.

## Lisans
[Lisans bilgisini buraya ekleyin]
