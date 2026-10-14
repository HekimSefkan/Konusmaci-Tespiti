# Ses Kayıtlarında Konuşmacı Değişimlerinin Tespiti (Speaker Diarization)

## Proje Hakkında
Bu proje, Fırat Üniversitesi Bilgisayar Mühendisliği Tasarım Dersi kapsamında **Grup 68** tarafından geliştirilmektedir. Projenin temel amacı; çoklu konuşmacı içeren ses kayıtlarını analiz ederek, "Kim, ne zaman konuştu?" sorusuna yanıt verebilen (Speaker Diarization) otomatik bir makine öğrenmesi sistemi tasarlamaktır. 

Proje kapsamında ses verileri üzerindeki gürültüler temizlenecek, Mel-Frekans Kepstral Katsayıları (MFCC) gibi öznitelikler çıkarılacak ve konuşmacıları ayırt etmek için çeşitli kümeleme algoritmaları (GMM, K-Means vb.) uygulanacaktır.

## Takım Üyeleri
* **Hekim Sefkan Elik**: Veri ve Test
* **Abdullah Şeyh Hilal**: Ön İşleme, VAD, Öznitelik ve Ortam
* **Cevat Can Aygüler**: Kümeleme, Çıktı ve Entegrasyon (Hafta 2 takım kaptanı)

## Kullanılacak Teknolojiler (Planlanan)
* **Programlama Dili:** Python
* **Sinyal İşleme:** Librosa, PyAudio
* **Makine Öğrenmesi & Derin Öğrenme:** Scikit-learn, PyTorch / TensorFlow (İhtiyaca göre)
* **Diarization Araçları:** Pyannote.audio (Referans ve karşılaştırma için)

## Proje Klasör Yapısı
```text
├── data/               # RTTM referansları, veri araçları (ses dosyaları burada değil, Drive'da)
│   ├── sample/         # Örnek kayıtların referans RTTM'leri
│   └── tools/          # Veri hazırlama betikleri
├── src/                # Kaynak kodlar (her dosyanın sahibi CONTRIBUTING.md'de)
├── tests/              # Testler (python tests/test_arayuz.py)
├── docs/               # Belgeler; modüller arası arayüz sözleşmesi: docs/arayuz.md
├── requirements.txt    # Python bağımlılıkları
├── CONTRIBUTING.md     # Dal, PR ve commit kuralları
└── README.md           # Proje tanıtım dosyası
```

## Kullanım (planlanan)
```bash
python -m src.main <girdi_ses> -o <cikti.rttm>
```
Çıkış kodları: 0 başarılı, 1 girdi hatası (bulunamadı/bozuk/desteklenmeyen), 2 iç hata. Modül fonksiyonları ve veri tipleri için bkz. [docs/arayuz.md](docs/arayuz.md); katkı kuralları için [CONTRIBUTING.md](CONTRIBUTING.md).

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
   git clone https://github.com/HekimSefkan/Konusmaci-Tespiti.git
   cd Konusmaci-Tespiti
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
   !git clone https://github.com/HekimSefkan/Konusmaci-Tespiti.git
   %cd Konusmaci-Tespiti
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
