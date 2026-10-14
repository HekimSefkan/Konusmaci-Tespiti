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
- **Python 3.10 veya 3.12** (64 bit). `requirements.txt` her iki sürüm için Windows ve Linux'ta hazır paketlerle (wheel) kurulur; C++ derleyici gerekmez.
- **Git**
- **ffmpeg** (mp3 dosyalarını okumak için)
- Yaklaşık 3 GB boş disk alanı (Windows'ta ölçülen: sanal ortam ~1,9 GB, ECAPA modeli ~0,1 GB)

### Windows

1. **Python'u kurun.** python.org'dan Python 3.10 (veya 3.12) yükleyicisini indirin. Kurulum ekranında **"Add Python to PATH"** kutusunu işaretleyin.
2. **Sürümü doğrulayın.** PowerShell açın (3.12 kullanıyorsanız `-3.12` yazın):
   ```powershell
   py -3.10 --version
   ```
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
5. **Sanal ortam oluşturun ve etkinleştirin.**
   ```powershell
   py -3.10 -m venv .venv
   .venv\Scriptsctivate
   ```
   "Bu sistemde betik çalıştırma devre dışı" hatası alırsanız bir kez şunu çalıştırın ve tekrar deneyin:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
   Etkinleşince satır başında `(.venv)` görünür.
6. **Bağımlılıkları kurun.**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Ölçülen süre: temiz kurulum yaklaşık 10 dakika (Windows 11, Python 3.10, ev interneti; torch CPU paketi ~200 MB). Süre bağlantı hızına göre değişir.
7. **Kurulumu doğrulayın.**
   ```powershell
   python tests/kurulum_kontrol.py
   ```
   Paketlerin yüklenmesini ve küçük duman testlerini (MFCC, Silero VAD, WebRTC VAD, DER, SpeechBrain ECAPA) çalıştırır; sonunda `KURULUM: OK` yazmalıdır. İlk çalıştırmada ECAPA modeli (~85 MB) indirilir ve kullanıcı önbelleğine (`~/.cache/konusmaci-tespiti`) kaydedilir; modeli indirmeden denemek için `python tests/kurulum_kontrol.py --hizli`.

### Google Colab

Colab'ın varsayılan Python'u 3.10 değil, daha yenidir (Python 3.12 veya üstü olabilir; `!python --version` ile bakın). `requirements.txt` 3.12 için doğrulanmıştır; Colab'ın gerçekte denenmesi ekipçe yapılacaktır.

1. [colab.research.google.com](https://colab.research.google.com) adresinde **Yeni not defteri** açın. GPU gerekmez; varsayılan CPU çalışma zamanı yeterlidir.
2. **Depoyu indirin.**
   ```python
   !git clone https://github.com/HekimSefkan/Konusmaci-Tespiti.git
   %cd Konusmaci-Tespiti
   ```
3. **ffmpeg'i kurun** (Colab'da genelde hazırdır).
   ```python
   !apt-get install -y ffmpeg
   ```
4. **Bağımlılıkları kurun.**
   ```python
   !pip install -r requirements.txt
   ```
   Colab'da torch önceden kuruludur; `requirements.txt` torch'u `2.5.1` (CPU sürümü, `--extra-index-url` satırı sayesinde) olarak sabitlediği için Colab'ın torch'u bu sürümle değiştirilir. Böylece Windows ve Colab aynı sürümü kullanır. Ek dosya gerekmez.
5. **Çalışma zamanını yeniden başlatın.** Kurulum Colab'ın önceden yüklü numpy gibi paketlerini değiştirir: *Çalışma zamanı → Oturumu yeniden başlat*. Yeniden başlattıktan sonra 2–4. adımları tekrarlamanız gerekmez.
6. **Kurulumu doğrulayın.**
   ```python
   !python tests/kurulum_kontrol.py
   ```
   Çıktının sonunda `KURULUM: OK` yazmalıdır.

### Notlar ve sorun giderme

- **`webrtcvad`:** Artık C++ derleyici gerekmez. `requirements.txt` kaynak paket olan `webrtcvad` yerine hazır paketi olan `webrtcvad-wheels` kullanır; kodda yine `import webrtcvad` yazılır. Eski bir ortamda `webrtcvad` kuruluysa `pip uninstall webrtcvad` yapıp `pip install -r requirements.txt` komutunu tekrar çalıştırın.
- **mp3 açılmıyor:** `ffmpeg -version` komutunun çalıştığını ve ffmpeg'in PATH'te olduğunu doğrulayın.
- **Colab'da sürüm çakışması uyarısı:** Colab önceden yüklü paketlerle çakışma bildirebilir (yalnızca uyarıdır, çoğu zaman çalışmayı engellemez). Sorun çıkarsa hata mesajındaki paketin sürümünü not edin, `requirements.txt` dosyasını buna göre güncelleyin ve değişikliği bu README'ye yansıtın.
- **`ModuleNotFoundError`:** Sanal ortamın etkin olduğunu (satır başında `(.venv)`) ve kurulumun hatasız bittiğini kontrol edin.
- **SpeechBrain modeli ilk çalıştırmada indirilir.** İnternet bağlantısı gerekir; model dosyaları önbelleğe alınır. Windows'ta modeli çağırırken `local_strategy=LocalStrategy.COPY` verin; aksi halde "Gereken ayrıcalık istemci tarafından sağlanmıyor" (WinError 1314) hatası çıkar (`tests/kurulum_kontrol.py` örnek kullanım içerir).
- **Silero VAD ve Türkçe karakterli yol:** Windows'ta `load_silero_vad()` varsayılan modeli `Sınıf` gibi ASCII dışı yollarda açamaz; `load_silero_vad(onnx=True)` kullanın.
- **Sürüm sabitleme:** `requirements.txt` içindeki tüm sürümler `==` ile sabittir; sürümleri değiştirmeden önce `python tests/kurulum_kontrol.py` komutunu yeniden çalıştırın.
