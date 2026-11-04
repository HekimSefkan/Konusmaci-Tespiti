# Hafta 5 doğrulama araçları

Bu araçlar mevcut ses→RTTM hattını ölçer. Varsayılan ECAPA yöntemi ve repo içindeki eşikler değiştirilmez. Sesler, modeller, önbellekler ve üretilen sonuçlar GitHub'a eklenmez.

| Araç | İşlev |
|---|---|
| `src.hafta5` | Eğitim tarafında E5 eşiği seçimi, kabul ve CPU ölçümü, kayıt karşılaştırması |
| `src.hafta5_gorsel` | Birden fazla RTTM'i aynı zaman ekseninde çizme, özet CSV grafiği |
| `src.vad_ornek_olcum` | Beş örnek ses ve dinleme listesindeki kesitlerde ham WebRTC/Silero karşılaştırması |
| `notebooks/hafta5_dogrulama.ipynb` | Colab kurulum, test ve örnek dosya doğrulaması |

Hafta 5 modülleri, açık Hafta 4 ek görev PR'ındaki `src.gorsel` ve `src.vad_olcum` araçlarıyla isim çakışmaması için ayrı adlar kullanır. Hafta 4 notebook'u değiştirilmez.

## Girdiler

`--kok`, `Hafta2/` ve `Hafta4/` klasörlerinin dış proje köküdür. Beş örnek WAV ve `edge/` dosyaları `Hafta2/ornek_ses/` altında olmalı. E5 deneyi için `Hafta4/bolme/{egitim,degerlendirme}.txt`, `Hafta4/modeller/degisim_modeli.joblib`, `Hafta4/onbellek/` altındaki MFCC/ECAPA/VAD NPZ'leri gerekir. Model ağırlıkları `Hafta4/modeller/ecapa/` içinden okunur.

Analiz, `Hafta4/sonuclar/` içindeki E1–E5 CSV'lerini, `ozet.csv` dosyasını ve E4/E5 RTTM klasörlerini kullanır. VAD örnek ölçümü ayrıca `Hafta4/sonuclar/dinleme_listesi.csv` ile `Hafta4/dinleme/` içindeki kesitleri okur. Örnek referansları repodaki `data/sample/` içindedir; `--referans` kullanılan VoxConverse sürümünün tam dev RTTM klasörüdür.

Eğitim ve değerlendirme listeleri kesişirse deney durdurulur. MFCC/ECAPA pencere dizileri uyuşmazsa önbellek reddedilir.

## Komutlar

Depo klasöründe çalıştırın; dış kökü ve referans yolunu kendi dizininize göre değiştirin.

```bash
python -m src.hafta5 --kok .. --referans ../Hafta5/girdi/voxconverse_dev_rttm --is deney
python -m src.hafta5 --kok .. --referans ../Hafta5/girdi/voxconverse_dev_rttm --is analiz
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m src.hafta5 --kok .. --is kabul --benchmark
OMP_NUM_THREADS=1 python -m src.vad_ornek_olcum --kok .. --referans ../Hafta5/girdi/voxconverse_dev_rttm
python -m src.hafta5_gorsel --rttm Referans=ref.rttm --rttm E4=e4.rttm --rttm E5=e5.rttm -o karsilastirma.png
python -m src.hafta5_gorsel --csv ozet.csv -o deneyler.png
```

Çıktılar `--kok/Hafta5/` altında yazılır. Aynı komut tekrar çalıştırılırsa aynı adlı ölçüm dosyaları yenilenir; önceki çalışma klasörünü ayrıca saklayın. Hafta 4 girdileri değiştirilmez. `ortam.json` Python/paket/commit bilgisini, `girdi_sha256.json` kod, model, bölme, ses, önbellek ve referansların hashlerini kaydeder. Bu dosyalarda yerel yollar bulunur; herkese açık paylaşmadan önce içeriği kontrol edin.

## Ölçümlerin anlamı

E5 aday eşikleri yalnız eğitim listesinde karşılaştırılır, seçilen eşik değerlendirme listesinde ölçülür. Yeni eşik `secilen_ayar.json` içine yazılır; `src/varsayilanlar.json` değiştirilmez. Değerlendirme kümesi önceki analizlerde görüldüğü için sonuç hiç görülmemiş bir final test başarısı sayılmaz.

Kabul DER'i beş kaydın aritmetik ortalamasıdır. Değişim F1, referansta değişim bulunan kayıtların aritmetik ortalamasıdır. DER için tam collar 0.5 saniye (±0.25), değişim eşleşme toleransı ±0.5 saniyedir. MFCC ve ECAPA ayrı tutulur; bir yöntemin `False` sonucu başka yöntemin başarısıyla değiştirilmez.

Kenar durumlarında çıkış kodu ve RTTM içeriği birlikte kontrol edilir. Beklenmeyen sonuçta CSV korunur ve komut hata verir. CPU ölçümü beş kez gerçek ses okuma ve öznitelik çıkarımı yapar; model önceden yüklenmiştir, torch bir iş parçacığı kullanır ve girdi başlığından 90 saniye olduğu doğrulanır. Önbellekli deney süreleri bu ölçümün yerine geçmez.

`vad_ornek_olcum`, referans konuşma aralıklarının birleşimini ve VAD kararlarını 10 ms merkez ızgarasında karşılaştırır. WebRTC ham 30 ms kararları kullanılır; kısa konuşma silme ve boşluk birleştirme uygulanmaz. Bu nedenle Hafta 4 ek görev aracındaki yumuşatılmış tam-kayıt WebRTC ölçümleriyle aynı yöntem olarak sunulmamalıdır. Gürültülü kesitler küçük, seçilmiş bir kümedir; tüm veri setini temsil ettikleri varsayılmaz.

## Colab doğrulaması ve hata kayıtları

Yeni notebook model eğitimi veya toplu E1–E5 deneylerini başlatmaz. Mevcut Colab torch/torchaudio paketlerini korur. Pip stdout/stderr tek günlükte saklanır; hata durumunda ayrıntı görünür. Otomatik sürüm gevşetme uygulanmaz.

4 Kasım 2026 kullanıcı doğrulamasında main `3018b19` sürümünde 128 test geçti, referans klasörü olmadığından bir test atlandı ve `KURULUM: OK` alındı. İlk 22 satırlı duman testinde kısa/sessiz dosyalar iki yöntemde Errno 95 verdi. Aynı dört satır ayrı süreçlerde tekrarlandığında tümü çıkış 0 ve doğru RTTM içerikleriyle geçti. İlk hatanın teknik nedeni kesinleşmedi. Bu geçmiş gözlem yeni notebook'un Colab'da çalıştırıldığı anlamına gelmez.

NumPy için `uninstall-no-record-file` hatası alınırsa notebook'taki belirli onarım yönergesini kullanın; genel bağımlılık çatışmalarını bu komutla geçiştirmeyin. Paket kurulumu başarılı tamamlanınca oturumu yeniden başlatın. İlk başarısız test kaydını koruyup tekrar sonucunu ayrı kaydedin.
