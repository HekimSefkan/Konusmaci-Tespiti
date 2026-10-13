# Veri (data/)

Grup_68 — "Ses Kayıtlarında Konuşmacı Değişimlerinin Tespiti" projesinin örnek veri paketi ve veri araçları.

```text
data/
├── README.md        # Bu dosya
├── sample/          # Kırpılmış örnek kayıtların referans RTTM'leri (ses dosyaları burada DEĞİL)
│   └── edge/        # Kenar durum ve yapay gürültü test dosyalarının RTTM'leri
└── tools/           # Veri betikleri (yalnızca Python standart kütüphanesi + ffmpeg)
```

## 1. Kaynak ve sürüm

- **Veri seti:** VoxConverse, **sürüm 0.3**. Sürüm 0.3, VoxConverse GitHub deposunun `master` dalıdır; depo README'sine göre bu sürümde bazı test RTTM hataları düzeltilmiştir. Önceki sürüm `ver0.2` dalındadır.
  - Etiketler (RTTM): <https://github.com/joonson/voxconverse>. Kullanılan commit: `24bf60b`, 2022-07-25.
  - Proje sayfası: <https://www.robots.ox.ac.uk/~vgg/data/voxconverse/>
- **Kullanılan bölüm:** Yalnızca **dev** seti: 216 kayıt, `voxconverse_dev_wav.zip`, 1.988.647.478 bayt.
- **Zip bütünlüğü:** Arşiv Colab'da tam olarak test edildi: 217 giriş, bozuk dosya yok. Ekibin Drive kopyasının bayt boyutu resmi dosyayla birebir eşleşiyor. Örnekler çıkarılırken zipfile her dosyanın CRC-32 kontrolünü yaptı ve 5 dosyanın beşi de sorunsuz geçti.

## 2. Lisans ve atıf

- VoxConverse, araştırma amaçlı olarak **Creative Commons Attribution 4.0 International (CC BY 4.0)** lisansıyla sunulur. **Orijinal videoların telif hakkı video sahiplerindedir.** Kaynak: VoxConverse GitHub README.
- Kayıtlar YouTube'daki siyasi tartışma, haber ve benzeri programlardan alınmıştır. Bu yüzden ses dosyalarını açık repoya koymuyoruz; repoda yalnızca etiketler (RTTM) var.
- Veri setini kullanan her çalışmada şu makaleye atıf yapılmalıdır:

  > J. S. Chung, J. Huh, A. Nagrani, T. Afouras, A. Zisserman, "Spot the conversation: speaker diarisation in the wild", *Interspeech 2020*, s. 299–303. doi:10.21437/Interspeech.2020-2337

## 3. Ses dosyaları nerede?

- Kırpılmış örnek sesler ekibin ortak Drive klasöründe durur: **`BilgisayarTasarim_Proje68/Hafta2/ornek_ses/`**. Kenar durum dosyaları aynı klasörün altında, `edge/` içindedir.
- Drive bağlantısı: <https://drive.google.com/drive/folders/1zR0s5wEpKxUicSy4hBGeFZljpCZtpA5G?usp=drive_link>
- **Aynı dosyaları kendiniz üretmek için:** `voxconverse_dev_wav.zip` dosyasını Drive'daki `veri_kaynak/` klasöründen ya da resmi sayfadan edinin. VoxConverse deposunu klonlayıp aşağıdaki komutları çalıştırın. Yollar parametre olarak verilir; `<...>` kısımlarını kendi yollarınızla değiştirin.

```bash
python data/tools/ornek_kirp.py --zip <voxconverse_dev_wav.zip> --rttm-klasor <voxconverse/dev> --kayit ikgcq --bas 28 --sure 90 --wav-cikti <ses_klasoru> --rttm-cikti <rttm_klasoru> --gecici <gecici_klasor>
```

| Kayıt | `ornek_kirp.py` parametreleri |
|---|---|
| abjxc | `--kayit abjxc` (`--sure` yok, kırpılmadan alınır) |
| ikgcq | `--kayit ikgcq --bas 28 --sure 90` |
| sldwj | `--kayit sldwj --bas 0 --sure 90` |
| luvfz | `--kayit luvfz --bas 113 --sure 90` |
| migzj | `--kayit migzj --bas 66.5 --sure 70` |

Yapay gürültülü ve kenar durum dosyaları için:

```bash
python data/tools/gurultu_ekle.py --wav <ses_klasoru>/ikgcq_028.0-118.0.wav --rttm data/sample/ikgcq_028.0-118.0.rttm --snr 10 5 --wav-cikti <edge_klasoru> --rttm-cikti <rttm_klasoru>
```

```bash
python data/tools/edge_uret.py --kaynak-wav <ses_klasoru>/ikgcq_028.0-118.0.wav --kaynak-rttm data/sample/ikgcq_028.0-118.0.rttm --kisa-bas 10 --wav-cikti <edge_klasoru> --rttm-cikti <rttm_klasoru>
```

## 4. Örnek kayıt tablosu

**Nasıl hesaplandı:**
- Değerler `data/tools/rttm_istatistik.py` ile RTTM'lerden hesaplandı. WAV süreleri `ffprobe` ile ölçüldü.
- Tüm WAV'lar 16 kHz, mono, 16-bit PCM'dir. Orijinal dev WAV'ları da zaten bu biçimdeydi.
- **Kırpma:** Dört kayıt 60–90 sn'lik parçalara kırpıldı. Pencereler, kaydın tüm konuşmacıları içeride olacak (her biri en az 2 sn konuşacak) şekilde seçildi. abjxc kırpılmadı.

**RTTM kırpma kuralları:**
- Pencerenin dışındaki segmentler atıldı.
- Sınırda bölünen segmentler sınırda kesildi.
- Zamanlar 0'a kaydırıldı.
- 2. alan (dosya adı) yeni dosya adıyla değiştirildi.

**Sütunların anlamı:**
- **RTTM süresi:** Son segmentin bitişi. WAV'ın gerçek süresinden kısa olabilir.
- **Örtüşme:** Örtüşen konuşma süresi / konuşma süresi.
- **Değişim:** Konuşmacı değişim noktası sayısı. Tanımı bir sonraki bölümde.

| Dosya (kırpılmış) | Orijinal kayıt | Orijinal WAV süresi | Orijinal: konuşmacı / RTTM süresi / örtüşme / değişim | Pencere (orijinal zaman) | Kırpılmış WAV süresi | Kırpılmış: konuşmacı / RTTM süresi / örtüşme / değişim | Not |
|---|---|---|---|---|---|---|---|
| `abjxc_tam` | abjxc | 68,38 sn | 1 / 64,64 sn / %0 / 0 | tamamı | 68,38 sn | 1 / 64,64 sn / %0 / 0 | **Kırpılmamış** |
| `ikgcq_028.0-118.0` | ikgcq | 129,02 sn | 2 / 124,32 sn / %1,13 / 10 | 28,0–118,0 sn | 90,00 sn | 2 / 89,76 sn / %0 / 8 | Sınırda kesilen segment yok |
| `sldwj_000.0-090.0` | sldwj | 140,33 sn | 3 / 140,28 sn / %0 / 15 | 0,0–90,0 sn | 90,00 sn | 3 / 90,00 sn / %0 / 10 | spk02 segmenti [82,68–91,84] → [82,68–90,00] kesildi |
| `luvfz_113.0-203.0` | luvfz | 219,38 sn | 7 / 202,84 sn / %3,45 / 47 | 113,0–203,0 sn | 90,00 sn | 7 / 89,84 sn / %3,76 / 23 | spk03 segmenti [111,80–117,60] → [113,00–117,60] kesildi |
| `migzj_066.5-136.5` | migzj | 175,63 sn | 4 / 175,60 sn / %34,32 / 26 | 66,5–136,5 sn | 70,00 sn | 4 / 69,94 sn / %29,56 / 12 | Sınırda kesilen segment yok; belirgin örtüşme |

Kırpılmış parçaların hiçbirinde konuşmacı sayısı değişmedi.

**Kenar durum ve yapay gürültü dosyaları** (sesler Drive'da `ornek_ses/edge/` altında, RTTM'ler `sample/edge/` altında):

| Dosya | Açıklama | Referans RTTM |
|---|---|---|
| `ikgcq_028.0-118.0_yapay_gurultu_snr10dB.wav` | **Yapay gürültü eklenmiş test dosyası**: temiz ikgcq parçasına SNR 10 dB beyaz gürültü eklendi (seed 68). Gerçek gürültülü ortam kaydı **değildir**. | Kaynak RTTM ile aynı zamanlar |
| `ikgcq_028.0-118.0_yapay_gurultu_snr5dB.wav` | Aynı işlem, SNR 5 dB | Kaynak RTTM ile aynı zamanlar |
| `kisa_1sn.wav` | ikgcq parçasının 10,0–11,0 sn aralığı (1 sn, tek konuşmacı) | `kisa_1sn.rttm` (1 segment) |
| `sessiz_5sn.wav` | 5 sn, tamamen sıfır örnek | `sessiz_5sn.rttm`: **boş dosya** |
| `bozuk_yarim.wav` | Geçerli bir WAV'ın ilk yarısı: başlık sağlam, ama başlıkta yazan veri uzunluğu dosyada yok. **Not:** ffprobe bu dosyayı hata vermeden 45 sn olarak okuyor; yani bazı okuyucular bozukluğu fark etmiyor. | Yok |
| `desteklenmeyen_bicim.xyz` | Ses olmayan, desteklenmeyen uzantılı küçük bir metin dosyası | Yok |

**Sessiz dosyada referans neden boş?** Kayıtta hiç konuşma olmadığı için doğru referans boş bir RTTM'dir. Bu durumda DER'in paydası (toplam referans konuşma süresi) 0 olur ve oran tanımsızdır. Bu yüzden sessiz dosya için DER yerine **yanlış alarm süresi (sn)** raporlanır; beklenen değer 0 sn'dir.

pyannote.metrics 4.1 bu durumu şöyle ele alıyor:
- Boş referans ve boş sistem çıktısında DER = 0 döner.
- Boş referansa karşı herhangi bir yanlış alarmda DER = 1,0 (%100) döner.

## 5. RTTM biçimi

RTTM (Rich Transcription Time Marked) dosyalarında her satır bir konuşma segmentidir. Satırda boşlukla ayrılmış 10 alan bulunur. Örnek:

```text
SPEAKER ikgcq_028.0-118.0 1 29.080 1.480 <NA> <NA> spk01 <NA> <NA>
```

| # | Alan | Örnekteki değer | Anlamı |
|---|---|---|---|
| 1 | Tür | `SPEAKER` | Segment türü; diarization için her zaman `SPEAKER` |
| 2 | Dosya kimliği | `ikgcq_028.0-118.0` | Ses dosyasının uzantısız adı |
| 3 | Kanal | `1` | Kanal numarası (1'den başlar) |
| 4 | Başlangıç | `29.080` | Segmentin başladığı an (sn) |
| 5 | **Süre** | `1.480` | Segmentin **SÜRESİ** (sn). Bitiş değildir; bitiş = alan 4 + alan 5 = 30,56 sn |
| 6 | Ortografi | `<NA>` | Kullanılmaz |
| 7 | Konuşmacı türü | `<NA>` | Kullanılmaz |
| 8 | Konuşmacı adı | `spk01` | Konuşmacı etiketi; yalnızca dosya içinde tekildir |
| 9 | Güven skoru | `<NA>` | Kullanılmaz |
| 10 | Sinyal ön-bakış süresi | `<NA>` | Kullanılmaz |

Alan tanımlarının kaynağı: dscore README (<https://github.com/nryant/dscore>).

Örtüşen konuşmada aynı zaman aralığı için birden fazla satır bulunur.

**Konuşmacı değişim noktası tanımı** (`rttm_istatistik.py`):
- Örtüşen ve sessiz bölgeler atılır.
- Kalan tek konuşmacılı (non-overlap) parçalardan aynı konuşmacıya ait ardışık olanlar birleştirilir.
- Zaman sırasına göre ardışık iki parçanın konuşmacıları farklıysa, ikinci parçanın başlangıcı bir değişim noktasıdır.

## Betikler (data/tools/)

| Betik | Görevi |
|---|---|
| `rttm_istatistik.py` | Süre, konuşmacı sayısı, konuşma süresi, örtüşme oranı ve değişim noktası sayısını hesaplar. İsteğe bağlı olarak 90 sn'lik pencere analizi yapar ve CSV çıktısı verir. |
| `ornek_kirp.py` | Zip'ten tek bir kaydı çıkarır (zip'in tamamını açmaz). WAV'ı ffmpeg ile, RTTM'i aynı aralığa göre kırpar. |
| `gurultu_ekle.py` | Belirtilen SNR'da beyaz gürültü ekleyerek yapay gürültülü test dosyası üretir (sabit seed). |
| `edge_uret.py` | Bozuk, sessiz, 1 sn'lik ve desteklenmeyen biçimdeki kenar durum dosyalarını üretir. |

- Betikler yalnızca Python 3.8+ standart kütüphanesini kullanır. `ornek_kirp.py` için ayrıca PATH'te `ffmpeg` gerekir.
- Hiçbir betik var olan bir dosyanın üzerine yazmaz; aynı ad varsa `_v2` ekler.
- Her betiğin parametreleri `-h` ile görülebilir.
