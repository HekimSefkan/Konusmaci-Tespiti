# Ses Kayıtlarında Konuşmacı Değişimlerinin Tespiti (Speaker Diarization)

## Proje Hakkında
Bu proje, Fırat Üniversitesi Bilgisayar Mühendisliği Tasarım Dersi kapsamında **Grup 68** tarafından geliştirilmektedir. Projenin temel amacı; çoklu konuşmacı içeren ses kayıtlarını analiz ederek, "Kim, ne zaman konuştu?" sorusuna yanıt verebilen (Speaker Diarization) otomatik bir makine öğrenmesi sistemi tasarlamaktır. 

Proje kapsamında ses verileri üzerindeki gürültüler temizlenecek, Mel-Frekans Kepstral Katsayıları (MFCC) gibi öznitelikler çıkarılacak ve konuşmacıları ayırt etmek için çeşitli kümeleme algoritmaları (GMM, K-Means vb.) uygulanacaktır.

## Takım Üyeleri
* **Hekim Sefkan Elik** (Takım Kaptanı / Geliştirici)
* **Cevat Can Aygüler** (Araştırmacı / Geliştirici)
* **Abdullah Şeyh Hilal** (Araştırmacı / Test ve Veri Hazırlama)

## Kullanılacak Teknolojiler (Planlanan)
* **Programlama Dili:** Python
* **Sinyal İşleme:** Librosa, PyAudio
* **Makine Öğrenmesi & Derin Öğrenme:** Scikit-learn, PyTorch / TensorFlow (İhtiyaca göre)
* **Diarization Araçları:** Pyannote.audio (Referans ve karşılaştırma için)

## Proje Klasör Yapısı (Taslak)
```text
├── data/               # Kullanılan veri setleri (wav, mp3 formatında)
├── src/                # Kaynak kodlar (Öznitelik çıkarımı, modelleme)
├── docs/               # Proje raporları ve literatür tarama notları
├── notebooks/          # Veri analizi ve testler için Jupyter Notebook dosyaları
└── README.md           # Proje tanıtım dosyası
