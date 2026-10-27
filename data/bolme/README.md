# Eğitim / değerlendirme bölmesi (data/bolme/)

VoxConverse **dev** setinin 216 kaydı iki yarıya ayrılır:

| Dosya | Kullanım |
|---|---|
| `egitim.txt` | AHC eşik taraması ve konuşmacı değişim modelinin eğitimi (model seçimi, olasılık eşiği, tepe ayarları) |
| `degerlendirme.txt` | E1–E5 deneylerinin raporlanan sonuçları; bu kayıtlar hiçbir ayarda kullanılmaz |

Her satırda bir kayıt kimliği vardır (VoxConverse RTTM dosya adı, uzantısız). Listeler alfabetik sıralıdır.

## Nasıl üretildi?

```bash
python -m src.bolme --rttm-dizin <voxconverse/dev> --cikti-dizin <klasor>
```

- Kaynak: VoxConverse `master` dalı (sürüm 0.3), `dev/` altındaki 216 RTTM.
- Kayıtlar referanstaki farklı konuşmacı sayısına göre dört tabakaya ayrılır: **1**, **2**, **3–4**, **5+**.
- Her tabaka sıralanır, `random.Random(68)` ile karıştırılır ve ikiye bölünür. Tek sayılı tabakalarda artan kayıt sırayla bir eğitime, bir değerlendirmeye verilir; toplamda yaklaşık 108 / 108 olur.
- Aynı RTTM'ler ve aynı seed her zaman aynı bölmeyi verir.

## Durum

Liste dosyaları Colab'da (`notebooks/hafta4_deneyler.ipynb`, bölüm 4) üretilip Drive'a yazılır (`Hafta4/bolme/`). İçerikleri doğrulandıktan sonra bu klasöre eklenecek.
