# Katkı Kuralları

## 1. Dal kuralı
- Herkes kendi dalında çalışır: `ad/konu` (ör. `abdullah/vad`, `sefkan/veri`, `cevat/kumeleme`).
- `main`'e doğrudan push yapılmaz. GitHub web arayüzünden "Add files via upload" ile `main`'e yükleme de yapılmaz.
- Her değişiklik Pull Request (PR) ile `main`'e girer.

## 2. Pull Request akışı
PR aç, en az 1 ekip arkadaşı inceler, kaptan veya modülün sahibi birleştirir.

```bash
git clone https://github.com/HekimSefkan/Konusmaci-Tespiti.git   # bir kez
cd Konusmaci-Tespiti
git checkout main && git pull origin main      # main'i güncelle
git checkout -b abdullah/vad                   # yeni dal
# ... değişiklik yap ...
git add src/vad.py
git commit -m "Silero VAD ile konuşma bölgeleri bulundu"
git push -u origin abdullah/vad
gh pr create --base main --fill                # veya GitHub'da "Compare & pull request" düğmesi
```

- Dal uzun sürerse `main`'i dalına al: `git pull origin main` (dalındayken), çakışma varsa çöz, tekrar push et.
- İnceleyen kişi PR sayfasında yorum yapar veya "Approve" verir; değişiklik istenirse aynı dala yeni commit eklenir.
- Birleştirmeden önce `python tests/test_arayuz.py` geçmelidir.

## 3. Commit mesajı
Türkçe, kısa, ne yapıldığını söyler.
- İyi: `Silero VAD ile konuşma bölgeleri bulundu`, `RTTM yazma fonksiyonu eklendi`
- Kötü: `düzeltme`, `son hali`, `asdf`, `Add files via upload`

## 4. Hangi dosya nereye?
| Tür | Yer |
|---|---|
| Kod | `src/` |
| Test | `tests/` |
| Belge, diyagram | `docs/` |
| RTTM ve küçük veri | `data/` |
| Ses dosyaları ve 1 MB üstü her dosya | Repoya girmez, ekibin Drive klasöründe durur (bkz. `data/README.md`) |

## 5. Modül sahipliği
| Dosya | Sorumlu |
|---|---|
| `src/ses_girdi.py`, `src/vad.py`, `src/oznitelik.py`, `requirements.txt` | Abdullah |
| `src/kumeleme.py`, `src/cikti.py`, `src/main.py`, `docs/arayuz.md`, `CONTRIBUTING.md` | Cevat |
| `src/degerlendirme.py`, `data/`, `tests/` (veri ve test dosyaları) | Sefkan |

Başkasının modülünde değişiklik gerekiyorsa PR'da sahibi incelemeci olarak eklenir ve ona sorulur.

## 6. Kod stili
- PEP 8. Değişken adları Türkçe olabilir, ama bir dosyada tutarlı olmalıdır.
- Fonksiyon imzaları [docs/arayuz.md](docs/arayuz.md) ile aynı olmalıdır. İmza değişikliği önce `docs/arayuz.md` içinde PR ile yapılır, kod sonra uyarlanır.
