# Hafta 8: web yayını ve Google girişi

Uygulama varsayılan olarak yalnızca yerel HTTP adresinde çalışır. Bu dalda doğrulama localhost HTTP ile yapıldı; genel bir hostta yayın yapılmadı ve sabit adres seçilmedi. Docker varsayılanı `127.0.0.1:8765` üzerinden yalnızca aynı makineden erişilir. `--share` mevcut yerel Gradio akışında isteğe bağlı, geçici bir genel bağlantı üretir; sabit web adresi değildir. Google girişi de isteğe bağlıdır; gerçek sağlayıcı ayarları henüz bu depoya eklenmemiştir.

## Yerel kullanım

Mevcut akış değişmedi:

```bash
python app.py
```

Varsayılan adres `http://127.0.0.1:7860`'dır. Google modunu seçmeden yerel e-posta/şifre hesapları ve `KT_GECMIS_DIZIN` ayarı kullanılmaya devam eder.

## Docker kullanımı

Docker Engine ve Compose kurulduktan sonra:

```bash
docker compose up --build -d
```

Arayüz `http://127.0.0.1:8765` adresinde açılır. Durdurmak için `docker compose down`, kalıcı uygulama verisini de kaldırmak istenirse `docker compose down -v` çalıştırın. Named `kt_state` volume hesap dosyasını, analiz geçmişini ve çalışma sırasında indirilen modeli `/var/lib/kt` altında tutar. Compose yayınlama portunu varsayılan olarak loopback'e bağlar. Başka makinelere açık hale getirmeden önce TLS kullanan bir ters vekil (reverse proxy) ve kararlı alan adı yapılandırın.

Görüntü `python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64` tabanını ve sürümü sabitlenmiş `requirements.txt` paketlerini kullanır. Yalnızca `app.py`, `src/` ve bağımlılık listesi imaja kopyalanır; `data/`, notebook'lar, hesaplar, geçmiş, model ağırlıkları ve OAuth sırları imajda bulunmaz. ECAPA modeli ilk kullanımda kalıcı volume'a indirilir. Konteyner root kullanıcısı olmadan çalışır. Model CPU kullanırken Compose `KT_ECAPA_CPU_THREADS=4` ve `KT_ECAPA_CPU_INTEROP_THREADS=1` değerlerini verir; makinenize göre `.env` dosyasında pozitif tam sayılarla değiştirebilirsiniz.

## CPU ve kümeleme CLI

CPU iş parçacığı ayarları yerel `python app.py` ve `python -m src.main` komutlarına da uygulanır. İki `KT_ECAPA_CPU_*` değişkeni de ayarlanmamışsa uygulamanın mevcut davranışı sürer; Silero içe aktarımı CPU iş parçacığı sayısını 1'e ayarlayabilir. CUDA varsa bu CPU ayarları uygulanmaz.

Tek dosyalı komut `python -m src.main ses.wav -o sonuc.rttm [--yontem ecapa|mfcc] [--konusmaci-sayisi N] [--esik E] [--min-destek-pencere N] [--maks-sure SN]` biçimindedir. `--min-destek-pencere` yalnızca otomatik konuşmacı sayısı kipinde, düşük destekli kümeleri en yakın ana küme AHC eşiği içindeyse birleştirir; varsayılan 0 ile kapalıdır. `--maks-sure 0` süre sınırını kaldırır.

## Google OIDC'yi etkinleştirme

Yerel HTTP doğrulaması `http://127.0.0.1:8765` üzerinden yapıldı; bu modda callback adresi `http://127.0.0.1:8765/auth/callback` olur. Otomatik testlerde sahte OIDC sağlayıcısı kullanıldı; gerçek Google hesabı veya kimlik bilgisi kullanılmadı. Kod, HTTP callback'i yalnızca loopback adreslerinde kabul eder; dışarıdan görülen `KT_PUBLIC_URL` HTTPS olmalıdır. `--google-auth` modu `Authlib` üzerinden Google'ın OIDC keşif belgesini kullanır. Authorization Code akışında state, nonce ve S256 PKCE kullanılır. ID token imza, issuer, audience, süre ve nonce doğrulamasından geçmeden oturum açılmaz. Ek olarak `email_verified` tam olarak `true` olmalıdır.

Google Cloud Console'da bir OAuth istemcisi oluşturduktan sonra:

1. OAuth izin ekranını ve test/yayın durumunu kurumunuzun kullanımına göre hazırlayın.
2. Uygulama türünü **Web application** seçin.
3. Authorized redirect URI değerini kararlı adresin sonuna `/auth/callback` ekleyerek girin. Örneğin `https://speech.example.edu/auth/callback`. Uygulama yalnızca alan adının kökünde çalışır; path-prefix desteklenmez.
4. Proje kökünde `.env.example` dosyasını `.env` olarak kopyalayın. Bu yerel dosya Git ve Docker build context dışında tutulur.
5. Aşağıdaki ortam değişkenlerini `.env` içinde doldurun. `KT_SESSION_SECRET` için `openssl rand -hex 32` kullanın; değer uygulama oturumlarını imzalamak ve Google `sub` değerinden kalıcı kullanıcı dizin kimliği üretmek için sabit kalmalıdır.

| Değişken | Değer |
|---|---|
| `KT_GOOGLE_CLIENT_ID` | Google Web OAuth istemci kimliği |
| `KT_GOOGLE_CLIENT_SECRET` | Google Web OAuth istemci sırrı; gizli yönetim alanında saklayın |
| `KT_PUBLIC_URL` | HTTPS ile başlayan, uygulamanın dışarıdan görülen kök adresi; yol, sorgu veya fragment yok |
| `KT_SESSION_SECRET` | En az 32 bayt rastgele sır; kalıcı ve gizli tutun |
| `KT_WEB_PORT` | Yerel Docker portu; `8765` varsayılanı |
| `KT_ECAPA_CPU_THREADS` | ECAPA CPU işlem iş parçacığı sayısı; `4` varsayılanı |
| `KT_ECAPA_CPU_INTEROP_THREADS` | Torch CPU inter-op iş parçacığı sayısı; `1` varsayılanı |

Google modunu aynı kalıcı volume ile servis olarak başlatın:

```bash
docker compose -f compose.yaml -f compose.google.yaml up --build -d
```

Bu komut `.env` değerlerini ve `compose.yaml` içindeki named volume/loopback portunu kullanır. Genel yayında HTTPS'i ters vekil sonlandırmalı; `KT_PUBLIC_URL` dış HTTPS kök adresi olmalı ve proxy `/auth/callback` yolunu değiştirmeden uygulamaya iletmelidir. `KT_SESSION_SECRET` değişirse türetilen geçmiş kullanıcı kimlikleri de değişir; bu geçmişin yeni oturumda görünmemesine neden olur.

Google modu yerel hesap dosyasını kullanmaz ve doğrulanmış Google hesabını mevcut e-posta hesabıyla birleştirmez. Hesap izolasyonu doğrulanmış `sub` değerinden HMAC ile türetilmiş kimliğe dayanır; e-posta adresi anahtar olarak kullanılmaz. Opaque giriş cookie'si `HttpOnly`, `SameSite=Lax`, HTTPS'te `Secure` olarak verilir. Gradio istemci durumu cookie'den ayrı, tahmin edilmesi zor bir token taşır; her işlem token'ı sunucu tarafında gelen OIDC cookie'siyle eşleştirir. Gradio'nun upload, download, API ve UI yolları ayrıca cookie doğrulama bağımlılığı arkasındadır; kökte yalnızca giriş sayfası açıktır. Oturumlar 12 saat geçerlidir ve süreç belleğinde tutulduğu için servis yeniden başlayınca kullanıcıların yeniden giriş yapması gerekir. Çıkış yalnızca aynı origin'den gelen POST isteğiyle kabul edilir.

## Yerel doğrulama listesi

- `docker compose config` Compose değerlerinin çözüldüğünü göstermeli.
- `docker compose up --build -d` sonrasında `http://127.0.0.1:8765` arayüzü dönmeli.
- İmaj içinden `data/`, `notebooks/`, hesap, geçmiş veya model ağırlığı kopyalanmadığını doğrulayın.
- Named volume ile oluşturulan yerel hesabın ve analiz geçmişinin `docker compose restart` sonrasında kaldığını kontrol edin.
- Google ayarları eklendikten sonra redirect URI'nin Console'daki değerle birebir aynı olduğunu, kullanıcı iptalinde oturum açılmadığını, çıkıştan sonra eski sayfadaki işlem token'ının reddedildiğini kontrol edin.
