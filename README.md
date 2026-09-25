# Jev Review Lab

Türkçe ürün yorumlarında **olumlu / olumsuz duygu sınıflandırması**. Gerçek veri, ücretsiz yerel karşılaştırmalar ve 5 kredilik hesap için sınırlı Jev pilotu.

## Hazır sonuçları aç

`index.html` dosyasını tarayıcıda aç veya `start.ps1` çalıştır. `artifacts/report.html` de aynı arayüzün bağımsız kopyasıdır. Rapor bağımsızdır; sunucu, Docker, internet veya API anahtarı gerektirmez. Test/pilot görünümü, hata matrisi, yorum arama ve yanlış tahmin filtresi içerir. Dış modeller çalıştırılana kadar puanları gösterilmez.

## Veri

- Kaynak: [Fatih Barmanbay — Turkish Product Reviews](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews).
- Sabit sürüm: `3c8bef32a2b6cb1f70d686783edeecaec6287cbf`; 235.165 gerçek yorum; ham indirme yaklaşık 24 MB.
- Kaynak alanları: `sentence`, `sentiment`; 0 = negative, 1 = positive.
- 20–1.200 karakter aralığı; metin kesilmez. Türkçe küçük harf ve noktalama normalleştirmesiyle tekrarlar bulunur. Aynı normalleştirilmiş metnin çelişen etiketli tüm satırları çıkarılır, kalan tekrarlar tekilleştirilir.
- 6.000 olumlu + 6.000 olumsuz: 8.400 eğitim, 1.200 doğrulama, 2.400 test. Tohum 42.
- Model sonuçları görülmeden seçilen 24 test yorumu, Jev ve diğer dil modellerinin ortak pilotudur. Eğitim kümesinde bulunmaz. Her API paketinde 6 olumlu + 6 olumsuz örnek vardır; **etiketler isteğe dahil edilmez**.
- İnsan tarafından yeniden etiketlenmiş bir test değildir. Nötr sınıf yoktur; yorumlar karışık duygu içerebilir. Kaynak etiketler hatalı olabilir.
- Ürün kimliği olmadığı için ürün bazında ayrım yapılamaz. Yakın tekrarlar ve hazır modellerin bu açık veriyi eğitimde görmüş olması olası sınırlamalardır.
- Örnekleme ve temizlik dökümü: `artifacts/dataset.json`; CSV dosyaları `data/` altında.

Atıf: Fatih Barmanbay, *Turkish Product Reviews*. Bu projede uzunluk filtresi, tekilleştirme, dengeleme ve eğitim/test ayrımı uygulandı. Kaynak README, [CC-BY-SA-4.0](https://github.com/fthbrmnby/turkish-text-data/blob/master/LICENCE) lisansını belirtir; Hugging Face metadata alanı `unknown` der. Kaynak açıklaması `data/raw/SOURCE_README.md` içinde korunmuştur. Türetilmiş veri için kaynak README'de belirtilen CC-BY-SA-4.0 koşulları ve atıf korunmalıdır; metadata tutarsızlığı yeniden dağıtım öncesinde dikkate alınmalıdır. Bu çalışma yereldir, dışarı yayımlanmamıştır.

## Yerel modeller

Ortak TF-IDF (kelime 1–2 gram, en fazla 60.000 özellik) üzerinde Multinomial Naive Bayes, Logistic Regression ve Linear SVM. TF-IDF yalnızca eğitim verisine fit edilir. Test verisi ayar seçimi için kullanılmaz; doğrulama kümesi gelecekteki ayar seçimi için ayrılmıştır. Varsayılan model ayarları sabittir. Eğitim ve tahmin süreleri, doğruluk, macro-F1 ve hata matrisi kaydedilir.

Bunlar klasik makine öğrenmesi modelleridir, büyük dil modelleri değildir. Jev/dil modelleri etiketli eğitim almadan istemle çalışırken yerel modeller 8.400 etiketli örnekle eğitilir. Rapor bu farkı açık tutar. 2.400 örneklik yerel test sonucu, 24 örneklik Jev sonucuyla doğrudan sıralanmaz; ortak pilot ayrıca gösterilir.

## Çalıştırma (PowerShell)

Bu makinedeki `.venv`, yer kaplamamak için mevcut sistem paketlerini kullanır. Başka makinede bağımsız ortam kurulabilir:

```powershell
cd C:\Users\ASUS\Downloads\JevReviewLab
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\download-data.ps1
.\.venv\Scripts\python.exe lab.py prepare
.\.venv\Scripts\python.exe lab.py benchmark
.\.venv\Scripts\python.exe -m unittest -v
```

## Beş krediyi koruma

[jev-ai.pro Batch](https://jev-ai.pro/batch) her satır için 1 kredi alır; AI ile judge oluşturmak da 1 kredi alır. **12 bin yorumluk CSV'yi Batch ekranında çalıştırma.**

[API dokümanı](https://jev-ai.pro/docs): tek istekte en fazla 64 soru; ücretli token yoksa istek başına 1 kredi. Bu proje 12 yorumu kimlikleriyle aynı `state` içinde ve her yorum için ayrı bir `choice` sorusu olarak paketler. Her soru yalnızca kendi yorumuna bakmasını ister. Bu, sitedeki satır bazlı Batch işlemi değildir. Ortak bağlamın kaliteye etkisi ve modelin bağlam sınırı bu küçük pilotta ayrıca değerlendirilmelidir; çok sorulu isteğin başarılı olacağı önceden varsayılmaz.

- İlk aşama: 1 paket / 12 yorum / beklenen 1 kredi.
- İlk yanıt ve faturalama başlıkları incelendikten sonra ikinci paket: 12 yorum / beklenen 1 kredi.
- **Proje genelinde en fazla 2 ücretli istek denemesi**; diğer 3 kredi yedek.
- Varsayılan komut kuru denemedir. Ücretli çalıştırma için ayrıca `--spend-one-credit` gerekir.
- Otomatik tekrar deneme, arka planda çalıştırma veya judge oluşturma yok.
- Zaman aşımı/hata olsa bile deneme hakkı tüketilmiş sayılır. Aynı payload tekrar gönderilmez. Eşzamanlı gönderim kilidi vardır.
- Ham yanıt ve `X-Jev-*` faturalama başlıkları kaydedilir. Kalan bakiye için sağlayıcının başlığı esas alınır. Bu yerel sınır, web sitesinde veya başka uygulamalarda yapılan harcamaları kontrol edemez.
- `artifacts/credit-ledger/` klasörünü silme: harcama sınırı orada tutulur. Beklenmedik kapanmadan kalan `request.lock` varsa hesap ve yanıt durumu anlaşılmadan silme.

API anahtarı **jev-ai.pro hesabından** olmalıdır; TypeSafe'ın ayrı hizmetindeki anahtar farklıdır. Anahtarı sohbet veya kaynak koduna koyma. PowerShell oturumunda gizli giriş:

```powershell
$jevSecret = Read-Host 'Jev API anahtarı' -AsSecureString
$env:JEV_AI_API_KEY = [System.Net.NetworkCredential]::new('', $jevSecret).Password
# Ücretsiz önizleme:
.\.venv\Scripts\python.exe lab.py jev --batch 1
# Yalnızca gerçek çağrı yapmak istediğinde (beklenen 1 kredi):
.\.venv\Scripts\python.exe lab.py jev --batch 1 --spend-one-credit
Remove-Item Env:JEV_AI_API_KEY
```

İkinci paket otomatik çalışmaz. Gerçek çağrı yapıldığında `artifacts/jev-response-1.json` dosyasındaki bakiye ve ücret kontrol edilmelidir. Proje hazırlanırken hiçbir gerçek Jev çağrısı yapılmadı.

## Diğer dil modelleriyle karşılaştırma

Mevcut erişimin olan bir dil modeline `artifacts/llm-prompt-1.txt` ve `llm-prompt-2.txt` içeriğini ayrı yeni sohbetlerde ver. Aynı sınıflar, aynı yorumlar, etiket bilgisi olmadan kullanılır. Model adını/sürümünü ve çalıştırma tarihini kaydet; yanlış sonuçları elle düzeltme. Metin çıktısını `id,prediction` başlıklı tek CSV'de birleştir. Tüm 24 benzersiz ID ve yalnızca `negative` / `positive` etiketleri zorunludur.

```powershell
.\.venv\Scripts\python.exe lab.py import --model MODEL_SURUMU --file model-results.csv
```

Ücretsiz sohbet hesabında elle yapılan çalıştırmaların API maliyeti ve gecikmesi ölçülmüş sayılmaz. İçe aktarılan sonuçlar için hız/maliyet uydurulmaz. Model sıralaması yalnızca aynı ID'lerde hesaplanır. 24 örnek bir ön deneydir; kesin üstünlük iddiası için yeterli değildir.

## Dosyalar

- `lab.py`: veri hazırlama, yerel kıyas, kontrollü Jev istemcisi, sonuç içe aktarma.
- `report-template.html`: bağımsız etkileşimli rapor şablonu.
- `test_lab.py`: kredi sınırı ve yanıt doğrulama testleri.
- `artifacts/local-results.json`: gerçek yerel ölçümler.
- `artifacts/local-predictions.csv`: test örneği bazında yerel model tahminleri.
- `artifacts/jev-request-1.json`, `jev-request-2.json`: etiket sızdırmayan hazır API istekleri.
- `artifacts/pilot-inputs.csv`: yalnızca ID ve yorum içeren ortak pilot girdisi.

Kaynaklar 25 Eylül 2026 tarihinde kontrol edildi. Hesaptaki güncel limit ve fiyatlar değişebilir.


## Profesyonel arayüz

- Genel bakış: gerçek metrik kartları, başarı grafiği ve sınıf dağılımı.
- Model karşılaştırma: test/pilot seçimi, doğruluğa göre sıralama ve hata matrisi.
- Veri seti: tüm 12.000 yorumda arama, veri bölümü/duygu/hata filtreleri, sayfalama, yorum detayı ve filtrelenmiş CSV indirme.
- Jev deneyi: iki hazır istek paketini ve diğer modellere verilecek istemleri indirme. Arayüzden API çağrısı yapılmaz.
- Yöntem ve kaynaklar: veri temizliği, deney sınırları ve manifest indirme.
- `ui/styles.css` ve `ui/app.js` arayüz kaynaklarıdır. Değişiklikten sonra `.\.venv\Scripts\python.exe lab.py report` ile `index.html` yeniden üretilir.
- Uygulama dosya olarak çevrimdışı açılır. `start.ps1` yalnızca index.html dosyasını varsayılan tarayıcıda açar, sunucu başlatmaz ve kredi harcamaz.
