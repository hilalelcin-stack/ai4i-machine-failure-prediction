# AI4I 2020: Makine Arıza Tahmini

**Hilal Elçin Şahin** · 2511317001 · MAKÜ Veri Bilimi ve Analitiği
Makine Öğrenmesi dersi **vize** case çalışması
Veri: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10.000 kayıt)

**Rapor:** [`Rapor_HilalElcinSahin.pdf`](Rapor_HilalElcinSahin.pdf) — üç sayfa, her soru bir sayfa. Cevaplar `1a`, `1b` … `3c` etiketleriyle işaretli; soru metinleri tekrar yazılmadı.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `Rapor_HilalElcinSahin.pdf` | 3 sayfalık rapor |
| `ai4i_ariza_tahmini.py` | Bütün analiz tek dosyada. Bölüm başlıkları rapordaki etiketlerle aynı (`SORU 1a` … `SORU 3c`) |
| `ai4i2020.csv` | UCI'den indirilen ham veri (değiştirilmedi) |
| `requirements.txt` | Kütüphane sürümleri |
| `AI4I_Ariza_Tahmini_HilalElcinSahin.zip` | Yukarıdakilerin hepsi tek pakette (mail eki) |

## Çalıştırma

```bash
pip install -r requirements.txt
python ai4i_ariza_tahmini.py
```

Betik yaklaşık 1 dakikada çalışır ve `outputs/` klasörünü üretir: rapordaki bütün tablolar (`.csv`), grafikler (`.png`), `sonuclar.json`, işlenmiş eğitim/test verisi ve tam çalışma logu. Bütün `random_state` değerleri 42 olduğu için skorlar birebir aynı çıkar.

## Doğrulama düzeni

- Veri %80 geliştirme, %20 test olarak bölündü (stratified, seed 42). Test kümesi hiçbir seçimde kullanılmadı, yalnızca sonuç raporlamak için ölçüldü.
- Temizlik, özellik, hiperparametre ve eşik seçimlerinin hepsi geliştirme kümesinde 5 katlı stratified CV ile yapıldı.
- Satır atan temizlik yalnızca eğitim katlarına uygulandı; doğrulama katları ve test eksiksiz kaldı.
- Ölçekleme ve veriden öğrenilen dönüşümler `Pipeline` içinde olduğu için her katta yeniden fit edildi.
- Hiperparametre seçiminin skoru şişirip şişirmediği iç içe (nested) CV ile kontrol edildi.
- Karar eşiği out-of-fold tahminlerden, maliyet kuralıyla seçildi (varsayım: 1 kaçırılan arıza = 10 yanlış alarm).
- Test sonuçları bootstrap %95 güven aralıklarıyla raporlandı.
- Case kısıtına uyularak TWF, HDF, PWF, OSF, RNF, UDI ve Product ID model girdisine alınmadı.

## Özet sonuç

| Model | Test PR-AUC | Recall (eşik 0,50) | CV PR-AUC | Nested CV |
|---|---|---|---|---|
| Logistic Regression (C=1) | 0,5007 | 0,2353 | 0,530 ± 0,059 | 0,526 |
| KNN (k=25) | 0,6067 | 0,1618 | 0,683 ± 0,046 | 0,686 |
| **Decision Tree (depth 7, leaf 10)** | **0,8482** | **0,7206** | **0,883 ± 0,033** | **0,891** |

Seçilen modelde önerilen eşik 0,105'tir. Bu eşikte test kümesindeki 68 arızanın 57'si yakalanır (recall 0,838) ve 17 yanlış alarm çıkar (precision 0,770).
