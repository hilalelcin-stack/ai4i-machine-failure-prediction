> 🇬🇧 English summary below · 🇹🇷 [Türkçe açıklama](#ai4i-2020-makine-arıza-tahmini)

# AI4I 2020: Machine Failure Prediction

**Hilal Elçin Şahin** · BSc Data Science and Analytics, Mehmet Akif Ersoy University (MAKÜ)
Individual case study for the Machine Learning course.
Data: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 records, 3.4% failures)

## What this project does

It predicts machine failure from five sensor readings and the product type, compares three classifiers, and picks a decision threshold from an explicit cost rule instead of the default 0.50.

## Results

| Model | Test PR-AUC | Recall (threshold 0.50) | CV PR-AUC | Nested CV |
|---|---|---|---|---|
| Logistic Regression (C=1) | 0.5007 | 0.2353 | 0.530 ± 0.059 | 0.526 |
| KNN (k=25) | 0.6067 | 0.1618 | 0.683 ± 0.046 | 0.686 |
| **Decision Tree (depth 7, leaf 10)** | **0.8482** | **0.7206** | **0.883 ± 0.033** | **0.891** |

For the selected model the recommended threshold is 0.105. At that threshold the model catches 57 of the 68 failures in the test set (recall 0.838) with 17 false alarms (precision 0.770).

## Validation design

- 80% development / 20% test split (stratified, seed 42). The test set was used only to report final results, never for any choice.
- Cleaning, feature, hyperparameter and threshold choices were all made on the development set with 5-fold stratified cross-validation.
- Row-dropping cleaning was applied to training folds only; validation folds and the test set stayed complete.
- Scaling and other learned transformations sit inside a `Pipeline`, so they are refitted in every fold.
- Nested cross-validation checks whether hyperparameter selection inflated the score.
- The decision threshold was chosen from out-of-fold predictions under a cost rule (assumption: 1 missed failure = 10 false alarms).
- Test results are reported with bootstrap 95% confidence intervals.
- Following the case constraints, TWF, HDF, PWF, OSF, RNF, UDI and Product ID were not used as model inputs.

## How to run

```bash
pip install -r requirements.txt
python ai4i_ariza_tahmini.py
```

The script runs in about one minute and regenerates the whole `outputs/` folder. Every `random_state` is 42, so the scores reproduce exactly.

## Repository guide

| File | Content |
|---|---|
| `Rapor_HilalElcinSahin.pdf` | Three-page report (in Turkish) |
| `ai4i_ariza_tahmini.py` | The full analysis in one script |
| `ai4i2020.csv` | Raw data from UCI, unmodified |
| `outputs/grafikler/` | Figures used in the report |
| `outputs/*.csv`, `outputs/sonuclar.json` | Every table and number in the report |
| `outputs/calisma_log.txt` | Full console log of a run |

The report, code comments and file names are in Turkish.

---

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
| `outputs/grafikler/` | Rapordaki grafikler: `soru1.png`, `soru2_karar_agaci.png`, `soru3.png` |
| `outputs/*.csv` | Rapordaki bütün tablolar (karşılaştırmalar, ileri seçim, eşik politikaları, maliyet, derinlik eğrisi, değişken önemi, `skorlar.csv`) |
| `outputs/sonuclar.json` | Rapordaki her sayının kaynağı, tek dosyada |
| `outputs/calisma_log.txt` | Betiğin baştan sona çalışma çıktısı |
| `outputs/islenmis_*.csv` | İşlenmiş eğitim ve test veri setleri |
| `outputs/s2d_agac_kurallari.txt` | Karar ağacının tam kural listesi |

`outputs/` klasörü hazır olarak eklenmiştir; betiği çalıştırmaya gerek kalmadan incelenebilir.

## Çalıştırma

```bash
pip install -r requirements.txt
python ai4i_ariza_tahmini.py
```

Betik yaklaşık 1 dakikada çalışır ve `outputs/` klasörünün tamamını yeniden üretir. Bütün `random_state` değerleri 42 olduğu için skorlar birebir aynı çıkar.

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
