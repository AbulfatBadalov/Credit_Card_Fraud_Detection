# 💳 Kredi Kartı Dolandırıcılığı Tespiti Uygulaması

Bu proje, kredi kartı işlemlerinin dolandırıcılık içerip içermediğini tahmin etmek amacıyla bir makine öğrenmesi modeli geliştirmekte ve bu modeli kullanıcı dostu bir arayüzle sunmaktadır. Uygulama, **Streamlit** ile geliştirilmiş görsel bir arayüz üzerinden kullanıcıdan veri alır ve önceden eğitilmiş modeli kullanarak tahminlerde bulunur.

---

## 📌 Proje Özeti

- **Veri Seti:** Kaggle'dan alınan anonimleştirilmiş kredi kartı işlem verileri (284,807 işlem, %0.17 dolandırıcılık oranı)
- **Model:** Decision Tree Classifier (Karar Ağacı)
- **Arayüz:** Streamlit
- **Özellikler:** V1–V28 (PCA ile dönüştürülmüş), Amount (işlem tutarı)

---

## 📁 Dosya Yapısı

credit-card-fraud-detection/
├── app.py # Streamlit arayüz uygulaması
├── credit_card_model.pkl # Eğitilmiş makine öğrenmesi modeli
├── creditcard.csv # Orijinal veri seti (Kaggle)
├── fraud.py # Model eğitimi ve değerlendirme kodu
└── README.md # Proje tanıtımı (bu dosya)

yaml
Copy
Edit

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji        | Açıklama                              |
|------------------|----------------------------------------|
| Python           | Programlama dili                      |
| pandas, numpy    | Veri işleme ve analiz                 |
| scikit-learn     | Makine öğrenmesi modelleri ve metrikler |
| joblib           | Model kaydetme ve yükleme             |
| Streamlit        | Web arayüzü                           |

---

## 🚀 Kurulum ve Çalıştırma

1. Gerekli paketleri yükleyin:

```bash
pip install streamlit pandas numpy scikit-learn joblib
Proje klasörüne gidin ve uygulamayı çalıştırın:

bash
Copy
Edit
streamlit run app.py
Tarayıcıda açılan sayfada işlem özelliklerini girerek modelin tahminini alın.

🔍 Özellik Girişi
Uygulama aşağıdaki girişleri beklemektedir:

V1 - V28: PCA ile dönüştürülmüş anonimleştirilmiş işlem verileri

Amount: İşlem tutarı (TL veya $)

Tahmin sonucu:

✅ Güvenli İşlem

⚠️ Şüpheli İşlem (Fraud)

Ayrıca, modelin verdiği tahminin olasılığı da yüzde olarak sunulur.

📊 Model Eğitimi (fraud.py)
Model, fraud.py dosyasında eğitilmiştir:

Veri önce dengesizliğe karşı incelenmiş ve gerekirse yeniden örnekleme yapılmıştır.

Özellikler ölçeklendirilmiş ve model train_test_split ile değerlendirilmiştir.

Eğitim sonunda model credit_card_model.pkl olarak kaydedilmiştir.

📈 Önerilen Geliştirmeler
Farklı modellerle (Random Forest, XGBoost) karşılaştırma

Oversampling (SMOTE) veya undersampling uygulamaları

Confusion matrix, ROC-AUC grafiği ve diğer görsel çıktılar

Çoklu işlem tahmini için CSV yükleme desteği

👤 Hazırlayan
Badalov Abulfat
Makine Öğrenmesi ve Veri Bilimi Çalışmaları
