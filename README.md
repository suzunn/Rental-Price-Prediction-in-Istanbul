# 📊 İstanbul Konut Kiraları: Zingat Verisi Bazlı Tahmin Modeli

[English version below](#-istanbul-housing-rentals-prediction-model-based-on-zingat-data)

Bu proje, **Zingat.com** sitesinden İstanbul'daki konut kiraları verisini **web scraping** yöntemiyle toplayarak, bu verileri kullanarak kiraları tahmin etmek için regresyon modelleri geliştirmeyi amaçlamaktadır. Proje, web scraping, veri analizi, özellik mühendisliği ve regresyon modelleri kullanılarak gerçekleştirilmiştir.

## Web Uygulaması

🌐 **[Streamlit Web Uygulaması](rental-price-prediction-in-istanbul.streamlit.app)**

Projemiz, kullanıcı dostu bir web arayüzü ile birlikte gelir. Bu uygulama şunları sunar:

- İstanbul'daki konutlar için gerçek zamanlı kira tahmini
- Kolay kullanılabilir form arayüzü
- Çoklu dil desteği (Türkçe/İngilizce)
- Detaylı fiyat aralığı gösterimi
- İnteraktif kullanıcı deneyimi

## Proje Adımları

1. **Veri Toplama (Web Scraping)**:
   - **[Zingat_Webscrap.ipynb](notebooks/Zingat_Webscrap.ipynb)** not defteri, **Zingat.com** sitesinden kiralık daire ilanlarını toplar.
   - Veri, dairenin **ilçe**, **fiyat**, **net alan**, **odalar** gibi çeşitli özelliklerini içerir.
   
2. **Veri Analizi ve Modellenme**:
   - **[Zingat_Regression.ipynb](notebooks/Zingat_Regression.ipynb)** not defterinde, toplanan veriler üzerinde keşifsel veri analizi (EDA) ve modelleme yapılır.
   - Regresyon modelleri, daire özellikleri kullanılarak kiraların tahmin edilmesini sağlar.

3. **Veri Temizleme ve Özellik Mühendisliği**:
   - Toplanan veriler temizlenir ve özellik mühendisliği uygulamaları yapılır.
   - Eksik değerler ve aykırı değerler düzeltilir.

4. **Model Eğitim ve Tahmin**:
   - **TabPFN** ve **CatBoost** gibi iki regresyon modeli eğitilir.
   - Modellerin doğruluğu değerlendirilir ve tahminler yapılır.

## Dosya Yapısı

```
Rental-Price-Prediction-Project  
├── app.py                                 # Streamlit web uygulaması
├── assets                                 # Uygulama görselleri
│   ├── houses.jpg
│   ├── houses_2.jpg
│   └── houses_3.jpg
├── data  
│   ├── raw  
│   │   ├── zingat_istanbul.csv              # Web sitesinden toplanan ham veri
│   └── processed  
│       ├── zingat_istanbul_cleaned.csv      # Temizlenmiş ve işlenmiş veri  
├── notebooks  
│   ├── Zingat_Webscrap.ipynb                # Web scraping not defteri  
│   └── Zingat_Regression.ipynb              # Veri analizi ve regresyon modelleme not defteri  
├── models  
│   ├── tabpfn_model.pkl                    # Eğitilmiş TabPFN modeli  
│   └── catboost_model.pkl                  # Eğitilmiş CatBoost modeli
│  
├── README.md                              # Proje açıklama dosyası  
└── requirements.txt                       # Proje için gerekli Python paketleri  
```

## Gereksinimler

Projenin çalışabilmesi için gerekli Python paketleri şunlardır:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `catboost`
- `tabpfn`
- `matplotlib`
- `seaborn`
- `requests`
- `beautifulsoup4`
- `joblib`

## Kullanım

1. Repoyu klonlayın:
```bash
git clone https://github.com/suzunn/Rental-Price-Prediction-in-Istanbul.git
```

2. Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

3. Web uygulamasını başlatmak için:
```bash
streamlit run app.py
```

4. Web scraping, veri analizi ve model eğitimi için not defterlerini çalıştırın:
   - **[Zingat_Webscrap.ipynb](notebooks/Zingat_Webscrap.ipynb)**: Veri toplama
   - **[Zingat_Regression.ipynb](notebooks/Zingat_Regression.ipynb)**: Veri analizi ve model eğitimi

5. Modeli kullanarak tahmin yapmak için eğitilen modelleri **models** klasöründen yükleyebilirsiniz:
   - **TabPFN** ve **CatBoost** modellerini `joblib` ile yükleyerek yeni veriler üzerinde tahmin yapabilirsiniz.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakabilirsiniz.

---

# 📊 Istanbul Housing Rentals: Prediction Model Based on Zingat Data

[Türkçe versiyon için yukarı kaydırın](#-i̇stanbul-konut-kiraları-zingat-verisi-bazlı-tahmin-modeli)

This project aims to collect rental price data of apartments in Istanbul from the **Zingat.com** website using web scraping and develop regression models to predict rental prices based on the collected data. The project involves web scraping, data analysis, feature engineering, and regression modeling.

## Web Application

🌐 **[Streamlit Web Application](rental-price-prediction-in-istanbul.streamlit.app)**

Our project comes with a user-friendly web interface that offers:

- Real-time rental price predictions for properties in Istanbul
- Easy-to-use form interface
- Multi-language support (Turkish/English)
- Detailed price range visualization
- Interactive user experience

## Project Steps

1. **Data Collection (Web Scraping)**:
   - **[Zingat_Webscrap.ipynb](notebooks/Zingat_Webscrap.ipynb)** notebook collects rental apartment listings from **Zingat.com**.
   - The data includes various features such as **district**, **price**, **net area**, and **rooms**.
   
2. **Data Analysis and Modeling**:
   - In the **[Zingat_Regression.ipynb](notebooks/Zingat_Regression.ipynb)** notebook, exploratory data analysis (EDA) and modeling are performed on the collected data.
   - Regression models enable the prediction of rents using apartment features.

3. **Data Cleaning and Feature Engineering**:
   - The collected data is cleaned and feature engineering applications are made.
   - Missing values and outliers are corrected.

4. **Model Training and Prediction**:
   - Two regression models are trained: **TabPFN** and **CatBoost**.
   - The accuracy of the models is evaluated and predictions are made.

## File Structure

```
Rental-Price-Prediction-Project  
├── app.py                                 # Streamlit web application
├── assets                                 # Application images
│   ├── houses.jpg
│   ├── houses_2.jpg
│   └── houses_3.jpg
├── data  
│   ├── raw  
│   │   ├── zingat_istanbul.csv              # Raw data scraped from the website  
│   └── processed  
│       ├── zingat_istanbul_cleaned.csv      # Cleaned and processed data  
├── notebooks  
│   ├── Zingat_Webscrap.ipynb                # Web scraping notebook  
│   └── Zingat_Regression.ipynb              # Data analysis and regression modeling notebook  
├── models  
│   ├── tabpfn_model.pkl                    # Trained TabPFN model  
│   └── catboost_model.pkl                  # Trained CatBoost model   
│
├── README.md                              # Project description file  
└── requirements.txt                       # Required Python packages for the project  
```

## Requirements

The required Python packages for the project to work are:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `catboost`
- `tabpfn`
- `matplotlib`
- `seaborn`
- `requests`
- `beautifulsoup4`
- `joblib`

## Usage

1. Clone the repository:
```bash
git clone https://github.com/suzunn/Rental-Price-Prediction-in-Istanbul.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. To start the web application:
```bash
streamlit run app.py
```

4. Run the notebooks for web scraping, data analysis, and model training:
   - **[Zingat_Webscrap.ipynb](notebooks/Zingat_Webscrap.ipynb)**: Data collection
   - **[Zingat_Regression.ipynb](notebooks/Zingat_Regression.ipynb)**: Data analysis and model training

5. To make predictions using the model, you can load the trained models from the **models** folder:
   - You can load the **TabPFN** and **CatBoost** models with `joblib` to make predictions on new data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
