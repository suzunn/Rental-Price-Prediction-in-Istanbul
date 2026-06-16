# Istanbul Housing Rental Price Prediction

[Turkce surum asagida](#istanbul-konut-kirasi-tahmini)

This project predicts Istanbul apartment rental prices from Zingat listing data. It combines scraping notebooks, exploratory analysis, feature engineering, trained regression models, and a Streamlit application that serves rental estimates from the saved model artifacts.

## Live App

[Streamlit Web Application](https://rental-price-prediction-in-istanbul.streamlit.app)

The app provides:

- Rental price estimates for Istanbul districts
- Inputs for district, net area, gross area, room count, bathroom count, and studio status
- Turkish and English interface text
- A predicted price range around the model estimate

## Project Workflow

1. **Data collection**
   - [`notebooks/Zingat_Webscrap.ipynb`](notebooks/Zingat_Webscrap.ipynb) collects rental listing data from Zingat.
   - Raw listing data is stored under [`data/raw`](data/raw).

2. **Data cleaning and feature engineering**
   - The notebooks clean missing values, normalize listing fields, and prepare model features.
   - Processed data is stored under [`data/processed`](data/processed).

3. **Modeling**
   - [`notebooks/Zingat_Regression.ipynb`](notebooks/Zingat_Regression.ipynb) trains and evaluates the regression models.
   - Saved model artifacts are stored under [`models`](models).

4. **Prediction UI**
   - [`app.py`](app.py) loads the trained CatBoost model and serves the Streamlit prediction form.

## Repository Layout

```text
.
|-- app.py
|-- assets/
|   |-- houses.jpg
|   |-- houses_2.jpg
|   `-- houses_3.jpg
|-- data/
|   |-- raw/
|   |   `-- zingat_istanbul.csv
|   `-- processed/
|       `-- zingat_istanbul_cleaned.csv
|-- models/
|   |-- catboost_model.pkl
|   `-- tabpfn_model.pkl
|-- notebooks/
|   |-- Zingat_Regression.ipynb
|   `-- Zingat_Webscrap.ipynb
|-- requirements.txt
`-- README.md
```

## Local Setup

Use Python 3.11 or a compatible Python version supported by the pinned dependencies.

```bash
git clone https://github.com/suzunn/Rental-Price-Prediction-in-Istanbul.git
cd Rental-Price-Prediction-in-Istanbul
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Model Artifacts

The Streamlit app expects the trained CatBoost model at:

```text
models/catboost_model.pkl
```

The notebooks can be used to regenerate the processed dataset and model artifacts when the source data or feature engineering changes.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

# Istanbul Konut Kirasi Tahmini

Bu proje, Zingat ilan verilerini kullanarak Istanbul'daki konut kiralarini tahmin eder. Veri toplama not defterleri, kesifsel analiz, ozellik muhendisligi, egitilmis regresyon modelleri ve kaydedilmis model dosyalarini kullanan bir Streamlit uygulamasi icerir.

## Canli Uygulama

[Streamlit Web Uygulamasi](https://rental-price-prediction-in-istanbul.streamlit.app)

Uygulama sunlari saglar:

- Istanbul ilceleri icin kira fiyati tahmini
- Ilce, net alan, brut alan, oda sayisi, banyo sayisi ve studio durumu girdileri
- Turkce ve Ingilizce arayuz metinleri
- Model tahmini etrafinda fiyat araligi

## Proje Akisi

1. **Veri toplama**
   - [`notebooks/Zingat_Webscrap.ipynb`](notebooks/Zingat_Webscrap.ipynb) Zingat uzerinden kiralik ilan verilerini toplar.
   - Ham ilan verisi [`data/raw`](data/raw) altinda tutulur.

2. **Veri temizleme ve ozellik muhendisligi**
   - Not defterleri eksik degerleri temizler, ilan alanlarini normalize eder ve model girdilerini hazirlar.
   - Islenmis veri [`data/processed`](data/processed) altinda tutulur.

3. **Modelleme**
   - [`notebooks/Zingat_Regression.ipynb`](notebooks/Zingat_Regression.ipynb) regresyon modellerini egitir ve degerlendirir.
   - Kaydedilen model dosyalari [`models`](models) altinda tutulur.

4. **Tahmin arayuzu**
   - [`app.py`](app.py) egitilmis CatBoost modelini yukler ve Streamlit tahmin formunu calistirir.

## Yerel Kurulum

Python 3.11 veya sabitlenmis bagimliliklarla uyumlu bir Python surumu kullanin.

```bash
git clone https://github.com/suzunn/Rental-Price-Prediction-in-Istanbul.git
cd Rental-Price-Prediction-in-Istanbul
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS veya Linux icin sanal ortami su komutla etkinlestirin:

```bash
source .venv/bin/activate
```

## Model Dosyalari

Streamlit uygulamasi egitilmis CatBoost modelini su konumda bekler:

```text
models/catboost_model.pkl
```

Kaynak veri veya ozellik muhendisligi degistiginde, not defterleri islenmis veri setini ve model dosyalarini yeniden uretmek icin kullanilabilir.

## Lisans

Bu proje MIT lisansi ile lisanslanmistir. Ayrintilar icin [`LICENSE`](LICENSE) dosyasina bakin.
