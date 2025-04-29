import streamlit as st
import joblib
import pandas as pd
import numpy as np
import torch


# Page configuration
st.set_page_config(
    page_title="Istanbul Rental Price Prediction | AI-Powered",
    page_icon="🏠",
    layout="centered"
)

# Load models
try:
    # GPU check
    if torch.cuda.is_available():
        # Load TabPFN model if GPU is available
        # tabpfn_loaded = joblib.load("models/tabpfn_model.pkl")
        pass
    
    # Always load CatBoost model
    catboost_loaded = joblib.load("models/catboost_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Initialize session state for language and page
if "language" not in st.session_state:
    st.session_state.language = "tr"
if "page" not in st.session_state:
    st.session_state.page = "predict"

# Initialize default values
if "county" not in st.session_state:
    st.session_state.county = "Kadıköy"
if "net_area" not in st.session_state:
    st.session_state.net_area = "85"
if "gross_area" not in st.session_state:
    st.session_state.gross_area = "95"
if "room_count" not in st.session_state:
    st.session_state.room_count = 2
if "living_room_count" not in st.session_state:
    st.session_state.living_room_count = 1
if "bathroom_count" not in st.session_state:
    st.session_state.bathroom_count = 1
if "is_studio" not in st.session_state:
    st.session_state.is_studio = False

# Language options
def get_translation(text_en, text_tr):
    return text_tr if st.session_state.language == "tr" else text_en

def switch_language():
    st.session_state.language = "en" if st.session_state.language == "tr" else "tr"

# Sidebar
with st.sidebar:
    # Language and GitHub buttons at the top
    col1, col2 = st.columns(2)
    with col1:
        button_text = "EN" if st.session_state.language == "tr" else "TR"
        if st.button("🌐 " + button_text, use_container_width=True):
            switch_language()
            st.rerun()
    
    with col2:
        st.markdown("""
            <a href="https://github.com/suzunn/Rental-Price-Prediction-in-Istanbul" target="_blank" class="github-button">
                <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub">
            </a>
        """, unsafe_allow_html=True)
    
    # Add image
    st.image("assets/houses_3.jpg", use_container_width=True, output_format="JPEG", clamp=True)
    
    # Navigation
    st.markdown("""
        <h1 style='font-size: 32px; margin-bottom: 0px;'>
            {}
        </h1>
        <p style='font-size: 16px; margin-top: 5px;'>
            {}
        </p>
    """.format(
        get_translation("Istanbul Housing Rentals", "İstanbul Konut Kiraları"),
        get_translation("Zingat Data Based Prediction Project", "Zingat Verisi Bazlı Tahmin Projesi")
    ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button(get_translation("🏠 Price Prediction", "🏠 Fiyat Tahmini"), use_container_width=True):
        st.session_state.page = "predict"
        st.rerun()
    
    if st.button(get_translation("ℹ️ About Project", "ℹ️ Proje Hakkında"), use_container_width=True):
        st.session_state.page = "about"
        st.rerun()
    
    st.markdown("---")

# Custom CSS
st.markdown("""
    <style>
    .github-button {
        background: none;
        border: none;
        padding: 0;
        margin: 0;
        cursor: pointer;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s;
    }
    .github-button:hover {
        background: rgba(255,255,255,0.08);
    }
    .github-button img {
        width: 28px;
        height: 28px;
        filter: invert(1) grayscale(1) brightness(1.5);
        border-radius: 50%;
    }
    [data-testid="stSidebarNav"] {
        background-image: none;
    }
    [data-testid="stSidebarNav"]::before {
        content: none;
    }
    .predict-button {
        background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%);
        color: white !important;
        font-weight: bold !important;
        padding: 0.75rem !important;
        font-size: 1.2rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.2);
        transition: all 0.3s ease !important;
    }
    .predict-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 43, 0.3);
    }
    .predict-button:active {
        transform: translateY(0px);
    }
    .predict-button:disabled {
        background: linear-gradient(90deg, #cccccc 0%, #999999 100%) !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)

def ensemble_predict_house_price(county, net_area, gross_area, room_living_str, bathroom_count, is_studio, models, feature_columns):
    # Create a DataFrame with input features
    df = pd.DataFrame([{
        "County": county,
        "Net Area (m²)": net_area,
        "Gross Area (m²)": gross_area,
        "Room-Living Room Count": room_living_str,
        "Bathroom Count": bathroom_count,
        "Is_Studio": int(is_studio)
    }])
    
    # Feature engineering
    df['Room Count'] = df['Room-Living Room Count'].str.extract(r'^(\d+)').astype(int)
    df['Area_ratio'] = df['Net Area (m²)'] / df['Gross Area (m²)']
    df['NetArea_per_Room'] = df['Net Area (m²)'] / df['Room Count']
    df['Area_Loss'] = df['Gross Area (m²)'] - df['Net Area (m²)']
    df['Room_Count_Extracted'] = df['Room-Living Room Count'].str.extract(r'^(\d+)').astype(float).astype('Int64')
    df['Living_Room_Count'] = df['Room-Living Room Count'].str.extract(r'\+(\d+)').astype(float).astype('Int64')
    df['Total_Rooms'] = df['Room_Count_Extracted'].fillna(0) + df['Living_Room_Count'].fillna(0)

    # One-hot encoding for county
    for col in [c for c in feature_columns if c.startswith("County_")]:
        df[col] = 0
    county_col = f"County_{county}"
    if county_col in df.columns:
        df[county_col] = 1
    
    # Reorder columns to match model features
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Make prediction using only CatBoost model
    prediction = np.expm1(catboost_loaded.predict(df)[0])
    lower_bound = np.expm1(catboost_loaded.predict(df)[0] - 0.2)
    upper_bound = np.expm1(catboost_loaded.predict(df)[0] + 0.2)

    return prediction, lower_bound, upper_bound

# Main content based on selected page
if st.session_state.page == "predict":
    st.title(get_translation('Real Estate Price Prediction', 'Gayrimenkul Fiyat Tahmini'))
    
    # Add first image
    st.image("assets/houses_2.jpg", caption=get_translation(
        "Colorful houses and historic streets of Istanbul", 
        "İstanbul'un renkli evleri ve tarihi sokakları"
    ))
    
    # User inputs
    st.markdown("### " + get_translation('Property Information', 'Emlak Bilgileri'))
    
    # First row - District and area information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(get_translation('District', 'İlçe'))
        st.caption(get_translation(
            "Select the district where the property is located in Istanbul. This affects the price estimation.",
            "Evin bulunduğu İstanbul ilçesini seçin. Bu seçim fiyat tahminini önemli ölçüde etkiler."
        ))
        county = st.selectbox("", options=[
            "Adalar", "Arnavutköy", "Ataşehir", "Avcılar", "Bağcılar", "Bahçelievler", "Bakırköy", "Başakşehir",
            "Bayrampaşa", "Beşiktaş", "Beykoz", "Beylikdüzü", "Beyoğlu", "Büyükçekmece", "Çekmeköy", "Esenler",
            "Esenyurt", "Eyüpsultan", "Fatih", "Gaziosmanpaşa", "Kadıköy", "Kartal", "Küçükçekmece", "Maltepe",
            "Pendik", "Sancaktepe", "Silivri", "Sultanbeyli", "Sultangazi", "Şile", "Şişli", "Tuzla", "Zeytinburnu"
        ], key="county")

    with col2:
        st.markdown(get_translation('Net Area (m²)', 'Net Alan (m²)'))
        st.caption(get_translation(
            "Enter the usable living space excluding walls and common areas. This is your actual living area.",
            "Duvarlar ve ortak alanlar hariç kullanılabilir yaşam alanını girin. Gerçek yaşam alanınız."
        ))
        net_area_str = st.text_input("", key="net_area")
        try:
            net_area = int(net_area_str)
            if net_area < 1:
                st.error(get_translation("Net area must be at least 1.", "Net alan en az 1 olmalıdır."))
        except ValueError:
            st.error(get_translation("Please enter a valid number.", "Lütfen geçerli bir sayı girin."))
            net_area = 1

    with col3:
        st.markdown(get_translation('Gross Area (m²)', 'Brüt Alan (m²)'))
        st.caption(get_translation(
            "Enter the total area including walls and common areas. This is the area shown in property deed.",
            "Duvarlar ve ortak alanlar dahil toplam alanı girin. Tapuda gösterilen resmi alanınız."
        ))
        gross_area_str = st.text_input("", key="gross_area")
        try:
            gross_area = int(gross_area_str)
            if gross_area < 1:
                st.error(get_translation("Gross area must be at least 1.", "Brüt alan en az 1 olmalıdır."))
        except ValueError:
            st.error(get_translation("Please enter a valid number.", "Lütfen geçerli bir sayı girin."))
            gross_area = 1

    # Second row - Room, living room, bathroom
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown(get_translation('Room', 'Oda'))
        st.caption(get_translation("Room count.", "Oda sayısı."))
        room_count = st.number_input("", min_value=0, key="room_count")

    with col5:
        st.markdown(get_translation('Living Room', 'Salon'))
        st.caption(get_translation("Living room count.", "Salon sayısı."))
        living_room_count = st.number_input("", min_value=0, key="living_room_count")

    with col6:
        st.markdown(get_translation('Bathroom', 'Banyo'))
        st.caption(get_translation("Number of bathrooms.", "Banyo sayısı."))
        bathroom_count = st.number_input("", min_value=0, key="bathroom_count")

    # Studio apartment checkbox
    is_studio = st.checkbox(get_translation("Studio Apartment", "Stüdyo Daire"), key="is_studio")

    # Error checks
    has_error = False
    if net_area > gross_area:
        st.error(get_translation('Net area cannot be greater than gross area!', 'Net alan, brüt alandan büyük olamaz!'))
        has_error = True
    if room_count == 0 and living_room_count == 0:
        st.error(get_translation('At least one room or living room is required.', 'En az bir oda veya salon olmalı.'))
        has_error = True
    if is_studio and (room_count != 1 or living_room_count != 0):
        st.error(get_translation('For studio, room should be 1 and living room 0.', 'Stüdyo daire için oda sayısı 1 ve salon sayısı 0 olmalı.'))
        has_error = True

    # Create room living string
    room_living_str = f"{int(room_count)}+{int(living_room_count)}"

    # Feature columns for model prediction
    feature_columns = ['Net_Area_(m²)', 'Gross_Area_(m²)', 'Room_Count', 'Bathroom_Count', 'Area_ratio', 'NetArea_per_Room', 'Area_Loss', 
                       'Is_Studio', 'Living_Room_Count', 'Total_Rooms', 'County_Arnavutköy', 'County_Ataşehir', 'County_Avcılar', 
                       'County_Bahçelievler', 'County_Bakırköy', 'County_Bağcılar', 'County_Başakşehir', 'County_Beykoz', 'County_Beylikdüzü', 
                       'County_Beyoğlu', 'County_Beşiktaş', 'County_Büyükçekmece', 'County_Esenler', 'County_Esenyurt', 'County_Eyüpsultan', 
                       'County_Fatih', 'County_Gaziosmanpaşa', 'County_Güngören', 'County_Kadıköy', 'County_Kartal', 'County_Kağıthane', 
                       'County_Küçükçekmece', 'County_Maltepe', 'County_Pendik', 'County_Sancaktepe', 'County_Sarıyer', 'County_Sultanbeyli', 
                       'County_Sultangazi', 'County_Tuzla', 'County_Zeytinburnu', 'County_Çekmeköy', 'County_Ümraniye', 'County_Üsküdar', 
                       'County_Şişli']

    # Predict button
    predict_button = st.button(
        get_translation('🎯 Predict Price', '🎯 Fiyatı Tahmin Et'),
        disabled=has_error,
        key="predict_button",
        use_container_width=True
    )
    
    if predict_button:
        with st.spinner(get_translation('Calculating price prediction...', 'Fiyat tahmini hesaplanıyor...')):
            final_prediction, final_lower, final_upper = ensemble_predict_house_price(
                county=county,
                net_area=net_area,
                gross_area=gross_area,
                room_living_str=room_living_str,
                bathroom_count=bathroom_count,
                is_studio=is_studio,
                models=[],
                feature_columns=feature_columns
            )
            
            # Price prediction results - CSS styling
            st.markdown("""
                <style>
                .price-box { padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center; }
                .main-price { background-color: #FF4B2B; color: white; }
                .price-range-container { margin-top: 10px; }
                .price-range-title { 
                    color: #666; 
                    font-size: 16px; 
                    margin-bottom: 5px; 
                    text-align: left;
                    padding-left: 5px;
                }
                .price-range { 
                    background-color: white; 
                    border: 2px solid #FF4B2B; 
                    display: flex; 
                    justify-content: space-between; 
                    align-items: center; 
                    padding: 15px 30px;
                    color: #333;
                }
                .price-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
                .price-label { font-size: 16px; opacity: 0.9; }
                .price-arrow { 
                    font-size: 24px; 
                    color: #FF4B2B; 
                    font-weight: bold;
                    margin: 0 20px;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Main prediction display
            st.markdown(
                f'<div class="price-box main-price">'
                f'<div class="price-label">{get_translation("Predicted Price", "Tahmin Edilen Fiyat")}</div>'
                f'<div class="price-value">{final_prediction:,.0f} TL</div>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            # Price range display
            st.markdown(
                f'<div class="price-range-container">'
                f'<div class="price-range-title">{get_translation("Price Range", "Fiyat Aralığı")}</div>'
                f'<div class="price-box price-range">'
                f'<div>'
                f'<div class="price-label">{get_translation("Minimum", "Minimum")}</div>'
                f'<div class="price-value">{final_lower:,.0f} TL</div>'
                f'</div>'
                f'<div class="price-arrow">→</div>'
                f'<div>'
                f'<div class="price-label">{get_translation("Maximum", "Maksimum")}</div>'
                f'<div class="price-value">{final_upper:,.0f} TL</div>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

else:  # About page
    st.title(get_translation("About the Project", "Proje Hakkında"))
    
    # Add second image
    st.image("assets/houses.jpg", caption=get_translation(
        "Istanbul cityscape with Galata Tower", 
        "Galata Kulesi ile İstanbul silüeti"
    ))
    
    st.markdown(get_translation("""
    ## 📊 Istanbul Housing Rentals: Prediction Model Based on Zingat Data
    
    This project aims to collect rental price data of apartments in Istanbul from the **Zingat.com** website using web scraping 
    and develop regression models to predict rental prices based on the collected data.
    
    ### Project Steps:
    
    1. **Data Collection (Web Scraping)**
       * Collecting rental apartment listings from Zingat.com
       * Data includes district, price, net area, and rooms information
    
    2. **Data Analysis and Modeling**
       * Exploratory data analysis (EDA)
       * Feature engineering and data cleaning
       * Development of regression models
    
    3. **Model Training**
       * Using TabPFN and CatBoost algorithms
       * Model evaluation and optimization
    
    4. **Prediction Interface**
       * User-friendly web interface
       * Real-time price predictions
       * Multi-language support
    
    ### Technologies Used:
    * Python
    * Streamlit
    * CatBoost
    * TabPFN
    * Pandas & NumPy
    * Scikit-learn
    """,
    """
    ## 📊 İstanbul Konut Kiraları: Zingat Verisi Bazlı Tahmin Modeli
    
    Bu proje, **Zingat.com** sitesinden İstanbul'daki konut kiraları verisini web scraping yöntemiyle toplayarak, 
    bu verileri kullanarak kiraları tahmin etmek için regresyon modelleri geliştirmeyi amaçlamaktadır.
    
    ### Proje Adımları:
    
    1. **Veri Toplama (Web Scraping)**
       * Zingat.com'dan kiralık daire ilanlarının toplanması
       * İlçe, fiyat, net alan ve oda bilgilerini içeren veri seti
    
    2. **Veri Analizi ve Modellenme**
       * Keşifsel veri analizi (EDA)
       * Özellik mühendisliği ve veri temizleme
       * Regresyon modellerinin geliştirilmesi
    
    3. **Model Eğitimi**
       * TabPFN ve CatBoost algoritmalarının kullanımı
       * Model değerlendirme ve optimizasyon
    
    4. **Tahmin Arayüzü**
       * Kullanıcı dostu web arayüzü
       * Gerçek zamanlı fiyat tahminleri
       * Çoklu dil desteği
    
    ### Kullanılan Teknolojiler:
    * Python
    * Streamlit
    * CatBoost
    * TabPFN
    * Pandas & NumPy
    * Scikit-learn
    """))