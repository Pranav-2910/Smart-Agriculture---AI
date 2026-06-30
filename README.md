# Smart Agriculture - AI 🌱

Smart Agriculture - AI is an advanced, end-to-end agritech decision engine designed to recommend optimal crops and predict expected crop yields using machine learning. The system helps farmers, agronomists, and agricultural stakeholders make data-driven decisions by analyzing soil nutrient profiles and meteorological parameters, calculating dynamic crop cultivation economics, and integrating live local weather telemetry.

---

## 📊 Core Features & Capabilities

### 1. Dual-Model Machine Learning Engine
The core of the decision engine combines two machine learning models into a single, cohesive inference pipeline:
*   **Localized Crop Recommendation (Classifier):** Uses a depth-constrained **Extra Trees Classifier** trained on our extensive regional database. By incorporating N, P, K, pH, weather averages, and one-hot encoded city features, the model achieves a **92.58% test accuracy**. It predicts the optimal crop to plant based on historical regional success, ensuring recommendations match localized farming patterns (e.g., Visakhapatnam yields Rice/Coconut, whereas dry Tirupati yields Jowar/Cotton).
*   **Yield & Production Forecasting (Regressor):** Employs a depth-constrained **Random Forest Regressor** to predict numerical crop yields in Metric Tons per Hectare (MT/ha), achieving an $R^2$ score of **98.82%**. This model calculates the crop yield based on soil inputs, water availability, and climate indicators.

### 2. High-Fidelity 10-Year Regional Database (`crop_dataset.csv`)
The models are trained on a comprehensive dataset of **79,563 rows** generated from real-world regional crop parameters:
*   **Time Span:** Covers 10 consecutive years of farming data (2016–2025).
*   **Geographic Scope:** Tracks **30 states** and **250+ unique Indian cities** (including major hubs and secondary farming cities) using regional climate and soil centroids.
*   **Variables:** Records crop types (55 crops), year, season (Kharif, Rabi, Summer, Whole Year), area, production, fertilizer usage, pesticide usage, yield (MT/ha), temperature, humidity, rainfall, Nitrogen (N), Phosphorus (P), Potassium (K), and pH.

### 3. Live Weather Telemetry Integration
*   The application integrates directly with the **OpenWeatherMap API** to fetch live temperature, humidity, and rainfall data for the selected city.
*   If the weather API cannot be reached, the system automatically falls back to the specific city's 10-year historical weather averages for the selected season.

### 4. Dynamic Soil Health & Fertilizer Advisor
*   **Soil Health Compatibility Score:** Calculates a dynamic percentage score (0% to 100%) indicating how well the current soil test NPK and pH values match the biological optima of the recommended crop. It displays a color-coded status bar (Excellent, Good, Moderate, Poor Balance).
*   **Fertilizer Requirement Calculator:** Compares current soil test results with optimal crop requirements and calculates the exact deficit. It translates this deficit into the total kilograms of **Urea (N)**, **SSP (P)**, and **MOP (K)** needed for the farmer's specific planting area.

### 5. Crop Cultivation Economics (Net Profit Dashboard)
*   **Gross Revenue:** Computes estimated revenue based on predicted yields, local market prices, and planting area.
*   **Cultivation Input Costs:** Sums fertilizer costs (calculated using current subsidized Urea, SSP, and MOP market rates in India) and standard seed/labor/irrigation rates for the specific crop.
*   **Net Profit / ROI:** Displays a bold, color-coded card highlighting the net profit (green) or potential loss (red) to help the user evaluate crop viability.

### 6. Seasonal Sowing & Harvesting Calendar
*   Provides a visual sowing window and harvesting period based on the crop's season (Kharif, Rabi, Summer, Whole Year) alongside the typical growth duration (in days) to help plan crop rotations.

---

## 🔄 Step-by-Step Implementation Workflow

The project was developed in a structured machine learning pipeline divided into the following sequential steps (documented in `Smart_Agriculture.ipynb` and implemented via our automation scripts):

### Step 1: Regional Profiling & Dataset Generation
1.  **Meteorological Centroids:** Defined baseline temperatures, humidities, and rainfall patterns for 250+ cities across 30 states.
2.  **Crop Selection Logic:** Created deterministic biological crop profiles (N, P, K, pH, temp, humidity, and rainfall envelopes) for 55 distinct crops.
3.  **Soil Synthesis:** Generated soil profiles (N, P, K, pH) centered tightly around each crop's biological optimal requirements to model realistic farm planting environments.
4.  **Yield Math Formulation:** Calculated expected crop yields using a multi-factor Gaussian decay curve representing soil and climatic suitability, scaled by real-world state output multipliers.
5.  **Output Export:** Merged and exported all profiles into a clean, 10-year CSV file (`crop_dataset.csv`) containing 79,563 rows.

### Step 2: Data Preprocessing & Cleaning
1.  **categorical Encoding:** One-hot encoded categorical inputs using `pd.get_dummies()`:
    *   **Classifier:** Encoded `season` and `city` (dropped first column to prevent local collinearity).
    *   **Regressor:** Encoded `crop`, `season`, and `city`.
2.  **Target Coding:** Encoded the classifier target variable (`crop`) using `LabelEncoder()` to map text labels into numerical indices.
3.  **Workspace Housekeeping:** Deleted old, redundant, or unformatted CSV files (`crop_yield.csv`, `state_soil_data.csv`, and `state_weather_data_1997_2020.csv`) to ensure a clean local directory and push.

### Step 3: Feature Engineering
Calculated interaction variables to capture biological and farming input contexts:
1.  `total_nutrients`: Combined sum of Nitrogen, Phosphorus, and Potassium.
2.  `N_P_ratio` & `K_P_ratio`: Chemical balance ratios in the soil.
3.  `temp_humidity_index`: Multiplicative thermal-humidity interaction index.
4.  `rain_ph_interaction`: Water-acidity interaction variable.
5.  **Input Intensity Indicators:** Added `fertilizer_per_area`, `pesticide_per_area`, `fertilizer_pesticide_ratio`, and `rain_fertilizer_interaction` to capture input efficiency relative to planting area.

### Step 4: Model Training & Hyperparameter Tuning
1.  **Dataset Splits:** Split the preprocessed dataset into an 80% training set and a 20% validation set using `train_test_split(random_state=42)`.
2.  **Standardization:** Scaled features using `StandardScaler` to remove scale bias across NPK values (0-150) and pH levels (3.5-9.9).
3.  **Classifier Training (Extra Trees):**
    *   Fit an `ExtraTreesClassifier` to recommend crops.
    *   **Optimization:** Constrained the tree depth to `max_depth=14` and `n_estimators=100`. This shrank the serialized file size by **77% (from 241MB to 54MB)** to comply with GitHub's 100MB file limit, while maintaining a peak test accuracy of **92.58%**.
4.  **Regressor Training (Random Forest):**
    *   Fit a `RandomForestRegressor` to predict yield.
    *   **Optimization:** Constrained the depth to `max_depth=15` and `n_estimators=100` to shrink the model size to **67MB** while preserving a **98.82%** $R^2$ score.

### Step 5: Serialization & Deployment
1.  Exported the trained models, encoders, and feature column structures into serialized pickle binary files (`.pkl`) for rapid loading in the Streamlit app.
2.  Built the `app.py` user dashboard to load the artifacts, fetch current city weather, and run sequential dual-model inference (using the Classifier to recommend the crop, and the Regressor to estimate its harvest economics).

---

## 🛠️ Installation & Setup (Local)

To run the application locally on your computer:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Pranav-2910/Smart-Agriculture---AI.git
    cd Smart-Agriculture---AI
    ```

2.  **Install dependencies:**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the Streamlit dashboard:**
    ```bash
    streamlit run app.py
    ```
    Once started, the app will run locally at `http://localhost:8501`.

---

## 📊 Machine Learning Model Details & Serialization

To keep the models lightweight for seamless deployment on Streamlit Cloud and GitHub, the tree depths were constrained during training:

1.  **Classifier (Extra Trees):**
    *   **Features:** `N`, `P`, `K`, `pH`, `avg_temp_c`, `avg_humidity_percent`, `total_rainfall_mm`, `area`, `fertilizer`, `pesticide`, `season` (one-hot), `city` (one-hot).
    *   **Constraints:** `n_estimators=100`, `max_depth=14`.
    *   **Pickle Size:** **~54.4 MB** (retains **92.58%** test accuracy).
    *   **Serialized Artifacts:** `crop_recommendation_model.pkl`, `scaler.pkl`, `feature_cols_clf.pkl`, `label_encoder.pkl`.

2.  **Regressor (Random Forest):**
    *   **Features:** `N`, `P`, `K`, `pH`, `avg_temp_c`, `avg_humidity_percent`, `total_rainfall_mm`, `area`, `fertilizer`, `pesticide`, `crop` (one-hot), `season` (one-hot), `city` (one-hot).
    *   **Constraints:** `n_estimators=100`, `max_depth=15`.
    *   **Pickle Size:** **~67.1 MB** (retains **98.82%** $R^2$ score).
    *   **Serialized Artifacts:** `crop_yield_model.pkl`, `reg_scaler.pkl`, `feature_cols.pkl`.
