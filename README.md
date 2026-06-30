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

To keep the models lightweight for seamless deployment on Streamlit Cloud and GitHub (under 100MB), the tree depths were constrained during training:

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
