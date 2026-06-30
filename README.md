# Smart Agriculture - AI 🌱

[![Streamlit App](https://static.streamlit.io/badge-github-white.svg)](https://share.streamlit.io/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered smart agriculture decision engine designed to recommend optimal crops and predict crop yields using machine learning. The system analyzes soil nutrients and climate parameters, calculates crop financial forecasts, and provides live local weather integration via a sleek Streamlit web interface.

---

## 🚀 Key Features

* **Crop Recommendation (Classifier):** Uses an *Extra Trees Classifier* (96.21% test accuracy) to recommend the most compatible crop based on soil chemistry.
* **Yield Prediction (Regressor):** Employs a *Random Forest Regressor* (97.22% $R^2$ score) to predict crop yield in Metric Tons per Hectare (MT/ha).
* **Live Weather Integration:** Fetches real-time temperature, humidity, and rainfall metrics for Indian cities using the **OpenWeatherMap API**.
* **Harvest Economics:** Estimates market price and revenue per hectare based on current crop pricing data.
* **Agronomic Advice:** Provides actionable soil amendment suggestion tips (e.g., how to treat low Nitrogen, high acidity, or high alkalinity).
* **Interactive UI:** A highly polished web dashboard built with Streamlit featuring responsive CSS styles, nutrient ratio progress bars, and custom pH scale gauges.

---

## 📁 File Structure

```
Smart_Agriculture/
├── crop_dataset.csv                # Unified crop dataset with NPK, pH, climate and yield
├── Smart_Agriculture.ipynb         # Model training & exploratory data analysis (EDA)
├── app.py                          # Streamlit web app source code
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
└── *.pkl                           # Serialized models, scalers, and encoders
```

---

## 🛠️ Installation & Setup (Local)

To run the application locally on your computer:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pranav-2910/Smart-Agriculture---AI.git
   cd Smart-Agriculture---AI
   ```

2. **Install the dependencies:**
   It is recommended to run this inside a virtual environment or using `uv`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Machine Learning Model Details

1. **Classifier Engine (Extra Trees Classifier):**
   * **Purpose:** Predicts crop suitability.
   * **Evaluation:** Accuracy of **96.21%** on the validation set.
   * **Saved Artifacts:** `crop_recommendation_model.pkl`, `scaler.pkl`, `label_encoder.pkl`.

2. **Regressor Engine (Random Forest Regressor):**
   * **Purpose:** Predicts numerical crop yields (MT/ha).
   * **Evaluation:** $R^2$ Score of **97.22%**.
   * **Saved Artifacts:** `crop_yield_model.pkl`, `reg_scaler.pkl`, `feature_cols.pkl`.

---

## 🏷️ License

This project is licensed under the MIT License - see the LICENSE file for details.
