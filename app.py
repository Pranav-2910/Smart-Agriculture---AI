import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests


# 1. Setting up page configuration...
st.set_page_config(
    page_title="Smart Crop Recommender (Regressor-Based Engine)",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Rendering custom styles...
st.markdown("""
<style>
    /* Importing modern Google Font... */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Styling title banner... */
    .title-banner {
        background: linear-gradient(135deg, #10B981 0%, #059669 50%, #047857 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .title-banner h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: white !important;
    }
    .title-banner p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        color: white !important;
    }
    
    /* Styling metric cards... */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: #1E293B !important;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Styling prediction header success box... */
    .prediction-header {
        background: #ECFDF5;
        border-left: 8px solid #10B981;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        color: #065F46 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Loading serialized models, scalers, and column lists...
@st.cache_resource
def load_artifacts():
    """Loading serialized artifacts from disk..."""
    with open("crop_yield_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("reg_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

@st.cache_data
def load_data_and_profiles():
    """Loading unified dataset and generating crop averages..."""
    df_m = pd.read_csv("crop_dataset.csv")
    
    # Extracting unique lists...
    states = sorted(df_m["state"].unique().tolist())
    seasons = sorted(df_m["season"].unique().tolist())
    crops = sorted(df_m["crop"].unique().tolist())
    
    # Generating crop yields and ideal profiles...
    crop_yields = df_m.groupby("crop")["yield"].mean().to_dict()
    crop_ideals = df_m.groupby("crop")[["N", "P", "K", "pH", "total_rainfall_mm"]].mean().to_dict(orient="index")
    
    return states, seasons, crops, crop_yields, crop_ideals, df_m

# 4. Defining feature engineering function...
def engineer_features(df):
    """Engineering interaction variables..."""
    df_feat = df.copy()
    df_feat['total_nutrients'] = df_feat['N'] + df_feat['P'] + df_feat['K']
    eps = 1e-5
    df_feat['N_P_ratio'] = df_feat['N'] / (df_feat['P'] + eps)
    df_feat['K_P_ratio'] = df_feat['K'] / (df_feat['P'] + eps)
    df_feat['temp_humidity_index'] = df_feat['avg_temp_c'] * df_feat['avg_humidity_percent'] / 100.0
    df_feat['rain_ph_interaction'] = df_feat['total_rainfall_mm'] * df_feat['pH']
    return df_feat

# 5. Configuring live weather fetch...
def get_weather(city):
    """Fetching current weather parameters from OpenWeatherMap..."""
    API_KEY = "239e4e1f1cb6d21585d4d1b424afe60d"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            rain_info = data.get('rain', {})
            rainfall = rain_info.get('1h', rain_info.get('3h', 0.0))
            return float(temp), float(humidity), float(rainfall)
    except:
        pass
    return 25.6, 71.5, 0.0 # Fallback averages...

def search_cities(query):
    """Searching matching cities using OpenWeatherMap Geocoding API..."""
    if not query or len(query.strip()) < 3:
        return []
    API_KEY = "239e4e1f1cb6d21585d4d1b424afe60d"
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={query}&limit=5&appid={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data:
                name = item.get("name", "")
                state = item.get("state", "")
                country = item.get("country", "")
                # Formatting display text...
                display = f"{name}"
                if state:
                    display += f", {state}"
                display += f" ({country})"
                results.append((display, name))
            return results
    except:
        pass
    return []

# Defining pricing profile (adjusting Coconut price to 15 INR per nut)...

CROP_PRICES = {
    "coconut": 15, "sugarcane": 3500, "banana": 16000, "tapioca": 12000,
    "potato": 15000, "onion": 18000, "sweet potato": 14000, "jute": 32000,
    "ginger": 50000, "mesta": 25000, "garlic": 60000, "maize": 18500,
    "turmeric": 70000, "cashewnut": 120000, "bajra": 16000, "rice": 22000,
    "tobacco": 80000, "dry chillies": 95000, "arecanut": 150000, "wheat": 20125,
    "oilseeds total": 45000, "cotton(lint)": 55000, "other oilseeds": 42000,
    "barley": 17000, "peas & beans (pulses)": 50000, "groundnut": 52000,
    "sannhamp": 15000, "ragi": 20000, "soyabean": 38000, "jowar": 18000,
    "arhar/tur": 58000, "guar seed": 35000, "sunflower": 40000, "gram": 48000,
    "other summer pulses": 50000, "other cereals": 16000, "black pepper": 350000,
    "cowpea(lobia)": 45000, "rapeseed &mustard": 50000, "khesari": 30000,
    "other  rabi pulses": 48000, "small millets": 22000, "masoor": 55000,
    "other kharif pulses": 45000, "castor seed": 40000, "coriander": 65000,
    "sesamum": 80000, "urad": 62000, "safflower": 35000, "moong(green gram)": 65000,
    "linseed": 45000, "horse-gram": 30000, "moth": 55000, "niger seed": 35000,
    "cardamom": 800000
}

# Defining main state-to-cities mapping dictionary...
MAIN_STATE_CITIES = {
    "Andhra Pradesh": ["Nandyal", "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati"],
    "Arunachal Pradesh": ["Itanagar", "Naharlagun"],
    "Assam": ["Guwahati", "Silchar", "Dibrugarh"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Bilaspur"],
    "Delhi": ["New Delhi"],
    "Goa": ["Panaji", "Margao"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat"],
    "Himachal Pradesh": ["Shimla", "Dharamshala"],
    "Jammu and Kashmir": ["Srinagar", "Jammu"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli"],
    "Kerala": ["Kochi", "Trivandrum", "Kozhikode"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Manipur": ["Imphal"],
    "Meghalaya": ["Shillong"],
    "Mizoram": ["Aizawl"],
    "Nagaland": ["Kohima", "Dimapur"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela"],
    "Puducherry": ["Puducherry", "Karaikal"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Sikkim": ["Gangtok"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad"],
    "Tripura": ["Agartala"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Ghaziabad", "Agra", "Varanasi", "Meerut", "Allahabad"],
    "Uttarakhand": ["Dehradun", "Haridwar"],
    "West Bengal": ["Kolkata", "Howrah", "Darjeeling", "Siliguri"]
}

# Defining other state-to-cities mapping dictionary (sorted alphabetically)...
OTHER_STATE_CITIES = {
    "Andhra Pradesh": ["Adoni", "Anantapur", "Chittoor", "Eluru", "Hindupur", "Kadapa", "Kakinada", "Kurnool", "Machilipatnam", "Madanapalle", "Narasaraopet", "Nellore", "Ongole", "Proddatur", "Rajamahendravaram", "Srikakulam", "Tenali", "Vizianagaram"],
    "Arunachal Pradesh": ["Along", "Bomdila", "Changlang", "Khonsa", "Namsai", "Pasighat", "Roing", "Seppa", "Tawang", "Tezu", "Ziro"],
    "Assam": ["Barpeta", "Bongaigaon", "Dhubri", "Diphu", "Goalpara", "Golaghat", "Hailakandi", "Jorhat", "Karimganj", "Lakhimpur", "Lumina", "Nagaon", "Nalbari", "Sivasagar", "Tezpur", "Tinsukia"],
    "Bihar": ["Arrah", "Begusarai", "Bettiah", "Bihar Sharif", "Buxar", "Darbhanga", "Gopalganj", "Hajipur", "Jehanabad", "Katihar", "Madhubani", "Motihari", "Munger", "Muzaffarpur", "Purnia", "Saharsa", "Samastipur", "Sasaram", "Siwan"],
    "Chhattisgarh": ["Ambikapur", "Champa", "Dhamtari", "Durg", "Jagdalpur", "Korba", "Mahasamund", "Raigarh", "Rajnandgaon", "Sunabeda"],
    "Delhi": ["Delhi", "Dwarka", "Rohini", "Narela", "Najafgarh"],
    "Goa": ["Bicholim", "Canacona", "Curchorem", "Mapusa", "Mormugao", "Ponda", "Quepem", "Sanguem", "Valpoi", "Vasco da Gama"],
    "Gujarat": ["Anand", "Ankleshwar", "Bharuch", "Bhavnagar", "Bhuj", "Gandhidham", "Gandhinagar", "Godhra", "Jamnagar", "Junagadh", "Mehsana", "Morbi", "Nadiad", "Navsari", "Patan", "Porbandar", "Surendranagar", "Valsad", "Vapi", "Veraval"],
    "Haryana": ["Ambala", "Bahadurgarh", "Bhiwani", "Hisar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Panchkula", "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"],
    "Himachal Pradesh": ["Baddi", "Bilaspur", "Chamba", "Hamirpur", "Kullu", "Mandi", "Nahan", "Palampur", "Solan", "Una"],
    "Jammu and Kashmir": ["Anantnag", "Baramulla", "Kathua", "Poonch", "Pulwama", "Sopore", "Udhampur"],
    "Jharkhand": ["Adityapur", "Bokaro", "Chaibasa", "Deoghar", "Dumka", "Giridih", "Hazaribagh", "Medininagar", "Phusro", "Ramgarh", "Sahibganj"],
    "Karnataka": ["Bagalkot", "Belgaum", "Bellary", "Bidar", "Bijapur", "Chikmagalur", "Chitradurga", "Davangere", "Gadag", "Gokak", "Gulbarga", "Hassan", "Hospet", "Karwar", "Kolar", "Koppal", "Mangalore", "Raichur", "Ranebennur", "Shimoga", "Tumkur", "Udupi"],
    "Kerala": ["Alappuzha", "Changanassery", "Kannur", "Kasaragod", "Kayamkulam", "Kollam", "Kottayam", "Malappuram", "Manjeri", "Nedumangad", "Neyyattinkara", "Palakkad", "Payyannur", "Ponnani", "Quilandy", "Taliparamba", "Thalassery", "Thrissur", "Tirur", "Vadakara"],
    "Madhya Pradesh": ["Betul", "Burhanpur", "Chhindwara", "Dewas", "Dhar", "Guna", "Hoshangabad", "Itarsi", "Khandwa", "Khargone", "Mandsaur", "Murwara", "Neemuch", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", "Shivpuri", "Singrauli", "Ujjain", "Vidisha"],
    "Maharashtra": ["Ahmednagar", "Akola", "Amravati", "Badlapur", "Baramati", "Bhandara", "Bhiwandi", "Bhusawal", "Chandrapur", "Dhule", "Gondia", "Ichalkaranji", "Jalgaon", "Jalna", "Kalyan-Dombivli", "Kolhapur", "Latur", "Malegaon", "Mira-Bhayandar", "Nanded", "Nandurbar", "Navi Mumbai", "Ozar", "Panvel", "Parbhani", "Sangli", "Satara", "Solapur", "Thane", "Ulhasnagar", "Wardha", "Yavatmal"],
    "Manipur": ["Chandel", "Churachandpur", "Kakching", "Lilong", "Mayang Imphal", "Senapati", "Thoubal", "Ukhrul"],
    "Meghalaya": ["Cherrapunji", "Jowai", "Nongpoh", "Nongstoin", "Resubelpara", "Tura", "Williamnagar"],
    "Mizoram": ["Champhai", "Kolasib", "Lawngtlai", "Lunglei", "Mamit", "Saiha", "Serchhip"],
    "Nagaland": ["Kiphire", "Mokokchung", "Mon", "Phek", "Tuensang", "Wokha", "Zunheboto"],
    "Odisha": ["Balasore", "Baripada", "Bhadrak", "Bhawanipatna", "Brahmapur", "Jeypore", "Jharsuguda", "Paradip", "Puri", "Sambalpur"],
    "Puducherry": ["Mahe", "Yanam", "Ozhukarai"],
    "Punjab": ["Abohar", "Barnala", "Bathinda", "Firozpur", "Hoshiarpur", "Khanna", "Malerkotla", "Moga", "Mohali", "Pathankot", "Phagwara", "Rupnagar", "Sangrur", "Sri Muktsar Sahib"],
    "Rajasthan": ["Ajmer", "Alwar", "Barmer", "Bharatpur", "Bhilwara", "Bikaner", "Chittorgarh", "Churu", "Ganganagar", "Hanumangarh", "Jaisalmer", "Jalor", "Jhalawar", "Jhunjhunu", "Kishangarh", "Pali", "Sikar", "Tonk"],
    "Sikkim": ["Geyzing", "Mangan", "Namchi", "Naya Bazar", "Ravangla", "Singtam"],
    "Tamil Nadu": ["Ambattur", "Avadi", "Dindigul", "Erode", "Hosur", "Kancheepuram", "Karaikudi", "Karur", "Kumbakonam", "Nagercoil", "Neyveli", "Pallavaram", "Pudukkottai", "Rajapalayam", "Sivakasi", "Thanjavur", "Thoothukudi", "Tirunelveli", "Tiruppur", "Tiruvannamalai", "Vellore"],
    "Telangana": ["Adilabad", "Jagtial", "Karimnagar", "Khammam", "Mahbubnagar", "Mancherial", "Miryalaguda", "Nalgonda", "Ramagundam", "Suryapet"],
    "Tripura": ["Ambassa", "Belonia", "Dharmanagar", "Kailasahar", "Khowai", "Ranirbazar", "Sabroom", "Udaipur"],
    "Uttar Pradesh": ["Aligarh", "Amroha", "Bareilly", "Bulandshahr", "Etawah", "Faizabad", "Firozabad", "Gorakhpur", "Hapur", "Jhansi", "Loni", "Mathura", "Mirzapur", "Moradabad", "Muzaffarnagar", "Noida", "Orai", "Rampur", "Saharanpur", "Sambhal"],
    "Uttarakhand": ["Haldwani", "Kashipur", "Mussoorie", "Nainital", "Pantnagar", "Pithoragarh", "Rishikesh", "Roorkee", "Rudrapur", "Srinagar"],
    "West Bengal": ["Asansol", "Baharampur", "Bally", "Balurghat", "Baranagar", "Bardhaman", "Bhatpara", "Bidhannagar", "Chinsurah", "Durgapur", "English Bazar", "Haldia", "Jalpaiguri", "Kamarhati", "Kharagpur", "Madhyamgram", "Maheshtala", "Midnapore", "Naihati", "Rajpur Sonarpur", "Shantipur", "Uluberia"]
}




# Setting up dynamic gauges and cards...
def get_ph_indicator(ph):
    if ph < 6.0:
        status = "Acidic 🧪"
        color = "#EF4444"
        pct = (ph - 3.5) / (9.9 - 3.5) * 100
    elif ph <= 7.5:
        status = "Neutral (Optimal) 🟢"
        color = "#10B981"
        pct = (ph - 3.5) / (9.9 - 3.5) * 100
    else:
        status = "Alkaline 🔵"
        color = "#3B82F6"
        pct = (ph - 3.5) / (9.9 - 3.5) * 100
    
    return f"""
    <div style='margin-top:0.5rem; margin-bottom: 1.5rem;'>
        <small style='color:#64748B;'>Soil pH Scale Meter</small>
        <div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#94A3B8; margin-bottom:2px;'>
            <span>Acidic (3.5)</span>
            <span style='color:{color}; font-weight:bold;'>{ph:.1f} - {status}</span>
            <span>Alkaline (9.9)</span>
        </div>
        <div style='width:100%; background-color:#E2E8F0; height:8px; border-radius:4px; position:relative;'>
            <div style='position:absolute; left:{pct}%; width:12px; height:12px; border-radius:50%; background-color:{color}; top:-2px; transform:translateX(-50%); box-shadow:0 0 4px rgba(0,0,0,0.3);'></div>
        </div>
    </div>
    """

def get_npk_breakdown(n, p, k):
    total = n + p + k
    if total == 0:
        return ""
    pct_n = n / total * 100
    pct_p = p / total * 100
    pct_k = k / total * 100
    return f"""
    <div style='margin-top:1rem; margin-bottom: 1.5rem;'>
        <small style='color:#64748B;'>Nutrient Ratio (N : P : K)</small>
        <div style='display:flex; height:20px; border-radius:6px; overflow:hidden; margin-top:5px; box-shadow:inset 0 1px 2px rgba(0,0,0,0.1);'>
            <div style='width:{pct_n}%; background-color:#3B82F6; color:white; font-size:0.75rem; text-align:center; line-height:20px; font-weight:bold;' title='Nitrogen'>N ({pct_n:.0f}%)</div>
            <div style='width:{pct_p}%; background-color:#F59E0B; color:white; font-size:0.75rem; text-align:center; line-height:20px; font-weight:bold;' title='Phosphorus'>P ({pct_p:.0f}%)</div>
            <div style='width:{pct_k}%; background-color:#8B5CF6; color:white; font-size:0.75rem; text-align:center; line-height:20px; font-weight:bold;' title='Potassium'>K ({pct_k:.0f}%)</div>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8; margin-top:3px;'>
            <span>N: {n}</span>
            <span>P: {p}</span>
            <span>K: {k}</span>
        </div>
    </div>
    """

def get_weather_card(temp, humidity, current_rain, seasonal_temp, seasonal_humidity, seasonal_rain, city):
    if seasonal_rain > 1500.0:
        icon = "🌧️"
        condition = "Humid / Monsoon"
        bg_color = "linear-gradient(135deg, #047857 0%, #10B981 100%)"
    elif temp > 30.0:
        icon = "☀️"
        condition = "Hot / Tropical"
        bg_color = "linear-gradient(135deg, #D97706 0%, #F59E0B 100%)"
    elif temp < 18.0:
        icon = "❄️"
        condition = "Cool / Temperate"
        bg_color = "linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)"
    else:
        icon = "☁️"
        condition = "Moderate / Pleasant"
        bg_color = "linear-gradient(135deg, #0D9488 0%, #14B8A6 100%)"
        
    return f"""
    <div style='background:{bg_color}; padding:1.5rem; border-radius:16px; color:white; box-shadow:0 10px 15px -3px rgba(0, 0, 0, 0.1); margin-top:1rem;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <h4 style='margin:0; font-size:1.4rem; font-weight:bold; color:white;'>📍 Weather: {city.upper()}</h4>
                <p style='margin:0.25rem 0 0 0; font-size:0.9rem; opacity:0.9; color:white;'>Condition: {condition}</p>
            </div>
            <span style='font-size:3rem; margin:0;'>{icon}</span>
        </div>
        <div style='display:flex; justify-content:space-between; margin-top:1.5rem; border-top:1px solid rgba(255,255,255,0.2); padding-top:0.75rem;'>
            <div style='text-align:center;'>
                <small style='font-size:0.75rem; opacity:0.8; color:white;'>Temp (Live/Seasonal)</small>
                <p style='margin:0; font-size:1.1rem; font-weight:bold; color:white;'>{temp:.1f} / {seasonal_temp:.1f}°C</p>
            </div>
            <div style='text-align:center;'>
                <small style='font-size:0.75rem; opacity:0.8; color:white;'>Humid (Live/Seasonal)</small>
                <p style='margin:0; font-size:1.1rem; font-weight:bold; color:white;'>{humidity:.0f} / {seasonal_humidity:.0f}%</p>
            </div>
            <div style='text-align:center;'>
                <small style='font-size:0.75rem; opacity:0.8; color:white;'>Rain (Live/Seasonal)</small>
                <p style='margin:0; font-size:1.1rem; font-weight:bold; color:white;'>{current_rain:.1f} / {seasonal_rain:.0f} mm</p>
            </div>
        </div>
    </div>
    """


def get_suitability_check(crop, n, p, k, ph, rainfall, crop_ideals):
    ideals = crop_ideals.get(crop)
    if not ideals:
        return ""
    
    checks = []
    ideal_n = ideals['N']
    if 0.7 * ideal_n <= n <= 1.3 * ideal_n:
        checks.append(f"<li>✅ <strong>Nitrogen (N)</strong>: {n} is optimal (Dynamic Ideal: {ideal_n:.1f})</li>")
    else:
        checks.append(f"<li>⚠️ <strong>Nitrogen (N)</strong>: {n} deviates from optimal {ideal_n:.1f}</li>")
        
    ideal_p = ideals['P']
    if 0.7 * ideal_p <= p <= 1.3 * ideal_p:
        checks.append(f"<li>✅ <strong>Phosphorus (P)</strong>: {p} is optimal (Dynamic Ideal: {ideal_p:.1f})</li>")
    else:
        checks.append(f"<li>⚠️ <strong>Phosphorus (P)</strong>: {p} deviates from optimal {ideal_p:.1f}</li>")
        
    ideal_k = ideals['K']
    if 0.7 * ideal_k <= k <= 1.3 * ideal_k:
        checks.append(f"<li>✅ <strong>Potassium (K)</strong>: {k} is optimal (Dynamic Ideal: {ideal_k:.1f})</li>")
    else:
        checks.append(f"<li>⚠️ <strong>Potassium (K)</strong>: {k} deviates from optimal {ideal_k:.1f}</li>")
        
    ideal_ph = ideals['pH']
    if abs(ph - ideal_ph) <= 0.8:
        checks.append(f"<li>✅ <strong>pH Level</strong>: {ph:.1f} is optimal (Dynamic Ideal: {ideal_ph:.1f})</li>")
    else:
        checks.append(f"<li>⚠️ <strong>pH Level</strong>: {ph:.1f} deviates from optimal {ideal_ph:.1f}</li>")
        
    ideal_rain = ideals['total_rainfall_mm']
    if 0.6 * ideal_rain <= rainfall <= 1.4 * ideal_rain:
        checks.append(f"<li>✅ <strong>Seasonal Rainfall</strong>: {rainfall:.0f}mm is optimal (Dynamic Ideal: {ideal_rain:.0f}mm)</li>")
    else:
        checks.append(f"<li>⚠️ <strong>Seasonal Rainfall</strong>: {rainfall:.0f}mm deviates from dynamic optimal {ideal_rain:.0f}mm</li>")
        
    return f"""
    <div style='background-color:#F8FAFC; border: 1px solid #E2E8F0; padding:1.2rem; border-radius:12px; margin-top:1rem; color:#1E293B !important;'>
        <h4 style='margin:0 0 0.5rem 0; color:#047857;'>🌱 Soil & Climate Suitability Checklist for {crop.title()}:</h4>
        <ul style='margin:0; padding-left:1.2rem; line-height:1.6; color:#1E293B;'>
            {"".join(checks)}
        </ul>
    </div>
    """

def get_amendment_tips(n, p, k, ph, crop, crop_ideals):
    ideals = crop_ideals.get(crop)
    if not ideals:
        return ""
    
    tips = []
    if n < 0.7 * ideals['N']:
        tips.append("<li>🧪 <strong>Low Nitrogen (N)</strong>: Plant leguminous cover crops or apply urea/compost.</li>")
    if p < 0.7 * ideals['P']:
        tips.append("<li>🧪 <strong>Low Phosphorus (P)</strong>: Blend superphosphate or bone meal into the soil.</li>")
    if k < 0.7 * ideals['K']:
        tips.append("<li>🧪 <strong>Low Potassium (K)</strong>: Scatter wood ash or apply muriate of potash.</li>")
    if ph < ideals['pH'] - 0.8:
        tips.append("<li>🧪 <strong>High Acidity</strong>: Add ground limestone to raise pH level.</li>")
    elif ph > ideals['pH'] + 0.8:
        tips.append("<li>🧪 <strong>High Alkalinity</strong>: Spread sulfur or peat moss to lower pH level.</li>")
        
    if not tips:
        tips.append("<li>✅ Your soil indicators are perfectly balanced for this crop profile!</li>")
        
    return f"""
    <div style='background-color:#FFFBEB; border: 1px solid #FEF3C7; padding:1.2rem; border-radius:12px; margin-top:1rem; color:#92400E !important;'>
        <h4 style='margin:0 0 0.5rem 0; color:#92400E;'>🛠️ Soil Amendment Suggestions:</h4>
        <ul style='margin:0; padding-left:1.2rem; line-height:1.6; color:#92400E;'>
            {"".join(tips)}
        </ul>
    </div>
    """

def get_fertilizer_calculator(n, p, k, recommended, area, crop_ideals):
    ideals = crop_ideals.get(recommended, {})
    opt_n = ideals.get("N", 60)
    opt_p = ideals.get("P", 40)
    opt_k = ideals.get("K", 40)
    
    def_n = max(0.0, opt_n - n)
    def_p = max(0.0, opt_p - p)
    def_k = max(0.0, opt_k - k)
    
    # Total requirements in kg
    urea = def_n * 2.17 * area
    ssp = def_p * 6.25 * area
    mop = def_k * 1.67 * area
    
    # Cost calculation: Urea = INR 6/kg, SSP = INR 12/kg, MOP = INR 30/kg (subsidized Indian rates)
    cost = (urea * 6.0) + (ssp * 12.0) + (mop * 30.0)
    
    # Render a premium CSS card
    html = f"""
    <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-top: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); color: #1E293B !important;'>
        <h4 style='margin: 0 0 0.75rem 0; color: #047857; font-weight: 600;'>⚖️ Fertilizer Requirement Calculator</h4>
        <p style='color: #64748B; font-size: 0.85rem; margin-top: 0; margin-bottom: 1rem;'>Estimated total inputs required for your planting area of <strong>{area:.1f} Hectares</strong>:</p>
        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem;'>
            <div style='background: #EFF6FF; padding: 0.75rem; border-radius: 8px; border: 1px solid #BFDBFE; text-align: center;'>
                <small style='color: #2563EB; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;'>Urea (N)</small>
                <h3 style='margin: 0.25rem 0; color: #1E3A8A; font-weight: 700; font-size: 1.2rem;'>{urea:.1f} kg</h3>
                <span style='color: #60A5FA; font-size: 0.65rem;'>Deficit: {def_n:.0f} kg/ha</span>
            </div>
            <div style='background: #FFF7ED; padding: 0.75rem; border-radius: 8px; border: 1px solid #FFEDD5; text-align: center;'>
                <small style='color: #EA580C; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;'>SSP (P)</small>
                <h3 style='margin: 0.25rem 0; color: #7C2D12; font-weight: 700; font-size: 1.2rem;'>{ssp:.1f} kg</h3>
                <span style='color: #FB923C; font-size: 0.65rem;'>Deficit: {def_p:.0f} kg/ha</span>
            </div>
            <div style='background: #ECFDF5; padding: 0.75rem; border-radius: 8px; border: 1px solid #D1FAE5; text-align: center;'>
                <small style='color: #059669; font-weight: 600; text-transform: uppercase; font-size: 0.7rem;'>MOP (K)</small>
                <h3 style='margin: 0.25rem 0; color: #064E3B; font-weight: 700; font-size: 1.2rem;'>{mop:.1f} kg</h3>
                <span style='color: #34D399; font-size: 0.65rem;'>Deficit: {def_k:.0f} kg/ha</span>
            </div>
        </div>
        <div style='margin-top: 0.75rem; display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748B;'>
            <span>Estimated Fertilizer Cost:</span>
            <strong style='color: #1E293B;'>INR {cost:,.2f}</strong>
        </div>
        <p style='color: #94A3B8; font-size: 0.7rem; margin: 0.5rem 0 0 0; text-align: center;'>* Standard conversions: Urea 46% N, SSP 16% P, MOP 60% K.</p>
    </div>
    """
    return html, cost

def get_soil_health_score(n, p, k, ph, crop, crop_ideals):
    ideals = crop_ideals.get(crop, {})
    opt_n = ideals.get("N", 60)
    opt_p = ideals.get("P", 40)
    opt_k = ideals.get("K", 40)
    opt_ph = ideals.get("pH", 6.5)
    
    score_n = max(0.0, min(100.0, 100.0 - abs(n - opt_n) / opt_n * 100.0))
    score_p = max(0.0, min(100.0, 100.0 - abs(p - opt_p) / opt_p * 100.0))
    score_k = max(0.0, min(100.0, 100.0 - abs(k - opt_k) / opt_k * 100.0))
    score_ph = max(0.0, min(100.0, 100.0 - abs(ph - opt_ph) / opt_ph * 100.0))
    
    overall_score = (score_n + score_p + score_k + score_ph) / 4.0
    
    if overall_score >= 85:
        status = "Excellent Soil Balance"
        color = "#059669"
        bg = "#ECFDF5"
    elif overall_score >= 70:
        status = "Good Balance (Minor Amendments Needed)"
        color = "#D97706"
        bg = "#FFFBEB"
    elif overall_score >= 50:
        status = "Moderate Balance (Requires Fertilizer Inputs)"
        color = "#EA580C"
        bg = "#FFF7ED"
    else:
        status = "Poor Balance (Critical Amendments Needed)"
        color = "#DC2626"
        bg = "#FEF2F2"
        
    return f"""
    <div style='background: {bg}; border: 1px solid {color}40; padding: 1rem; border-radius: 12px; display: flex; align-items: center; gap: 1rem; margin-top: 1rem; color: #1E293B !important;'>
        <div style='background: {color}; color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.25rem;'>
            {overall_score:.0f}%
        </div>
        <div>
            <small style='color: #64748B; font-weight: 600; text-transform: uppercase; font-size: 0.75rem;'>Soil Health Compatibility Score</small>
            <h4 style='margin: 0.15rem 0 0 0; color: {color}; font-weight: 700; font-size: 1rem;'>{status}</h4>
        </div>
    </div>
    """

def get_crop_calendar(recommended, season):
    if season == "Kharif":
        timeline = "Sow: 🌧️ June - July | Harvest: 🍂 October - November"
        duration = "120 - 150 Days"
    elif season == "Rabi":
        timeline = "Sow: ❄️ October - November | Harvest: ☀️ March - April"
        duration = "110 - 130 Days"
    elif season == "Summer":
        timeline = "Sow: ☀️ February - March | Harvest: ⛈️ May - June"
        duration = "90 - 110 Days"
    else: # Whole Year / Perennial
        timeline = "Sow: 🌱 Year-round | Harvest: 🔄 Cycle-based"
        duration = "Perennial / Multi-harvest"
        
    return f"""
    <div style='background: #F1F5F9; border: 1px solid #CBD5E1; padding: 1rem; border-radius: 12px; margin-top: 1rem; color: #1E293B !important;'>
        <h4 style='margin: 0 0 0.5rem 0; color: #475569; font-weight: 600;'>📅 Crop Sowing & Harvesting Calendar</h4>
        <div style='display: flex; justify-content: space-between; font-size: 0.85rem;'>
            <div><strong>Sowing Timeline:</strong><br/><span style='color: #475569;'>{timeline}</span></div>
            <div style='text-align: right;'><strong>Crop Duration:</strong><br/><span style='color: #475569;'>{duration}</span></div>
        </div>
    </div>
    """

# 6. Loading dynamic lists and dataframes...
states, seasons, crops, crop_yields, crop_ideals, df_full = load_data_and_profiles()
cities = sorted(df_full["city"].unique().tolist())

# Sidebar: Soil Input Controls
st.sidebar.markdown("### 🧪 Soil Chemistry Profile")
n = st.sidebar.slider("Nitrogen (N) - mg/kg", 10, 120, 50)
p = st.sidebar.slider("Phosphorus (P) - mg/kg", 5, 100, 40)
k = st.sidebar.slider("Potassium (K) - mg/kg", 5, 100, 30)

st.sidebar.markdown(get_npk_breakdown(n, p, k), unsafe_allow_html=True)

ph = st.sidebar.slider("Soil pH level", 3.5, 9.9, 6.5, step=0.1)
st.sidebar.markdown(get_ph_indicator(ph), unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Location & Crop Settings")
state_input = st.sidebar.selectbox("Select State", states, index=states.index("Andhra Pradesh") if "Andhra Pradesh" in states else 0)
season_input = st.sidebar.selectbox("Select Season", seasons, index=seasons.index("Kharif") if "Kharif" in seasons else 0)

# Querying dynamic defaults based on state/season...
state_season_data = df_full[(df_full['state'] == state_input) & (df_full['season'] == season_input)]
default_fert = float(state_season_data['fertilizer'].mean()) if not state_season_data.empty else 80000.0
default_pest = float(state_season_data['pesticide'].mean()) if not state_season_data.empty else 300.0
default_area = float(state_season_data['area'].mean()) if not state_season_data.empty else 1.0
default_rain_avg = float(state_season_data['total_rainfall_mm'].mean()) if not state_season_data.empty else 1000.0
default_temp_avg = float(state_season_data['avg_temp_c'].mean()) if not state_season_data.empty else 25.6
default_hum_avg = float(state_season_data['avg_humidity_percent'].mean()) if not state_season_data.empty else 70.0


planting_area = st.sidebar.number_input("Planting Area (Hectares)", 0.1, 10000000.0, default_area, step=0.5, key=f"area_{state_input}_{season_input}")
fertilizer_input = st.sidebar.number_input("Planned Fertilizer (kg)", 0.0, 100000000.0, default_fert, step=1000.0, key=f"fert_{state_input}_{season_input}")
pesticide_input = st.sidebar.number_input("Planned Pesticide (kg)", 0.0, 10000000.0, default_pest, step=50.0, key=f"pest_{state_input}_{season_input}")

st.sidebar.markdown("### 💧 Water Profile")
rainfall_input = st.sidebar.slider("Expected Seasonal Rainfall (mm)", 100, 3500, int(default_rain_avg), step=50, key=f"rain_slider_{state_input}_{season_input}")

st.sidebar.markdown("### 🔍 Weather API Settings")

# Selecting from main cities of the state...
main_cities = MAIN_STATE_CITIES.get(state_input, ["Hyderabad", "Bangalore", "Mumbai", "New Delhi"])
city_select = st.sidebar.selectbox("Select City", main_cities + ["Other Cities..."], index=0)

if city_select == "Other Cities...":
    # Selecting from other cities in the state alphabetically...
    other_cities = OTHER_STATE_CITIES.get(state_input, [])
    city = st.sidebar.selectbox("Select Other City", other_cities, index=0) if other_cities else main_cities[0]
else:
    city = city_select



# 7. Setting up recommendation click trigger...
btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    predict_button = st.button("Recommend Optimal Crop", use_container_width=True)

if predict_button:
    if not city:
        st.error("Please enter a valid city name.")
    else:
        with st.spinner("Fetching live weather information and performing regression predictions..."):
            model, scaler, feature_cols = load_artifacts()
            
            # Fetching weather parameters...
            temp, humidity, current_rain = get_weather(city)
            
            # Evaluating predicted yields for all candidate crops...
            candidate_rows = []
            for crop in crops:
                row = {
                    'year': 2026,
                    'avg_temp_c': temp,
                    'total_rainfall_mm': rainfall_input,
                    'avg_humidity_percent': humidity,
                    'N': n,
                    'P': p,
                    'K': k,
                    'pH': ph
                }
                # Assigning one-hot dummies...
                for c in crops:
                    row[f'crop_{c}'] = 1.0 if c == crop else 0.0
                for s in states:
                    row[f'state_{s}'] = 1.0 if s == state_input else 0.0
                for se in seasons:
                    row[f'season_{se}'] = 1.0 if se == season_input else 0.0
                for ci in cities:
                    row[f'city_{ci}'] = 1.0 if ci == city else 0.0
                candidate_rows.append(row)
                
            df_candidates = pd.DataFrame(candidate_rows)
            
            # Aligning with feature columns...
            df_candidates = df_candidates.reindex(columns=feature_cols, fill_value=0)
            
            # Engineering feature parameters...
            df_eng = engineer_features(df_candidates)
            
            # Scaling features and maintaining column names...
            df_scaled = pd.DataFrame(scaler.transform(df_eng), columns=df_eng.columns)
            
            # Running inference prediction...
            predicted_yields = model.predict(df_scaled)
            
            # Calculating revenues...
            results = []
            for idx_c, crop in enumerate(crops):
                py = predicted_yields[idx_c]
                price = CROP_PRICES.get(crop.lower().strip(), 30000)
                rev = py * price
                results.append((crop, py, rev, price))
                
            # Recommending the highest profit crop...
            results.sort(key=lambda x: x[2], reverse=True)
            recommended, yield_val, profit, price = results[0]
            
            # Enhanced Economics calculations...
            fert_html, fert_cost = get_fertilizer_calculator(n, p, k, recommended, planting_area, crop_ideals)
            
            CULTIVATION_COSTS = {
                "sugarcane": 25000, "coconut": 35000, "banana": 28000, "cardamom": 45000,
                "black pepper": 40000, "ginger": 30000, "turmeric": 28000, "rice": 18000,
                "wheat": 16000, "maize": 14000, "potato": 22000, "onion": 18000,
            }
            other_cost_rate = CULTIVATION_COSTS.get(recommended.lower().strip(), 12000)
            other_cost_total = other_cost_rate * planting_area
            total_input_cost = fert_cost + other_cost_total
            gross_revenue = yield_val * price * planting_area
            net_profit = gross_revenue - total_input_cost
            
            col_left, col_right = st.columns([3, 2], gap="large")
            
            with col_left:
                st.markdown(f"""
                <div class="prediction-header">
                    <h2 style='color:#065F46; margin:0;'>Optimal Crop recommendation: {recommended.upper()}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Dynamic Soil Health compatibility score
                st.markdown(get_soil_health_score(n, p, k, ph, recommended, crop_ideals), unsafe_allow_html=True)
                
                # Suitability and tips based on dynamic totals
                st.markdown(get_suitability_check(recommended, n, p, k, ph, rainfall_input, crop_ideals), unsafe_allow_html=True)
                st.markdown(get_amendment_tips(n, p, k, ph, recommended, crop_ideals), unsafe_allow_html=True)
                st.markdown(fert_html, unsafe_allow_html=True)
                st.markdown(get_crop_calendar(recommended, season_input), unsafe_allow_html=True)
            
            with col_right:
                # Displaying live metrics alongside seasonal averages in the card...
                st.markdown(get_weather_card(temp, humidity, current_rain, default_temp_avg, default_hum_avg, rainfall_input, city), unsafe_allow_html=True)
                
                st.markdown(f"<h4 style='margin:1.5rem 0 0.5rem 0; color:#047857;'>Cultivation Economics ({planting_area:.1f} Hectares):</h4>", unsafe_allow_html=True)
                
                profit_color = "#10B981" if net_profit >= 0 else "#EF4444"
                profit_text = "Net Profit" if net_profit >= 0 else "Net Loss (Check costs)"
                
                st.markdown(f"""
                <div style='display:flex; flex-direction:column; gap:1rem;'>
                    <div class="metric-card">
                        <small style='color:#64748B;'>Expected Crop Yield</small>
                        <h2 style='margin: 0.25rem 0; color:#047857;'>{yield_val:.2f} MT/ha</h2>
                        <span style='color:#94A3B8; font-size:0.85rem;'>Total Production: {(yield_val * planting_area):.1f} MT</span>
                    </div>
                    <div class="metric-card">
                        <small style='color:#64748B;'>Gross Revenue</small>
                        <h2 style='margin: 0.25rem 0; color:#047857;'>INR {gross_revenue:,.2f}</h2>
                        <span style='color:#94A3B8; font-size:0.85rem;'>Market Rate: INR {price:,}/MT</span>
                    </div>
                    <div class="metric-card">
                        <small style='color:#64748B;'>Total Cultivation Cost</small>
                        <h2 style='margin: 0.25rem 0; color:#475569;'>INR {total_input_cost:,.2f}</h2>
                        <span style='color:#94A3B8; font-size:0.85rem;'>Fertilizers + Seeds/Machinery/Labor</span>
                    </div>
                    <div class="metric-card" style='border-left: 4px solid {profit_color};'>
                        <small style='color:#64748B;'>{profit_text}</small>
                        <h2 style='margin: 0.25rem 0; color:{profit_color};'>INR {net_profit:,.2f}</h2>
                        <span style='color:#94A3B8; font-size:0.85rem;'>Return on Investment</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Spanning full width below the columns: Top Alternative Crops
            st.markdown("---")
            st.markdown("<h3 style='color:#065F46; margin-top:1.5rem; margin-bottom:1rem;'>📊 Economic Comparison of Top 5 Alternative Crops</h3>", unsafe_allow_html=True)
            
            col_chart, col_table = st.columns([3, 2], gap="large")
            
            with col_chart:
                df_alt_chart = pd.DataFrame({
                    'Crop': [r[0].capitalize() for r in results[:5]],
                    'Revenue (INR)': [r[2] for r in results[:5]]
                })
                st.bar_chart(data=df_alt_chart, x='Crop', y='Revenue (INR)', color='#059669', height=300)
                
            with col_table:
                html_list = "<div style='display:flex; flex-direction:column; gap:0.5rem;'>"
                for rank, (c, y, r, p) in enumerate(results[:5]):
                    bg = "#F0FDF4" if rank == 0 else "white"
                    border = "#DCFCE7" if rank == 0 else "#E2E8F0"
                    html_list += f"<div style='background: {bg}; border: 1px solid {border}; padding: 0.5rem 0.75rem; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; color: #1E293B !important;'><div><strong style='color:#1E293B;'>{rank+1}. {c.capitalize()}</strong><br/><span style='color:#64748B; font-size:0.75rem;'>Yield: {y:.2f} MT/ha</span></div><div style='text-align: right;'><strong style='color:#059669;'>INR {r:,.0f}</strong><br/><span style='color:#94A3B8; font-size:0.7rem;'>INR {p:,}/MT</span></div></div>"
                html_list += "</div>"
                st.markdown(html_list, unsafe_allow_html=True)