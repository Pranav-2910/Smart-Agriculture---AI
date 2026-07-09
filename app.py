import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os

# 1. Setting up page configuration...
st.set_page_config(
    page_title="Smart Crop Recommender",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rendering custom styles...
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
    
    /* Ensuring high-contrast black text inside custom HTML cards... */
    .custom-html-card, .custom-html-card * {
        color: #000000 !important;
    }
    
    /* Ensuring high-contrast black text inside metric cards... */
    .metric-card, .metric-card * {
        color: #000000 !important;
    }
    
    /* Weather card text must remain white on its dark gradient background... */
    .custom-weather-card, .custom-weather-card * {
        color: #ffffff !important;
    }
    
    /* Sidebar custom cards must have white text for visibility on the dark sidebar background... */
    [data-testid="stSidebar"] .custom-html-card,
    [data-testid="stSidebar"] .custom-html-card * {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Defining multilingual translation dictionary...
TRANSLATIONS = {
    "English": {
        "title": "Smart Crop Recommender",
        "subtitle": "Predicting optimal crop recommendations, expected yields, and cultivation economics based on real-time soil, location, and seasonal weather parameters.",
        "lang_select": "Select Language",
        "settings_location": "📍 Location & Crop Settings",
        "select_state": "Select State",
        "select_city": "Select City",
        "select_other_city": "Select Other City",
        "select_season": "Select Season",
        "settings_soil": "🧪 Soil Chemistry Profile",
        "soil_type": "Soil Type / Texture",
        "soil_type_custom": "Custom (I have lab test values)",
        "n_label": "Nitrogen (N) - mg/kg",
        "p_label": "Phosphorus (P) - mg/kg",
        "k_label": "Potassium (K) - mg/kg",
        "ph_label": "Soil pH level",
        "settings_cultivation": "🚜 Cultivation Parameters",
        "area_label": "Planting Area",
        "fert_label": "Planned Fertilizer",
        "pest_label": "Planned Pesticide",
        "settings_water": "💧 Water Profile",
        "rain_label": "Expected Seasonal Rainfall (mm)",
        "settings_override": "🔧 Weather API Override",
        "override_label": "Override Live Weather",
        "manual_temp": "Current Temperature (°C)",
        "manual_hum": "Current Humidity (%)",
        "manual_rain": "Current Rainfall (mm/hr)",
        "btn_recommend": "Recommend Optimal Crop",
        "loading": "Fetching weather information and performing predictions...",
        "err_city": "Please enter a valid city name.",
        "rec_header": "Optimal Crop recommendation",
        "calendar_header": "📅 Crop Sowing & Harvesting Calendar",
        "sow_timeline": "Sowing Timeline",
        "crop_duration": "Crop Duration",
        "economics_header": "Cultivation Economics",
        "pred_yield": "Predicted Yield",
        "market_price": "Local Market Price",
        "est_revenue": "Estimated Gross Revenue",
        "total_cost": "Estimated Input Cost",
        "net_profit": "Estimated Net Profit",
        "net_loss": "Estimated Net Loss (Check costs)",
        "soil_health_score": "Soil-Crop Compatibility Score",
        "risk_header": "📊 Market Price Volatility & Risk Analysis (±20%)",
        "risk_pessimistic": "Pessimistic Market (-20% Price)",
        "risk_optimistic": "Optimistic Market (+20% Price)",
        "risk_high_alert": "⚠️ HIGH RISK: A 20% drop in market price could result in a net loss of INR {loss:,.2f}.",
        "risk_low_alert": "✅ LOW RISK: Even with a 20% drop in market price, this crop is expected to remain profitable at INR {profit:,.2f}.",
        "alternative_crops": "💡 Alternative Crop Yield Leaderboard",
        "unit_hectares": "Hectares",
        "unit_acres": "Acres",
        "unit_kg": "kg",
        "unit_bags": "Bags (50kg)",
        "units_title": "📏 Measurement Units",
        "alignment_header": "💡 Agronomic Calendar Alignment",
        "alignment_msg": "The machine learning model recommended **Rice** based on your soil inputs. However, please note that in the Kharif season (July), grains/millets like **Maize** or **Jowar** are more commonly sown in this region. Rice is traditionally transplanted in October (Rabi).",
        "comparison_title": "🔍 Soil Nutrient Comparison: Your Field vs. Ideal {crop}",
        "your_soil_label": "Your Soil",
        "ideal_soil_label": "Ideal Soil",
        "advisory_title": "📢 Live Agronomic Advisory Alerts",
        "advisory_pest_risk": "🦟 <strong>Pest & Disease Risk</strong>: High humidity and warm temperatures increase susceptibility to fungal blasts and stem borers.",
        "advisory_water_risk": "🌊 <strong>Waterlogging Risk</strong>: Extreme cumulative rainfall. Ensure proper field drainage channels to prevent root rot.",
        "advisory_acid_risk": "🧪 <strong>High Acid Blockade</strong>: Low pH limits phosphorus availability. Apply agricultural lime to neutralize.",
        "advisory_alk_risk": "🧪 <strong>Alkaline Blockade</strong>: High pH reduces micronutrient absorption. Consider adding organic gypsum.",
        "advisory_optimal": "✅ <strong>Optimal Growth Window</strong>: No major environmental stress alerts for this cycle.",
        "leaderboard_advisory_title": "📊 Alternative Crop Analysis & Farmer Advisory",
        "leaderboard_revenue_opt": "💰 <strong>Maximum Revenue Option</strong>: Cultivating <strong>{crop}</strong> is expected to yield the highest market return of <strong>INR {revenue:,.2f}</strong>.",
        "leaderboard_yield_opt": "🌾 <strong>Maximum Bulk Production</strong>: If your goal is high yield volume, <strong>{crop}</strong> is the strongest performer, with an expected yield of <strong>{yield_val:.2f} {unit}</strong>.",
        "leaderboard_diversification": "🚜 <strong>Risk Diversification</strong>: Farmers are advised to split their land (e.g., 70% primary crop, 30% alternative crop) to hedge against crop failure and market price drops.",
        "help_n": "Nitrogen (N) in soil (recommended: 10-120 mg/kg). Promotes leaf growth and overall crop vigor.",
        "help_p": "Phosphorus (P) in soil (recommended: 5-100 mg/kg). Crucial for root development, seeding, and flower maturity.",
        "help_k": "Potassium (K) in soil (recommended: 5-100 mg/kg). Regulates water absorption and strengthens disease resistance.",
        "help_ph": "Soil pH index measures acidity/alkalinity. 7.0 is neutral; most crops grow best between 6.0 and 7.5.",
        "help_area_acres": "Total planting field area in Acres.",
        "help_area_ha": "Total planting field area in Hectares. 1 Hectare = 2.47 Acres.",
        "help_fert_bags": "Total fertilizer applied in standard 50 kg bags.",
        "help_fert_kg": "Total fertilizer quantity applied in kilograms (kg).",
        "help_pest_bags": "Total pesticide applied in standard 50 kg bags.",
        "help_pest_kg": "Total pesticide quantity applied in kilograms (kg).",
        "help_rain": "Total seasonal cumulative precipitation in millimeters (mm). 1 mm equals 1 Liter/m².",
        "help_manual_temp": "Manual override value for relative ambient temperature in Celsius (°C).",
        "help_manual_hum": "Manual override value for relative humidity percentage (%).",
        "help_manual_rain": "Manual override value for immediate hourly precipitation rate in millimeters (mm)."
    },
    "Hindi": {
        "title": "स्मार्ट फसल सलाहकार",
        "subtitle": "मिट्टी, स्थान और मौसमी मौसम के मापदंडों के आधार पर फसल की सिफारिशों, उपज और खेती के अर्थशास्त्र की भविष्यवाणी करना।",
        "lang_select": "भाषा चुनें",
        "settings_location": "📍 स्थान और फसल सेटिंग्स",
        "select_state": "राज्य चुनें",
        "select_city": "शहर चुनें",
        "select_other_city": "अन्य शहर चुनें",
        "select_season": "मौसम चुनें",
        "settings_soil": "🧪 मिट्टी रसायन प्रोफाइल",
        "soil_type": "मिट्टी का प्रकार",
        "soil_type_custom": "कस्टम (मेरे पास लैब टेस्ट रिपोर्ट है)",
        "n_label": "नाइट्रोजन (N) - मिलीग्राम/किग्रा",
        "p_label": "फास्फोरस (P) - मिलीग्राम/किग्रा",
        "k_label": "पोटेशियम (K) - मिलीग्राम/किग्रा",
        "ph_label": "मिट्टी का पीएच स्तर",
        "settings_cultivation": "🚜 खेती के मापदंड",
        "area_label": "खेती का क्षेत्र",
        "fert_label": "योजनाबद्ध उर्वरक",
        "pest_label": "योजनाबद्ध कीटनाशक",
        "settings_water": "💧 पानी की जानकारी",
        "rain_label": "अपेक्षित मौसमी वर्षा (मिमी)",
        "settings_override": "🔧 मौसम एपीआई ओवरराइड",
        "override_label": "लाइव मौसम ओवरराइड करें",
        "manual_temp": "वर्तमान तापमान (°C)",
        "manual_hum": "वर्तमान आर्द्रता (%)",
        "manual_rain": "वर्तमान वर्षा (मिमी/घंटा)",
        "btn_recommend": "फसल की सिफारिश प्राप्त करें",
        "loading": "मौसम की जानकारी प्राप्त की जा रही है और उपज की भविष्यवाणी की जा रही है...",
        "err_city": "कृपया एक वैध शहर का नाम दर्ज करें।",
        "rec_header": "सर्वोत्तम अनुशंसित फसल",
        "calendar_header": "📅 फसल बुवाई और कटाई कैलेंडर",
        "sow_timeline": "बुवाई की समय सीमा",
        "crop_duration": "फसल की अवधि",
        "economics_header": "खेती का अर्थशास्त्र",
        "pred_yield": "अनुमानित उपज",
        "market_price": "स्थानीय बाजार मूल्य",
        "est_revenue": "अनुमानित कुल राजस्व",
        "total_cost": "अनुमानित इनपुट लागत",
        "net_profit": "अनुमानित शुद्ध लाभ",
        "net_loss": "अनुमानित शुद्ध हानि",
        "soil_health_score": "मिट्टी-फसल अनुकूलता स्कोर",
        "risk_header": "📊 बाजार मूल्य अस्थिरता और जोखिम विश्लेषण (±20%)",
        "risk_pessimistic": "मंदी का बाजार (-20% कीमत)",
        "risk_optimistic": "तेजी का बाजार (+20% कीमत)",
        "risk_high_alert": "⚠️ उच्च जोखिम: बाजार मूल्य में 20% की गिरावट से INR {loss:,.2f} का शुद्ध नुकसान हो सकता है।",
        "risk_low_alert": "✅ कम जोखिम: बाजार मूल्य में 20% की गिरावट के बाद भी, इस फसल से INR {profit:,.2f} का शुद्ध लाभ मिलने की उम्मीद है।",
        "alternative_crops": "💡 वैकल्पिक फसल उपज लीडरबोर्ड",
        "unit_hectares": "हेक्टेयर",
        "unit_acres": "एकड़",
        "unit_kg": "किग्रा",
        "unit_bags": "बोरी (50 किग्रा)",
        "units_title": "📏 मापन इकाइयां",
        "alignment_header": "💡 कृषि कैलेंडर संरेखण",
        "alignment_msg": "मशीन लर्निंग मॉडल ने आपकी मिट्टी के इनपुट के आधार पर **चावल (Rice)** की सिफारिश की है। हालांकि, कृपया ध्यान दें कि खरीफ सीजन (जुलाई) में इस क्षेत्र में **मक्का (Maize)** या **ज्वार (Jowar)** जैसे अनाज/बाजरा अधिक उगाए जाते हैं। चावल की रोपाई पारंपरिक रूप से अक्टूबर (रबी सीजन) में की जाती है।",
        "comparison_title": "🔍 मिट्टी के पोषक तत्वों की तुलना: आपका खेत बनाम आदर्श {crop}",
        "your_soil_label": "आपके खेत की मिट्टी",
        "ideal_soil_label": "आदर्श मिट्टी",
        "advisory_title": "📢 लाइव कृषि सलाहकार चेतावनियां",
        "advisory_pest_risk": "🦟 <strong>कीट और रोग का जोखिम</strong>: उच्च आर्द्रता और गर्म तापमान फंगल ब्लास्ट और तना छेदक कीटों के प्रति संवेदनशीलता बढ़ाते हैं।",
        "advisory_water_risk": "🌊 <strong>जलभराव का जोखिम</strong>: अत्यधिक संचयी वर्षा। जड़ सड़न को रोकने के लिए उचित जल निकासी सुनिश्चित करें।",
        "advisory_acid_risk": "🧪 <strong>अम्लीय अवरोध</strong>: कम पीएच फॉस्फोरस की उपलब्धता को सीमित करता है। उदासीन करने के लिए कृषि चूना डालें।",
        "advisory_alk_risk": "🧪 <strong>क्षारीय अवरोध</strong>: उच्च पीएच सूक्ष्म पोषक तत्वों के अवशोषण को कम करता है। जैविक जिप्सम मिलाने पर विचार करें।",
        "advisory_optimal": "✅ <strong>इष्टतम विकास स्थिति</strong>: इस चक्र के लिए कोई बड़ा पर्यावरणीय तनाव अलर्ट नहीं है।",
        "leaderboard_advisory_title": "📊 वैकल्पिक फसल विश्लेषण और किसान सलाह",
        "leaderboard_revenue_opt": "💰 <strong>अधिकतम आय विकल्प</strong>: <strong>{crop}</strong> की खेती से <strong>INR {revenue:,.2f}</strong> का उच्चतम बाजार रिटर्न मिलने की उम्मीद है।",
        "leaderboard_yield_opt": "🌾 <strong>अधिकतम कुल उत्पादन</strong>: यदि आपका लक्ष्य उच्च उपज मात्रा है, तो <strong>{crop}</strong> सबसे मजबूत प्रदर्शनकर्ता है, जिसकी अपेक्षित उपज <strong>{yield_val:.2f} {unit}</strong> है।",
        "leaderboard_diversification": "🚜 <strong>जोखिम विविधीकरण</strong>: किसानों को सलाह दी जाती है कि वे फसल खराब होने और बाजार मूल्य में गिरावट के जोखिम से बचने के लिए अपनी जमीन को विभाजित करें (जैसे, 70% प्राथमिक फसल, 30% वैकल्पिक फसल)।",
        "help_n": "मिट्टी में नाइट्रोजन (N) (सिफारिश: 10-120 mg/kg)। पत्तियों की वृद्धि और फसल की ताकत को बढ़ावा देता है।",
        "help_p": "मिट्टी में फास्फोरस (P) (सिफारिश: 5-100 mg/kg)। जड़ों के विकास, बीजों के निर्माण और फूलों की परिपक्वता के लिए महत्वपूर्ण है।",
        "help_k": "मिट्टी में पोटेशियम (K) (सिफारिश: 5-100 mg/kg)। पानी के अवशोषण को नियंत्रित करता है और रोग प्रतिरोधक क्षमता को बढ़ाता है।",
        "help_ph": "मिट्टी पीएच स्तर अम्लता/क्षारीयता को मापता है। 7.0 तटस्थ है; अधिकांश फसलें 6.0 और 7.5 के बीच सबसे अच्छी बढ़ती हैं।",
        "help_area_acres": "एकड़ में कुल खेती का क्षेत्र।",
        "help_area_ha": "हेक्टेयर में कुल खेती का क्षेत्र। 1 हेक्टेयर = 2.47 एकड़।",
        "help_fert_bags": "50 किलोग्राम की मानक बोरियों में कुल उपयोग की गई खाद।",
        "help_fert_kg": "किलोग्राम (kg) में कुल उपयोग की गई खाद की मात्रा।",
        "help_pest_bags": "50 किलोग्राम की मानक बोरियों में कुल उपयोग की गई कीटनाशक।",
        "help_pest_kg": "किलोग्राम (kg) में कुल उपयोग की गई कीटनाशक की मात्रा।",
        "help_rain": "मिलीमीटर (mm) में अपेक्षित मौसमी वर्षा। 1 मिमी = 1 लीटर प्रति वर्ग मीटर।",
        "help_manual_temp": "सेल्सियस (°C) में तापमान को मैन्युअल रूप से बदलने के लिए।",
        "help_manual_hum": "सापेक्ष आर्द्रता प्रतिशत (%) को मैन्युअल रूप से बदलने के लिए।",
        "help_manual_rain": "वर्षा दर (मिमी/घंटा) को मैन्युअल रूप से बदलने के लिए।"
    },
    "Telugu": {
        "title": "స్మార్ట్ పంట సలహాదారు",
        "subtitle": "మట్టి, స్థానం మరియు వాతావరణ సమాచారం ఆధారంగా సరైన పంట సిఫార్సులు, దిగుబడులు మరియు వ్యవసాయ ఆదాయాన్ని అంచనా వేయడం.",
        "lang_select": "భాషను ఎంచుకోండి",
        "settings_location": "📍 ప్రాంతం & పంట సెట్టింగ్‌లు",
        "select_state": "రాష్ట్రం ఎంచుకోండి",
        "select_city": "నగరం ఎంచుకోండి",
        "select_other_city": "ఇతర నగరం ఎంచుకోండి",
        "select_season": "పంట కాలం ఎంచుకోండి",
        "settings_soil": "🧪 మట్టి రసాయన ప్రొఫైల్",
        "soil_type": "నేల రకం",
        "soil_type_custom": "కస్టమ్ (నా వద్ద ల్యాబ్ టెస్ట్ రిపోర్ట్ ఉంది)",
        "n_label": "నత్రజని (N) - mg/kg",
        "p_label": "భాస్వరం (P) - mg/kg",
        "k_label": "పొటాషియం (K) - mg/kg",
        "ph_label": "మట్టి pH స్థాయి",
        "settings_cultivation": "🚜 సాగు వివరాలు",
        "area_label": "సాగు వైశాల్యం",
        "fert_label": "ప్రణాళికాబద్ధమైన ఎరువులు",
        "pest_label": "ప్రణాళికాబద్ధమైన పురుగుమందులు",
        "settings_water": "💧 నీటి ప్రొఫైల్",
        "rain_label": "ఆశించిన కాలానుగుణ వర్షపాతం (mm)",
        "settings_override": "🔧 వాతావరణ API ఓవర్‌రైడ్",
        "override_label": "లైవ్ వాతావరణాన్ని ఓవర్‌రైడ్ చేయండి",
        "manual_temp": "ప్రస్తుత ఉష్ణోగ్రత (°C)",
        "manual_hum": "ప్రస్తుత తేమ (%)",
        "manual_rain": "ప్రస్తుత వర్షపాతం (mm/hr)",
        "btn_recommend": "సరైన పంటను సిఫార్సు చేయండి",
        "loading": "వాతావరణ సమాచారాన్ని పొందుతోంది మరియు పంట అంచనాలను లెక్కిస్తోంది...",
        "err_city": "దయచేసి సరైన నగర పేరు నమోదు చేయండి.",
        "rec_header": "అత్యంత అనుకూలమైన పంట",
        "calendar_header": "📅 పంట విత్తే & కోత కాలపట్టిక",
        "sow_timeline": "విత్తే కాలం",
        "crop_duration": "పంట కాలపరిమితి",
        "economics_header": "సాగు ఆర్థిక విశ్లేషణ",
        "pred_yield": "అంచనా దిగుబడి",
        "market_price": "స్థానిక మార్కెట్ ధర",
        "est_revenue": "అంచనా స్థూల ఆదాయం",
        "total_cost": "అంచనా సాగు ఖర్చు",
        "net_profit": "అంచనా నికర లాభం",
        "net_loss": "అంచనా నికర నష్టం",
        "soil_health_score": "నేల-పంట అనుకూలత స్కోరు",
        "risk_header": "📊 మార్కెట్ ధరల హెచ్చుతగ్గులు & రిస్క్ విశ్లేషణ (±20%)",
        "risk_pessimistic": "తక్కువ ధర మార్కెట్ (-20% ధర)",
        "risk_optimistic": "ఎక్కువ ధర మార్కెట్ (+20% ధర)",
        "risk_high_alert": "⚠️ అధిక రిస్క్: మార్కెట్ ధర 20% తగ్గితే నికర నష్టం INR {loss:,.2f} వచ్చే అవకాశం ఉంది.",
        "risk_low_alert": "✅ తక్కువ రిస్క్: మార్కెట్ ధర 20% తగ్గినప్పటికీ, ఈ పంట ద్వారా INR {profit:,.2f} నికర లాభం పొందే అవకాశం ఉంది.",
        "alternative_crops": "💡 ప్రత్యామ్నాయ పంటల దిగుబడి లీడర్‌బోర్డ్",
        "unit_hectares": "హెక్టార్లు",
        "unit_acres": "ఎకరాలు",
        "unit_kg": "కిలోలు",
        "unit_bags": "బస్తాలు (50 కేజీలు)",
        "units_title": "📏 కొలత ప్రమాణాలు",
        "alignment_header": "💡 పంట కాలపట్టిక క్రమబద్ధీకరణ",
        "alignment_msg": "మట్టి వివరాల ఆధారంగా మెషిన్ లెర్నింగ్ మోడల్ **వరి (Rice)** ని సిఫార్సు చేసింది. అయితే, ఖరీఫ్ సీజన్ (జూలై) లో ఈ ప్రాంతంలో **మొక్కజొన్న (Maize)** లేదా **జొన్నలు (Jowar)** వంటి తృణధాన్యాలు ఎక్కువగా పండిస్తారు. వరి సాధారణంగా అక్టోబర్ (రబీ సీజన్) లో నాటబడుతుంది.",
        "comparison_title": "🔍 మట్టి పోషకాల పోలిక: మీ నేల నిష్పత్తి వర్సెస్ ఆదర్శవంతమైన {crop}",
        "your_soil_label": "మీ మట్టి పోషకాలు",
        "ideal_soil_label": "ఆదర్శవంతమైన పోషకాలు",
        "advisory_title": "📢 ప్రత్యక్ష వ్యవసాయ సలహా హెచ్చరికలు",
        "advisory_pest_risk": "🦟 <strong>తెగుళ్ళ ముప్పు</strong>: ఎక్కువ తేమ మరియు వేడి ఉష్ణోగ్రతల వల్ల అగ్గితెగులు మరియు కాండం తొలిచే పురుగుల ఆశించే ముప్పు పెరుగుతుంది.",
        "advisory_water_risk": "🌊 <strong>నీరు నిలిచే ముప్పు</strong>: విపరీతమైన కాలానుగుణ వర్షపాతం. వేరు కుళ్లు తెగులు నివారణకు సరైన నీటి పారుదల కాలువలు ఏర్పాటు చేసుకోండి.",
        "advisory_acid_risk": "🧪 <strong>అధిక ఆమ్లత్వం</strong>: తక్కువ pH వల్ల వేర్లకు భాస్వరం అందదు. ఆమ్లత్వాన్ని తగ్గించడానికి వ్యవసాయ సున్నం ఉపయోగించండి.",
        "advisory_alk_risk": "🧪 <strong>క్షార గుణం</strong>: ఎక్కువ pH వల్ల సూక్ష్మ పోషకాలను నేల గ్రహించలేదు. జిప్సం ఉపయోగించాల్సిందిగా సలహా ఇవ్వడమైనది.",
        "advisory_optimal": "✅ <strong>అనుకూల వాతావరణం</strong>: ఈ పంట కాలానికి ఎటువంటి వాతావరణ పరమైన ముప్పులు లేవు.",
        "leaderboard_advisory_title": "📊 ప్రత్యామ్నాయ పంటల విశ్లేషణ & రైతు సలహా",
        "leaderboard_revenue_opt": "💰 <strong>అధిక ఆదాయ పంట</strong>: <strong>{crop}</strong> సాగు చేయడం వల్ల <strong>INR {revenue:,.2f}</strong> అత్యధిక ఆదాయం వచ్చే అవకాశం ఉంది.",
        "leaderboard_yield_opt": "🌾 <strong>అధిక దిగుబడి పంట</strong>: ఎక్కువ దిగుబడి పొందడమే మీ లక్ష్యమైతే, <strong>{crop}</strong> నాటడం ద్వారా <strong>{yield_val:.2f} {unit}</strong> అత్యుత్తమ దిగుబడి లభిస్తుంది.",
        "leaderboard_diversification": "🚜 <strong>ఆదాయ భద్రత</strong>: పంట నష్టం మరియు మార్కెట్ ధరల పతనం నుండి రక్షణ పొందడానికి రైతులు తమ పొలాన్ని విభజించి సాగు చేయాలి (ఉదాహరణకు, 70% ప్రధాన పంట, 30% ప్రత్యామ్నాయ పంట).",
        "help_n": "నెలలో నత్రజని (N) పరిమాణం (సిఫార్సు: 10-120 mg/kg). ఆకు పచ్చదనానికి మరియు పంట ఎదుగుదలకు తోడ్పడుతుంది.",
        "help_p": "నేలలో భాస్వరం (P) పరిమాణం (సిఫార్సు: 5-100 mg/kg). వేర్లు బలంగా పెరగడానికి మరియు పూతకు చాలా ముఖ్యం.",
        "help_k": "నేలలో పొటాషియం (K) పరిమాణం (సిఫార్సు: 5-100 mg/kg). తెగుళ్లను తట్టుకునే శక్తిని ఇస్తుంది.",
        "help_ph": "నేల pH విలువ ఆమ్ల/క్షార గుణాన్ని సూచిస్తుంది. 7.0 తటస్థ విలువ; పంటలు 6.0 నుండి 7.5 pH మధ్య బాగా పెరుగుతాయి.",
        "help_area_acres": "ఎకరాలలో సాగు వైశాల్యం.",
        "help_area_ha": "హెక్టార్లలో సాగు వైశాల్యం. 1 హెక్టార్ = 2.47 ఎకరాలు.",
        "help_fert_bags": "సాధారణ 50 కేజీల బస్తాల రూపంలో సాగుకు వాడే ఎరువులు.",
        "help_fert_kg": "కిలోగ్రాముల (kg) రూపంలో సాగుకు వాడే ఎరువుల పరిమాణం.",
        "help_pest_bags": "సాధారణ 50 కేజీల బస్తాల రూపంలో సాగుకు వాడే పురుగుమందులు.",
        "help_pest_kg": "కిలోగ్రాముల (kg) రూపంలో సాగుకు వాడే పురుగుమందుల పరిమాణం.",
        "help_rain": "మిల్లీమీటర్లలో (mm) ఆశించిన కాలానుగుణ వర్షపాతం. 1 mm = చదరపు మీటరుకు 1 లీటరు నీరు.",
        "help_manual_temp": "సెల్సియస్ (°C) లో ఉష్ణోగ్రతను మార్చుటకు ఉపయోగపడుతుంది.",
        "help_manual_hum": "తేమ శాతం (%) ని మార్చుటకు ఉపయోగపడుతుంది.",
        "help_manual_rain": "ప్రస్తుత వర్షపాత తీవ్రతను (mm/hr) మార్చుటకు ఉపయోగపడుతుంది."
    },
    "Marathi": {
        "title": "स्मार्ट पीक सल्लागार",
        "subtitle": "माती, हवामान आणि स्थानावर आधारित पीक शिफारसी, उत्पन्न आणि शेतीचे अर्थशास्त्र यांचे अंदाज लावणारे प्रणाली.",
        "lang_select": "भाषा निवडा",
        "settings_location": "📍 स्थान आणि पीक सेटिंग्ज",
        "select_state": "राज्य निवडा",
        "select_city": "शहर निवडा",
        "select_other_city": "इतर शहर निवडा",
        "select_season": "हंगाम निवडा",
        "settings_soil": "🧪 मातीचे रासायनिक प्रोफाइल",
        "soil_type": "मातीचा प्रकार",
        "soil_type_custom": "कस्टम (माझ्याकडे माती परीक्षण अहवाल आहे)",
        "n_label": "नायट्रोजन (N) - मिग्रॅ/किग्रॅ",
        "p_label": "फॉस्फरस (P) - मिग्रॅ/किग्रॅ",
        "k_label": "पोटॅशियम (K) - मिग्रॅ/किग्रॅ",
        "ph_label": "मातीचा पीएच (pH) स्तर",
        "settings_cultivation": "🚜 लागवड तपशील",
        "area_label": "लागवडीचे क्षेत्र",
        "fert_label": "नियोजित खते",
        "pest_label": "नियोजित कीटकनाशके",
        "settings_water": "💧 पाण्याचे प्रमाण",
        "rain_label": "अपेक्षित हंगामी पाऊस (मीमी)",
        "settings_override": "🔧 हवामान एपीआय ओव्हरराईड",
        "override_label": "थेट हवामान ओव्हरराईड करा",
        "manual_temp": "सध्याचे तापमान (°C)",
        "manual_hum": "सध्याची आर्द्रता (%)",
        "manual_rain": "सध्याचा पाऊस (मीमी/तास)",
        "btn_recommend": "उत्कृष्ट पीक शिफारस मिळवा",
        "loading": "हवामानाची माहिती गोळा करत आहे आणि पिकाचा अंदाज लावत आहे...",
        "err_city": "कृपया योग्य शहराचे नाव प्रविष्ट करा.",
        "rec_header": "सर्वोत्तम पीक शिफारस",
        "calendar_header": "📅 पीक पेरणी आणि कापणी वेळापत्रक",
        "sow_timeline": "पेरणीचा कालावधी",
        "crop_duration": "पिकाचा कालावधी",
        "economics_header": "लागवडीचे अर्थशास्त्र",
        "pred_yield": "अपेक्षित उत्पन्न",
        "market_price": "स्थानिक बाजार भाव",
        "est_revenue": "अपेक्षित एकूण उत्पन्न",
        "total_cost": "अपेक्षित लागवड खर्च",
        "net_profit": "अपेक्षित निव्वळ नफा",
        "net_loss": "अपेक्षित निव्वळ तोटा",
        "soil_health_score": "माती-पीक अनुकूलता गुणवत्ता",
        "risk_header": "📊 बाजार भाव चढ-उतार आणि जोखीम विश्लेषण (±२०%)",
        "risk_pessimistic": "मंदीचा बाजार (-२०% बाजार भाव)",
        "risk_optimistic": "तेजीचा बाजार (+२०% बाजार भाव)",
        "risk_high_alert": "⚠️ मोठी जोखीम: बाजार भावात २०% घसरण झाल्यास INR {loss:,.2f} चा निव्वळ तोटा होऊ शकतो.",
        "risk_low_alert": "✅ कमी जोखीम: बाजार भावात २०% घसरण झाली तरीही, या पिकातून INR {profit:,.2f} चा निव्वळ नफा मिळणे अपेक्षित आहे.",
        "alternative_crops": "💡 पर्यायी पीक उत्पन्न लीडरबोर्ड",
        "unit_hectares": "हेक्टर",
        "unit_acres": "एकर",
        "unit_kg": "किग्रॅ",
        "unit_bags": "गोणी (50 किग्रॅ)",
        "units_title": "📏 मोजमाप एकके",
        "alignment_header": "💡 पीक पेरणी वेळापत्रक जुळवणी",
        "alignment_msg": "मशीन लर्निंग मॉडेलने आपल्या मातीच्या घटकांनुसार **भात (Rice)** ची शिफारस केली आहे. तथापि, कृपया नोंद घ्या की खरीप हंगामात (जुलै) या भागात **मका (Maize)** किंवा **ज्वारी (Jowar)** यांसारखे धान्य पिके अधिक प्रमाणात घेतली जातात. भाताची लावणी प्रामुख्याने ऑक्टोबर (रब्बी हंगामात) केली जाते.",
        "comparison_title": "🔍 माती पोषक घटक तुलना: आपले शेत वि. आदर्श {crop}",
        "your_soil_label": "तुमची माती",
        "ideal_soil_label": "आदर्श माती",
        "advisory_title": "📢 थेट कृषी सल्लागार इशारे",
        "advisory_pest_risk": "🦟 <strong>कीड आणि रोगाचा धोका</strong>: हवेतील जास्त आर्द्रता आणि उष्ण हवामानामुळे खोडकिडा व बुरशीजन्य रोगांचा प्रादुर्भाव वाढू शकतो.",
        "advisory_water_risk": "🌊 <strong>पाणी साचण्याचा धोका</strong>: अतिवृष्टीमुळे पिकात पाणी साचू शकते. मुळे कुजण्यापासून वाचवण्यासाठी शेतात योग्य निचरा व्यवस्था करा.",
        "advisory_acid_risk": "🧪 <strong>अम्लीय अडथळा</strong>: कमी पीएच मुळे पिकांना स्फुरद (फॉस्फरस) मिळत नाही. माती उदासीन करण्यासाठी कृषी चुना वापरा.",
        "advisory_alk_risk": "🧪 <strong>क्षारीय अडथळा</strong>: जास्त पीएच मुळे सूक्ष्म अन्नद्रव्यांचे शोषण थांबते. शेतात सेंद्रिय जिप्सम वापरण्याचा सल्ला दिला जातो.",
        "advisory_optimal": "✅ <strong>अनुकूल हवामान</strong>: सध्याच्या पीक चक्रासाठी कोणताही हवामान किंवा पर्यावरणाचा धोका नाही.",
        "leaderboard_advisory_title": "📊 पर्यायी पीक विश्लेषण आणि शेतकरी सल्ला",
        "leaderboard_revenue_opt": "💰 <strong>कमाल उत्पन्न पर्याय</strong>: <strong>{crop}</strong> ची लागवड केल्यास <strong>INR {revenue:,.2f}</strong> चा सर्वाधिक नफा मिळण्याची शक्यता आहे.",
        "leaderboard_yield_opt": "🌾 <strong>कमाल घाऊक उत्पादन</strong>: जर आपले उद्दिष्ट जास्त उत्पादनाचे असेल, तर <strong>{crop}</strong> ची शिफारस केली जाते, ज्यामध्ये <strong>{yield_val:.2f} {unit}</strong> एवढे अपेक्षित उत्पन्न मिळू शकते.",
        "leaderboard_diversification": "🚜 <strong>जोखीम नियंत्रण</strong>: पीक नुकसान आणि बाजार भावातील घसरणीपासून बचावासाठी शेतकऱ्यांनी जमिनीची विभागणी करून लागवड करावी (उदा. ७०% मुख्य पीक, ३०% पर्यायी पीक).",
        "help_n": "मातीतील नायट्रोजन (N) चे प्रमाण (शिफारस: १०-१२० mg/kg). पानांच्या वाढीसाठी आणि पिकाच्या आरोग्यासाठी महत्त्वाचे.",
        "help_p": "मातीतील फॉस्फरस (P) चे प्रमाण (शिफारस: ५-१०० mg/kg). मुळांच्या वाढीसाठी आणि फुलोऱ्यासाठी आवश्यक.",
        "help_k": "मातीतील पोटॅशियम (K) चे प्रमाण (शिफारस: ५-१०० mg/kg). रोगप्रतिकारक शक्ती आणि पिकाचा दर्जा वाढवते.",
        "help_ph": "मातीचा पीएच स्तर आम्लता किंवा क्षारता मोजतो. ७.० उदासीन आहे; बहुतेक पिके ६.० ते ७.५ पीएच दरम्यान चांगली वाढतात.",
        "help_area_acres": "एकर मधील एकूण लागवड क्षेत्र.",
        "help_area_ha": "हेक्टर मधील एकूण लागवड क्षेत्र. १ हेक्टर = २.४७ एकर.",
        "help_fert_bags": "५० किलोच्या प्रमाणित गोणी स्वरूपात लागवडीचे खत.",
        "help_fert_kg": "किलोग्रॅम (kg) स्वरूपात लागवडीचे खत.",
        "help_pest_bags": "५० किलोच्या प्रमाणित गोणी स्वरूपात लागवडीचे कीटकनाशक.",
        "help_pest_kg": "किलोग्रॅम (kg) स्वरूपात लागवडीचे कीटकनाशक.",
        "help_rain": "हंगामी पाऊस मिलिमीटर (mm) मध्ये. १ मिमी = १ लिटर प्रति चौरस मीटर पाणी.",
        "help_manual_temp": "तापमान सेल्सिअस (°C) मध्ये मॅन्युअली बदलण्यासाठी.",
        "help_manual_hum": "हवेतील आर्द्रता टक्केवारी (%) मॅन्युअली बदलण्यासाठी.",
        "help_manual_rain": "ताशी पाऊस (मीमी/तास) मॅन्युअली बदलण्यासाठी."
    },
    "Tamil": {
        "title": "ஸ்மார்ட் பயிர் ஆலோசகர்",
        "subtitle": "மண், இருப்பிடம் மற்றும் பருவகால வானிலை அளவுருக்களின் அடிப்படையில் உகந்த பயிர் பரிந்துரைகள், எதிர்பார்க்கப்படும் விளைச்சல் மற்றும் சாகுபடி பொருளாதாரத்தை கணித்தல்.",
        "lang_select": "மொழியைத் தேர்ந்தெடுக்கவும்",
        "settings_location": "📍 இருப்பிடம் & பயிர் அமைப்புகள்",
        "select_state": "மாநிலத்தைத் தேர்ந்தெடுக்கவும்",
        "select_city": "நகரத்தைத் தேர்ந்தெடுக்கவும்",
        "select_other_city": "இதர நகரத்தைத் தேர்ந்தெடுக்கவும்",
        "select_season": "பருவத்தைத் தேர்ந்தெடுக்கவும்",
        "settings_soil": "🧪 மண் வேதியியல் சுயவிவரம்",
        "soil_type": "மண் வகை",
        "soil_type_custom": "விருப்பப்படி (மண் பரிசோதனை அறிக்கை உள்ளது)",
        "n_label": "நைட்ரஜன் (N) - மி.கி/கிகி",
        "p_label": "பாஸ்பரஸ் (P) - மி.கி/கிகி",
        "k_label": "பொட்டாசியம் (K) - மி.கி/கிகி",
        "ph_label": "மண் pH அளவு",
        "settings_cultivation": "🚜 சாகுபடி அளவுருக்கள்",
        "area_label": "சாகுபடி பரப்பு",
        "fert_label": "திட்டமிட்ட உரங்கள்",
        "pest_label": "திட்டமிட்ட பூச்சிக்கொல்லிகள்",
        "settings_water": "💧 நீர் சுயவிவரம்",
        "rain_label": "எதிர்பார்க்கப்படும் பருவகால மழை (மிமீ)",
        "settings_override": "🔧 வானிலை API மேலெழுதல்",
        "override_label": "தற்போதைய வானிலையை மேலெழுதுக",
        "manual_temp": "தற்போதைய வெப்பநிலை (°C)",
        "manual_hum": "தற்போதைய ஈரப்பதம் (%)",
        "manual_rain": "தற்போதைய மழைப்பொழிவு (மிமீ/மணி)",
        "btn_recommend": "உகந்த பயிரைப் பரிந்துரைக்கவும்",
        "loading": "வானிலை தகவல்களைச் சேகரித்து சாகுபடி முடிவுகளைக் கணிக்கும்...",
        "err_city": "தயவுசெய்து சரியான நகரப் பெயரை உள்ளிடவும்.",
        "rec_header": "சிறந்த பயிர் பரிந்துரை",
        "calendar_header": "📅 பயிர் விதைப்பு & அறுவடை காலண்டர்",
        "sow_timeline": "விதைப்பு காலம்",
        "crop_duration": "பயிர் காலம்",
        "economics_header": "சாகுபடி பொருளாதாரம்",
        "pred_yield": "எதிர்பார்க்கப்படும் விளைச்சல்",
        "market_price": "உள்ளூர் சந்தை விலை",
        "est_revenue": "மதிப்பிடப்பட்ட மொத்த வருவாய்",
        "total_cost": "மதிப்பிடப்பட்ட சாகுபடி செலவு",
        "net_profit": "மதிப்பிடப்பட்ட நிகர லாபம்",
        "net_loss": "மதிப்பிடப்பட்ட நிகர இழப்பு",
        "soil_health_score": "மண்-பயிர் இணக்கத்தன்மை மதிப்பெண்",
        "risk_header": "📊 சந்தை விலை ஏற்ற இறக்கம் & அபாய பகுப்பாய்வு (±20%)",
        "risk_pessimistic": "மந்தமான சந்தை (-20% விலை)",
        "risk_optimistic": "வளர்ச்சியான சந்தை (+20% விலை)",
        "risk_high_alert": "⚠️ அதிக அபாயம்: சந்தை விலையில் 20% வீழ்ச்சி ஏற்பட்டால் INR {loss:,.2f} நிகர இழப்பு ஏற்படலாம்.",
        "risk_low_alert": "✅ குறைந்த அபாயம்: சந்தை விலையில் 20% வீழ்ச்சி ஏற்பட்டாலும், இந்த பயிர் மூலம் INR {profit:,.2f} நிகர லாபம் கிடைக்கும் என எதிர்பார்க்கப்படுகிறது.",
        "alternative_crops": "💡 மாற்று பயிர் விளைச்சல் தரவரிசை",
        "unit_hectares": "ஹெக்டேர்",
        "unit_acres": "ஏக்கர்",
        "unit_kg": "கிலோ",
        "unit_bags": "மூட்டை (50கிகி)",
        "units_title": "📏 அளவீட்டு அலகுகள்",
        "alignment_header": "💡 சாகுபடி கால அட்டவணை சீரமைப்பு",
        "alignment_msg": "மண் குறியீடுகளின் அடிப்படையில் மெஷின் லேர்னிங் மாதிரி **நெல் (Rice)** சாகுபடியைப் பரிந்துரைத்துள்ளது. இருப்பினும், காரிஃப் பருவத்தில் (ஜூலை) இப்பகுதியில் **சோளம் (Maize)** அல்லது **கம்பு (Jowar)** போன்ற தானிய வகைகளே அதிகம் பயிரிடப்படுகின்றன. நெல் சாகுபடி பொதுவாக அக்டோபர் (ரபி பருவம்) மாதத்தில் நடவு செய்யப்படுகிறது.",
        "comparison_title": "🔍 மண் சத்துக்கள் ஒப்பீடு: உங்கள் மண் vs சிறந்த {crop}",
        "your_soil_label": "உங்கள் மண்",
        "ideal_soil_label": "தேவைப்படும் அளவு",
        "advisory_title": "📢 நேரடி விவசாய எச்சரிக்கைகள்",
        "advisory_pest_risk": "🦟 <strong>பூச்சி மற்றும் நோய் அபாயம்</strong>: அதிக ஈரப்பதம் மற்றும் சூடான காலநிலை காரணமாக குலை நோய் மற்றும் தண்டு துளைப்பான் பூச்சிகளின் அபாயம் அதிகமாகும்.",
        "advisory_water_risk": "🌊 <strong>வெள்ள அபாயம்</strong>: மிக அதிக மழைப்பொழிவு. வேரழுகல் நோயைத் தடுக்க சாகுபடி நிலத்தில் வடிகால் வசதியை ஏற்படுத்தவும்.",
        "advisory_acid_risk": "🧪 <strong>அமிலத்தன்மை அபாயம்</strong>: குறைந்த pH அளவு பாஸ்பரஸ் சத்து கிடைப்பதைத் தடுக்கும். நடுநிலைப்படுத்த விவசாய சுண்ணாம்பு இடவும்.",
        "advisory_alk_risk": "🧪 <strong>காரத்தன்மை அபாயம்</strong>: அதிக pH அளவு நுண்ணூட்டச் சத்துக்களை மண் உறிஞ்சுவதைத் தடுக்கும். ஜிப்சம் பயன்படுத்த பரிந்துரைக்கப்படுகிறது.",
        "advisory_optimal": "✅ <strong>சாதகமான வளரும் சூழல்</strong>: இந்த சாகுபடி காலத்தில் சுற்றுச்சூழல் அபாயங்கள் எதுவும் இல்லை.",
        "leaderboard_advisory_title": "📊 மாற்று பயிர் பகுப்பாய்வு & விவசாய ஆலோசனை",
        "leaderboard_revenue_opt": "💰 <strong>அதிக வருவாய் தரும் பயிர்</strong>: <strong>{crop}</strong> சாகுபடி செய்வதன் மூலம் <strong>INR {revenue:,.2f}</strong> வரை அதிகபட்ச வருவாய் கிடைக்க வாய்ப்புள்ளது.",
        "leaderboard_yield_opt": "🌾 <strong>அதிக விளைச்சல் தரும் பயிர்</strong>: உங்கள் இலக்கு அதிக விளைச்சல் எனில், <strong>{crop}</strong> பயிரிடுவதன் மூலம் <strong>{yield_val:.2f} {unit}</strong> வரை சிறந்த விளைச்சல் பெறலாம்.",
        "leaderboard_diversification": "🚜 <strong>வருவாய் பாதுகாப்பு முறை</strong>: பயிர் சேதம் மற்றும் சந்தை விலை வீழ்ச்சியிலிருந்து தப்பிக்க, விவசாயிகள் தங்கள் நிலத்தைப் பகிர்ந்து பயிரிடுமாறு அறிவுறுத்தப்படுகிறார்கள் (எ.கா. 70% முதன்மை பயிர், 30% மாற்று பயிர்).",
        "help_n": "மண்ணில் உள்ள தழைச்சத்து (N) அளவு (பரிந்துரை: 10-120 மி.கி/கிகி). பயிர் வளர்ச்சிக்கும் பச்சை நிறத்திற்கும் உதவும்.",
        "help_p": "மண்ணில் உள்ள மணிச்சத்து (P) அளவு (பரிந்துரை: 5-100 மி.கி/கிகி). வேர் வளர்ச்சி மற்றும் பூ பூக்க மிகவும் அவசியம்.",
        "help_k": "மண்ணில் உள்ள சாம்பல்சத்து (K) அளவு (பரிந்துரை: 5-100 மி.கி/கிகி). பயிர் நோய் எதிர்ப்பு சக்தியை அதிகரிக்கும்.",
        "help_ph": "மண் pH அளவு அமில/காரத் தன்மையைக் குறிக்கும். 7.0 நடுநிலை; பயிர்கள் 6.0  முதல் 7.5 pH அளவில் நன்றாக வளரும்.",
        "help_area_acres": "ஏக்கர் கணக்கில் சாகுபடி பரப்பு.",
        "help_area_ha": "ஹெக்டேர் கணக்கில் சாகுபடி பரப்பு. 1 ஹெக்டேர் = 2.47 ஏக்கர்.",
        "help_fert_bags": "50 கிலோ எடை கொண்ட மூட்டைகளில் உரங்களின் அளவு.",
        "help_fert_kg": "கிலோகிராம் (kg) கணக்கில் உரங்களின் அளவு.",
        "help_pest_bags": "50 கிலோ எடை கொண்ட மூட்டைகளில் பூச்சிக்கொல்லிகளின் அளவு.",
        "help_pest_kg": "கிலோகிராம் (kg) கணக்கில் பூச்சிக்கொல்லிகளின் அளவு.",
        "help_rain": "மில்லிமீட்டரில் (mm) பருவகால மழை அளவு. 1 மிமீ = ஒரு சதுர மீட்டருக்கு 1 லிட்டர் நீர்.",
        "help_manual_temp": "வெப்பநிலையை செல்சியஸ் (°C) அளவில் கைமுறையாக மாற்ற.",
        "help_manual_hum": "ஈரப்பதத்தை சதவீத (%) அளவில் கைமுறையாக மாற்ற.",
        "help_manual_rain": "மழைப்பொழிவை (மிமீ/மணி) கைமுறையாக மாற்ற."
    },
    "Bengali": {
        "title": "স্মার্ট ফসল উপদেষ্টা",
        "subtitle": "মাটি, স্থান এবং ঋতুভিত্তিক আবহাওয়ার তথ্যের ভিত্তিতে ফসলের সুপারিশ, ফলন এবং চাষের অর্থনৈতিক হিসাব অনুমান করা।",
        "lang_select": "ভাষা নির্বাচন করুন",
        "settings_location": "📍 স্থান এবং ফসল সেটিংস",
        "select_state": "রাজ্য নির্বাচন করুন",
        "select_city": "শহর নির্বাচন করুন",
        "select_other_city": "অন্য শহর নির্বাচন করুন",
        "select_season": "ঋতু নির্বাচন করুন",
        "settings_soil": "🧪 মাটির রাসায়নিক প্রোফাইল",
        "soil_type": "মাটির ধরন",
        "soil_type_custom": "কাস্টম (আমার কাছে মাটির পরীক্ষা রিপোর্ট আছে)",
        "n_label": "নাইট্রোজেন (N) - মিলিগ্রাম/কেজি",
        "p_label": "ফসফরাস (P) - মিলিগ্রাম/কেজি",
        "k_label": "পটাশিয়াম (K) - মিলিগ্রাম/কেজি",
        "ph_label": "মাটির পিএইচ (pH) মাত্রা",
        "settings_cultivation": "🚜 চাষের প্যারামিটার",
        "area_label": "চাষের এলাকা",
        "fert_label": "পরিকল্পিত সার",
        "pest_label": "পরিকল্পিত কীটনাশক",
        "settings_water": "💧 জলের প্রোফাইল",
        "rain_label": "প্রত্যাশিত মরসুমী বৃষ্টিপাত (মিমি)",
        "settings_override": "🔧 আবহাওয়া এপিআই ওভাররাইड করুন",
        "override_label": "লাইভ আবহাওয়া ওভাররাইড করুন",
        "manual_temp": "বর্তমান তাপমাত্রা (°C)",
        "manual_hum": "বর্তমান আর্দ্রতা (%)",
        "manual_rain": "বর্তমান বৃষ্টিপাত (মিমি/ঘণ্টা)",
        "btn_recommend": "উপযুক্ত ফসলের সুপারিশ পান",
        "loading": "আবহাওয়ার তথ্য সংগ্রহ করা হচ্ছে এবং ফলন গণনা করা হচ্ছে...",
        "err_city": "অনুগ্রহ করে একটি সঠিক শহরের নাম লিখুন।",
        "rec_header": "সর্বোত্তম প্রস্তাবিত ফসল",
        "calendar_header": "📅 ফসল বপন এবং সংগ্রহের ক্যালেন্ডার",
        "sow_timeline": "বপনের সময়কাল",
        "crop_duration": "ফসলের সময়কাল",
        "economics_header": "চাষের অর্থনৈতিক হিসাব",
        "pred_yield": "অনুমিত ফলন",
        "market_price": "স্থানীয় বাজার মূল্য",
        "est_revenue": "অনুমিত মোট আয়",
        "total_cost": "অনুমিত চাষের খরচ",
        "net_profit": "অনুমিত নিট লাভ",
        "net_loss": "অনুমিত নিট ক্ষতি",
        "soil_health_score": "মাটি-ফসলের সামঞ্জস্যের স্কোর",
        "risk_header": "📊 বাজার মূল্যের ওঠানামা এবং ঝুঁকি বিশ্লেষণ (±২০%)",
        "risk_pessimistic": "মন্দার বাজার (-২০% বাজার মূল্য)",
        "risk_optimistic": "তেজির বাজার (+২০% বাজার মূল্য)",
        "risk_high_alert": "⚠️ উচ্চ ঝুঁকি: বাজার মূল্যে ২০% পতন হলে INR {loss:,.2f} নিট লোকসান হতে পারে।",
        "risk_low_alert": "✅ কম ঝুঁকি: বাজার মূল্যে ২০% পতন হলেও, এই ফসল থেকে INR {profit:,.2f} নিট লাভ আশা করা যায়।",
        "alternative_crops": "💡 বিকল্প ফসলের ফলনের লিডারবোর্ড",
        "unit_hectares": "হেক্টর",
        "unit_acres": "একর",
        "unit_kg": "কেজি",
        "unit_bags": "বস্তা (৫০ কেজি)",
        "units_title": "📏 পরিমাপের একক",
        "alignment_header": "💡 ফসল বপন চক্র সমন্বয়",
        "alignment_msg": "মাটির উপাদানের উপর ভিত্তি করে মেশিন লার্নিং মডেল **ধান (Rice)** চাষের সুপারিশ করেছে। তবে, মনে রাখবেন যে খরিফ মরসুমে (জুলাই) এই অঞ্চলে **ভুট্টা (Maize)** বা **জোয়ার (Jowar)** জাতীয় শস্য বেশি চাষ হয়। ধান সাধারণত অক্টোবর (রবি মরসুমে) রোপণ করা হয়।",
        "comparison_title": "🔍 মাটির পুষ্টি উপাদানের তুলনা: আপনার জমি বনাম আদেশ {crop}",
        "your_soil_label": "আপনার জমির মাটি",
        "ideal_soil_label": "আদর্শ মাটি",
        "advisory_title": "📢 লাইভ কৃষি সংক্রান্ত সতর্কতা",
        "advisory_pest_risk": "🦟 <strong>পোকামাকড় ও রোগবালাইয়ের ঝুঁকি</strong>: উচ্চ আর্দ্রতা और উষ্ণ আবহাওয়া ব্লাস্ট রোগ ও মাজরা পোকার আক্রমণ বাড়িয়ে দেয়।",
        "advisory_water_risk": "🌊 <strong>জলবদ্ধতার ঝুঁকি</strong>: অতিরিক্ত মরসুমী বৃষ্টিপাত। মূল পচা রোগ প্রতিরোধ করতে জমিতে সঠিক নিষ্কাশন ব্যবস্থা রাখুন।",
        "advisory_acid_risk": "🧪 <strong>অম্লীয় মাটির প্রভাব</strong>: কম পিএইচ ফসফরাস গ্রহণ সীমিত করে। অম্লতা কমাতে জমিতে চুন প্রয়োগ করুন।",
        "advisory_alk_risk": "🧪 <strong>ক্ষারীয় মাটির প্রভাব</strong>: উচ্চ পিএইচ মাইক্রোনিউট্রিয়েন্ট শোষণে বাধা দেয়। জমিতে জিপসাম প্রয়োগ করার পরামর্শ দেওয়া হচ্ছে।",
        "advisory_optimal": "✅ <strong>অনুকূল আবহাওয়া</strong>: এই ফসল চক্রের জন্য বড় কোনো পরিবেশগত ঝুঁকি নেই।",
        "leaderboard_advisory_title": "📊 বিকল্প ফসল বিশ্লেষণ এবং চাষী পরামর্শ",
        "leaderboard_revenue_opt": "💰 <strong>সর্বোচ্চ আয়ের ফসল</strong>: <strong>{crop}</strong> চাষ করলে সর্বোচ্চ <strong>INR {revenue:,.2f}</strong> আয় করা সম্ভব।",
        "leaderboard_yield_opt": "🌾 <strong>সর্বোচ্চ ফলন</strong>: আপনার লক্ষ্য যদি বেশি ফলন হয়, তবে <strong>{crop}</strong> চাষ করুন, যা থেকে প্রত্যাশিত ফলন <strong>{yield_val:.2f} {unit}</strong> পাওয়া যাবে।",
        "leaderboard_diversification": "🚜 <strong>ঝুঁকি হ্রাসকরণ</strong>: ফসল নষ্ট হওয়া এবং বাজার দর কমে যাওয়ার ঝুঁকি এড়াতে কৃষকদের জমি ভাগ করে চাষ করার পরামর্শ দেওয়া হচ্ছে (যেমন, ৭০% প্রধান ফসল, ৩০% বিকল্প ফসল)।",
        "help_n": "মাটিতে নাইট্রোজেন (N) (প্রস্তাবিত: ১০-১২০ মিলিগ্রাম/কেজি)। পাতার বৃদ্ধি এবং ফসলের তেজ বাড়াতে সাহায্য করে।",
        "help_p": "মাটিতে ফসফরাস (P) (প্রস্তাবিত: ৫-১০০ মিলিগ্রাম/কেজি)। শিকড়ের বিকাশ और ফুল ফোটার জন্য প্রয়োজনীয়।",
        "help_k": "মাটিতে পটাশিয়াম (K) (প্রস্তাবিত: ৫-১০০ মিলিগ্রাম/কেজি)। রোগ প্রতিরোধ ক্ষমতা বাড়ায়।",
        "help_ph": "মাটির পিএইচ মাত্রা অম্লতা/ক্ষারত্ব নির্দেশ করে। ৭.০ নিরপেক্ষ; অধিকাংশ ফসল ६.० থেকে ৭.৫ পিএইচ-এ ভালো হয়।",
        "help_area_acres": "একর চাষের মোট এলাকা।",
        "help_area_ha": "হেক্টর চাষের মোট এলাকা। ১ হেক্টর = ২.৪৭ একর।",
        "help_fert_bags": "৫০ কেজি বস্তা হিসেবে পরিকল্পিত সারের মোট পরিমাণ।",
        "help_fert_kg": "কেজি (kg) হিসেবে পরিকল্পিত সারের মোট পরিমাণ।",
        "help_pest_bags": "৫০ কেজি বস্তา হিসেবে পরিকল্পিত কীটনাশকের মোট পরিমাণ।",
        "help_pest_kg": "কেজি (kg) হিসেবে পরিকল্পিত কীটনাশকের মোট পরিমাণ।",
        "help_rain": "মিমি (mm) হিসেবে প্রত্যাশিত বৃষ্টিপাত। ১ মিমি = প্রতি বর্গমিটারে ১ লিটার জল।",
        "help_manual_temp": "সেলসিয়াস (°C) হিসেবে তাপমাত্রা ম্যানুয়ালি সেট করার জন্য।",
        "help_manual_hum": "আর্দ্রতার শতকরা হার (%) ম্যানুয়ালি সেট করার लिए।",
        "help_manual_rain": "ঘণ্টায় বৃষ্টিপাত (মিমি) ম্যানুয়ালি সেট করার জন্য।"
    }
}

# 3. Defining soil type chemical profile constants...
SOIL_PROFILES = {
    "Alluvial Soil (River basins)": {"N": 70, "P": 50, "K": 60, "pH": 6.8},
    "Black Cotton Soil (Heavy clay)": {"N": 50, "P": 35, "K": 70, "pH": 7.8},
    "Red Soil (Drylands)": {"N": 40, "P": 30, "K": 35, "pH": 5.8},
    "Sandy Loam (Hilly/River beds)": {"N": 35, "P": 25, "K": 25, "pH": 6.0},
    "Laterite Soil (Acidic hills)": {"N": 30, "P": 20, "K": 20, "pH": 5.2}
}

# 4. Loading serialized models, scalers, and column lists...
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

@st.cache_resource
def load_clf_artifacts():
    """Loading serialized classifier artifacts from disk..."""
    with open("crop_recommendation_model.pkl", "rb") as f:
        model_clf = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler_clf = pickle.load(f)
    with open("feature_cols_clf.pkl", "rb") as f:
        feature_cols_clf = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model_clf, scaler_clf, feature_cols_clf, le

@st.cache_data
def load_data_and_profiles():
    """Loading dynamic dataframes and generating crop averages..."""
    df_m = pd.read_csv("crop_dataset.csv")
    states = sorted(df_m["state"].unique().tolist())
    seasons = sorted(df_m["season"].unique().tolist())
    crops = sorted(df_m["crop"].unique().tolist())
    
    crop_yields = df_m.groupby("crop")["yield"].mean().to_dict()
    crop_ideals = df_m.groupby("crop")[["N", "P", "K", "pH", "total_rainfall_mm"]].mean().to_dict(orient="index")
    
    return states, seasons, crops, crop_yields, crop_ideals, df_m


# 5. Defining feature engineering function...
def engineer_features(df):
    """Engineering interaction variables..."""
    df_feat = df.copy()
    eps = 1e-5
    
    # 1. Total Soil Nutrients...
    df_feat['total_nutrients'] = df_feat['N'] + df_feat['P'] + df_feat['K']
    
    # 2. Nutrient Ratios (avoiding division by zero)...
    df_feat['N_P_ratio'] = df_feat['N'] / (df_feat['P'] + eps)
    df_feat['K_P_ratio'] = df_feat['K'] / (df_feat['P'] + eps)
    
    # 3. Climate Interactions...
    df_feat['temp_humidity_index'] = df_feat['avg_temp_c'] * df_feat['avg_humidity_percent'] / 100.0
    df_feat['rain_ph_interaction'] = df_feat['total_rainfall_mm'] * df_feat['pH']
    
    # 4. Input Intensity Indicators (only for classifier if columns exist)...
    if 'fertilizer' in df_feat.columns and 'area' in df_feat.columns:
        df_feat['fertilizer_per_area'] = df_feat['fertilizer'] / (df_feat['area'] + eps)
        df_feat['pesticide_per_area'] = df_feat['pesticide'] / (df_feat['area'] + eps)
        df_feat['fertilizer_pesticide_ratio'] = df_feat['fertilizer'] / (df_feat['pesticide'] + eps)
        df_feat['rain_fertilizer_interaction'] = df_feat['total_rainfall_mm'] * df_feat['fertilizer']
        
    return df_feat


# 6. Configuring live weather fetch...
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
            
            # Fetching weather condition ID and main status description...
            weather_id = data['weather'][0]['id'] if 'weather' in data and len(data['weather']) > 0 else 800
            weather_main = data['weather'][0]['main'].lower() if 'weather' in data and len(data['weather']) > 0 else ""
            
            rain_info = data.get('rain', {})
            rainfall = rain_info.get('1h', rain_info.get('3h', 0.0))
            
            # Estimating rainfall from weather codes if the direct rain gauge value is missing or 0.0...
            if rainfall == 0.0:
                # Codes 2xx: Thunderstorm, 3xx: Drizzle, 5xx: Rain
                if 200 <= weather_id < 300:
                    rainfall = 5.0 # Estimating thunderstorm rain...
                elif 300 <= weather_id < 400:
                    rainfall = 1.5 # Estimating drizzle...
                elif 500 <= weather_id < 600:
                    if weather_id in [500, 520, 531]:
                        rainfall = 2.0 # Estimating light rain...
                    else:
                        rainfall = 6.0 # Estimating moderate/heavy rain...
                elif "rain" in weather_main or "drizzle" in weather_main or "thunderstorm" in weather_main:
                    rainfall = 3.0 # Setting general rain fallback...
                    
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

# 7. Setting up dynamic UI components...
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
    <div class="custom-html-card" style='margin-top:0.5rem; margin-bottom: 1.5rem; color:#1E293B !important;'>
        <small style='color:#64748B !important;'>Soil pH Scale Meter</small>
        <div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#94A3B8 !important; margin-bottom:2px;'>
            <span style='color:#94A3B8 !important;'>Acidic (3.5)</span>
            <span style='color:{color} !important; font-weight:bold;'>{ph:.1f} - {status}</span>
            <span style='color:#94A3B8 !important;'>Alkaline (9.9)</span>
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
    <div class="custom-html-card" style='margin-top:1rem; margin-bottom: 1.5rem; color:#1E293B !important;'>
        <small style='color:#64748B !important;'>Nutrient Ratio (N : P : K)</small>
        <div style='display:flex; height:20px; border-radius:6px; overflow:hidden; margin-top:5px; box-shadow:inset 0 1px 2px rgba(0,0,0,0.1);'>
            <div style='width:{pct_n}%; background-color:#3B82F6; color:white !important; font-size:0.75rem; text-align:center; line-height:20px; font-weight:bold;' title='Nitrogen'>N ({pct_n:.0f}%)</div>
            <div style='width:{pct_p}%; background-color:#F59E0B; color:white !important; font-size:0.75rem; text-align:center; line-height:20px; font-weight:bold;' title='Phosphorus'>P ({pct_p:.0f}%)</div>
            <div style='width:{pct_k}%; background-color:#8B5CF6; color:white !important; font-size:0.75rem; text-align:center; line-height:20px; font-weight:bold;' title='Potassium'>K ({pct_k:.0f}%)</div>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8 !important; margin-top:3px;'>
            <span style='color:#94A3B8 !important;'>N: {n}</span>
            <span style='color:#94A3B8 !important;'>P: {p}</span>
            <span style='color:#94A3B8 !important;'>K: {k}</span>
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
    <div class="custom-weather-card" style='background:{bg_color}; padding:1.5rem; border-radius:16px; color:white !important; box-shadow:0 10px 15px -3px rgba(0, 0, 0, 0.1); margin-top:1rem;'>
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
        checks.append(f"<li>✅ <strong style='color:#1E293B !important;'>Nitrogen (N)</strong>: {n} is optimal (Dynamic Ideal: {ideal_n:.1f})</li>")
    else:
        checks.append(f"<li>⚠️ <strong style='color:#1E293B !important;'>Nitrogen (N)</strong>: {n} deviates from optimal {ideal_n:.1f}</li>")
        
    ideal_p = ideals['P']
    if 0.7 * ideal_p <= p <= 1.3 * ideal_p:
        checks.append(f"<li>✅ <strong style='color:#1E293B !important;'>Phosphorus (P)</strong>: {p} is optimal (Dynamic Ideal: {ideal_p:.1f})</li>")
    else:
        checks.append(f"<li>⚠️ <strong style='color:#1E293B !important;'>Phosphorus (P)</strong>: {p} deviates from optimal {ideal_p:.1f}</li>")
        
    ideal_k = ideals['K']
    if 0.7 * ideal_k <= k <= 1.3 * ideal_k:
        checks.append(f"<li>✅ <strong style='color:#1E293B !important;'>Potassium (K)</strong>: {k} is optimal (Dynamic Ideal: {ideal_k:.1f})</li>")
    else:
        checks.append(f"<li>⚠️ <strong style='color:#1E293B !important;'>Potassium (K)</strong>: {k} deviates from optimal {ideal_k:.1f}</li>")
        
    ideal_ph = ideals['pH']
    if abs(ph - ideal_ph) <= 0.8:
        checks.append(f"<li>✅ <strong style='color:#1E293B !important;'>pH Level</strong>: {ph:.1f} is optimal (Dynamic Ideal: {ideal_ph:.1f})</li>")
    else:
        checks.append(f"<li>⚠️ <strong style='color:#1E293B !important;'>pH Level</strong>: {ph:.1f} deviates from optimal {ideal_ph:.1f}</li>")
        
    ideal_rain = ideals['total_rainfall_mm']
    if 0.6 * ideal_rain <= rainfall <= 1.4 * ideal_rain:
        checks.append(f"<li>✅ <strong style='color:#1E293B !important;'>Seasonal Rainfall</strong>: {rainfall:.0f}mm is optimal (Dynamic Ideal: {ideal_rain:.0f}mm)</li>")
    else:
        checks.append(f"<li>⚠️ <strong style='color:#1E293B !important;'>Seasonal Rainfall</strong>: {rainfall:.0f}mm deviates from dynamic optimal {ideal_rain:.0f}mm</li>")
        
    return f"""
    <div class="custom-html-card" style='background-color:#F8FAFC; border: 1px solid #E2E8F0; padding:1.2rem; border-radius:12px; margin-top:1rem; color:#1E293B !important;'>
        <h4 style='margin:0 0 0.5rem 0; color:#047857 !important;'>🌱 Soil & Climate Suitability Checklist for {crop.title()}:</h4>
        <ul style='margin:0; padding-left:1.2rem; line-height:1.6; color:#1E293B !important;'>
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
        tips.append("<li>🧪 <strong style='color:#92400E !important;'>Low Nitrogen (N)</strong>: Plant leguminous cover crops or apply urea/compost.</li>")
    if p < 0.7 * ideals['P']:
        tips.append("<li>🧪 <strong style='color:#92400E !important;'>Low Phosphorus (P)</strong>: Blend superphosphate or bone meal into the soil.</li>")
    if k < 0.7 * ideals['K']:
        tips.append("<li>🧪 <strong style='color:#92400E !important;'>Low Potassium (K)</strong>: Scatter wood ash or apply muriate of potash.</li>")
    if ph < ideals['pH'] - 0.8:
        tips.append("<li>🧪 <strong style='color:#92400E !important;'>High Acidity</strong>: Add ground limestone to raise pH level.</li>")
    elif ph > ideals['pH'] + 0.8:
        tips.append("<li>🧪 <strong style='color:#92400E !important;'>High Alkalinity</strong>: Spread sulfur or peat moss to lower pH level.</li>")
        
    if not tips:
        tips.append("<li>✅ Your soil indicators are perfectly balanced for this crop profile!</li>")
        
    return f"""
    <div class="custom-html-card" style='background-color:#FFFBEB; border: 1px solid #FEF3C7; padding:1.2rem; border-radius:12px; margin-top:1rem; color:#92400E !important;'>
        <h4 style='margin:0 0 0.5rem 0; color:#92400E !important;'>🛠️ Soil Amendment Suggestions:</h4>
        <ul style='margin:0; padding-left:1.2rem; line-height:1.6; color:#92400E !important;'>
            {"".join(tips)}
        </ul>
    </div>
    """

def get_soil_health_score(n, p, k, ph, crop, crop_ideals):
    ideals = crop_ideals.get(crop)
    if not ideals:
        return ""
    
    score = 100
    score -= min(30, abs(n - ideals['N']) / ideals['N'] * 30)
    score -= min(30, abs(p - ideals['P']) / ideals['P'] * 30)
    score -= min(20, abs(k - ideals['K']) / ideals['K'] * 20)
    score -= min(20, abs(ph - ideals['pH']) / ideals['pH'] * 100)
    
    score = max(10, score)
    
    if score >= 80:
        badge = "Excellent Match 🌟"
        color = "#10B981"
    elif score >= 50:
        badge = "Moderate Match ⚖️"
        color = "#F59E0B"
    else:
        badge = "Poor Match ⚠️"
        color = "#EF4444"
        
    return f"""
    <div class="custom-html-card metric-card" style='border-left: 6px solid {color}; margin-top: 1rem; background-color:#F8FAFC !important;'>
        <small style='color:#64748B !important;'>Soil Health Compatibility Score</small>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <h2 style='margin:0; color:{color} !important;'>{score:.0f}%</h2>
            <span style='background-color:{color}; color:white !important; padding:0.2rem 0.6rem; border-radius:12px; font-size:0.8rem; font-weight:bold;'>{badge}</span>
        </div>
    </div>
    """

def get_fertilizer_calculator(n, p, k, crop, area, crop_ideals):
    ideals = crop_ideals.get(crop)
    if not ideals:
        return "", 0.0
    
    req_n = max(0.0, ideals['N'] - n) * 2.17
    req_p = max(0.0, ideals['P'] - p) * 5.0
    req_k = max(0.0, ideals['K'] - k) * 1.67
    
    tot_n = req_n * area
    tot_p = req_p * area
    tot_k = req_k * area
    
    urea_bags = tot_n / 50.0
    ssp_bags = tot_p / 50.0
    mop_bags = tot_k / 50.0
    
    cost_urea = urea_bags * 300
    cost_ssp = ssp_bags * 450
    cost_mop = mop_bags * 900
    total_cost = cost_urea + cost_ssp + cost_mop
    
    html = f"""
    <div class="custom-html-card" style='background-color:#F0F9FF; border: 1px solid #BAE6FD; padding:1.2rem; border-radius:12px; margin-top:1rem; color:#0369A1 !important;'>
        <h4 style='margin:0 0 0.5rem 0; color:#0369A1 !important;'>🧮 Fertilizer Requirements & Cost Estimation ({area:.1f} Hectares):</h4>
        <ul style='margin:0; padding-left:1.2rem; line-height:1.6; color:#0369A1 !important;'>
            <li>🧪 <strong style='color:#0369A1 !important;'>Urea (N source)</strong>: {urea_bags:.1f} bags (~{tot_n:.0f} kg) | Est Cost: INR {cost_urea:,.0f}</li>
            <li>🧪 <strong style='color:#0369A1 !important;'>SSP (P source)</strong>: {ssp_bags:.1f} bags (~{tot_p:.0f} kg) | Est Cost: INR {cost_ssp:,.0f}</li>
            <li>🧪 <strong style='color:#0369A1 !important;'>MOP (K source)</strong>: {mop_bags:.1f} bags (~{tot_k:.0f} kg) | Est Cost: INR {cost_mop:,.0f}</li>
        </ul>
        <div style='margin-top:0.75rem; border-top:1px dashed #BAE6FD; padding-top:0.5rem; font-weight:bold; display:flex; justify-content:space-between; color:#0369A1 !important;'>
            <span style='color:#0369A1 !important;'>Total Fertilizer Cost:</span>
            <span style='color:#0369A1 !important;'>INR {total_cost:,.2f}</span>
        </div>
    </div>
    """
    return html, total_cost

def get_crop_calendar(crop, season):
    season_clean = season.lower().strip()
    if "kharif" in season_clean:
        timeline = "Sow: 🌧️ June - July | Harvest: 🌾 September - October"
        duration = "90 - 120 Days"
    elif "rabi" in season_clean:
        timeline = "Sow: ❄️ October - November | Harvest: 🌾 February - March"
        duration = "120 - 150 Days"
    elif "summer" in season_clean:
        timeline = "Sow: ☀️ February - March | Harvest: ⛈️ May - June"
        duration = "90 - 110 Days"
    else:
        timeline = "Sow: 🌱 Year-round | Harvest: 🔄 Cycle-based"
        duration = "Perennial / Multi-harvest"
        
    return f"""
    <div class="custom-html-card" style='background: #F1F5F9; border: 1px solid #CBD5E1; padding: 1rem; border-radius: 12px; margin-top: 1rem; color: #1E293B !important;'>
        <h4 style='margin: 0 0 0.5rem 0; color: #475569 !important; font-weight: 600;'>📅 Crop Sowing & Harvesting Calendar</h4>
        <div style='display: flex; justify-content: space-between; font-size: 0.85rem;'>
            <div><strong style='color:#1E293B !important;'>Sowing Timeline:</strong><br/><span style='color: #475569 !important;'>{timeline}</span></div>
            <div style='text-align: right;'><strong style='color:#1E293B !important;'>Crop Duration:</strong><br/><span style='color: #475569 !important;'>{duration}</span></div>
        </div>
    </div>
    """

# 8. Loading dynamic lists and dataframes...
states, seasons, crops, crop_yields, crop_ideals, df_full = load_data_and_profiles()
cities = sorted(df_full["city"].unique().tolist())

# Sidebar: Language Selection
lang_input = st.sidebar.selectbox("Select Language / भाषा चुनें", list(TRANSLATIONS.keys()), index=0)
t = TRANSLATIONS[lang_input]

# Page Header
st.markdown(f"""
<div class="title-banner">
    <h1>{t['title']}</h1>
    <p>{t['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: Measurement Units Selector
st.sidebar.markdown(f"### {t['units_title']}")
unit_system = st.sidebar.radio(
    t['units_title'],
    [t['unit_hectares'] + " & Kilograms", t['unit_acres'] + " & " + t['unit_bags']],
    index=0,
    label_visibility="collapsed"
)
use_acres = (t['unit_acres'] in unit_system)

# Sidebar: Location & Crop Settings
st.sidebar.markdown(f"### {t['settings_location']}")
state_input = st.sidebar.selectbox(t['select_state'], states, index=states.index("Andhra Pradesh") if "Andhra Pradesh" in states else 0)

# Selecting from main cities of the state...
main_cities = MAIN_STATE_CITIES.get(state_input, ["Hyderabad", "Bangalore", "Mumbai", "New Delhi"])
city_select = st.sidebar.selectbox(t['select_city'], main_cities + ["Other Cities..."], index=0)

if city_select == "Other Cities...":
    other_cities = OTHER_STATE_CITIES.get(state_input, [])
    city = st.sidebar.selectbox(t['select_other_city'], other_cities, index=0) if other_cities else main_cities[0]
else:
    city = city_select

season_input = st.sidebar.selectbox(t['select_season'], seasons, index=seasons.index("Kharif") if "Kharif" in seasons else 0)

# Querying dynamic defaults based on city/season...
city_season_data = df_full[(df_full['city'] == city) & (df_full['season'] == season_input)]
default_fert = float(city_season_data['fertilizer'].mean()) if not city_season_data.empty else 80000.0
default_pest = float(city_season_data['pesticide'].mean()) if not city_season_data.empty else 300.0
default_area = float(city_season_data['area'].mean()) if not city_season_data.empty else 1.0
default_rain_avg = float(city_season_data['total_rainfall_mm'].mean()) if not city_season_data.empty else 1000.0
default_temp_avg = float(city_season_data['avg_temp_c'].mean()) if not city_season_data.empty else 25.6
default_hum_avg = float(city_season_data['avg_humidity_percent'].mean()) if not city_season_data.empty else 70.0

default_n = int(city_season_data['N'].mean()) if not city_season_data.empty else 50
default_p = int(city_season_data['P'].mean()) if not city_season_data.empty else 40
default_k = int(city_season_data['K'].mean()) if not city_season_data.empty else 30
default_ph = float(city_season_data['pH'].mean()) if not city_season_data.empty else 6.5

# Sidebar: Soil Type Selector
st.sidebar.markdown(f"### {t['settings_soil']}")
soil_type = st.sidebar.selectbox(t['soil_type'], [t['soil_type_custom']] + list(SOIL_PROFILES.keys()), index=0)

if soil_type != t['soil_type_custom']:
    profile_key = next(k for k in SOIL_PROFILES.keys() if k in soil_type)
    soil_profile = SOIL_PROFILES[profile_key]
    n_init = soil_profile["N"]
    p_init = soil_profile["P"]
    k_init = soil_profile["K"]
    ph_init = soil_profile["pH"]
else:
    n_init = default_n
    p_init = default_p
    k_init = default_k
    ph_init = default_ph

n = st.sidebar.slider(t['n_label'], 10, 120, n_init, key=f"n_slider_{city}_{season_input}_{soil_type}", help=t['help_n'])
p = st.sidebar.slider(t['p_label'], 5, 100, p_init, key=f"p_slider_{city}_{season_input}_{soil_type}", help=t['help_p'])
k = st.sidebar.slider(t['k_label'], 5, 100, k_init, key=f"k_slider_{city}_{season_input}_{soil_type}", help=t['help_k'])

st.sidebar.markdown(get_npk_breakdown(n, p, k), unsafe_allow_html=True)

ph = st.sidebar.slider(t['ph_label'], 3.5, 9.9, ph_init, step=0.1, key=f"ph_slider_{city}_{season_input}_{soil_type}", help=t['help_ph'])
st.sidebar.markdown(get_ph_indicator(ph), unsafe_allow_html=True)

st.sidebar.markdown("---")

# Sidebar: Cultivation Parameters
st.sidebar.markdown(f"### {t['settings_cultivation']}")

if use_acres:
    # Converting defaults for Acres and Bags representation...
    area_default_val = default_area / 0.4047
    fert_default_val = default_fert / 50.0
    pest_default_val = default_pest / 50.0
    
    input_area = st.sidebar.number_input(f"{t['area_label']} ({t['unit_acres']})", 0.1, 25000000.0, area_default_val, step=0.5, key=f"area_{city}_{season_input}", help=t['help_area_acres'])
    input_fert = st.sidebar.number_input(f"{t['fert_label']} ({t['unit_bags']})", 0.0, 2000000.0, fert_default_val, step=10.0, key=f"fert_{city}_{season_input}", help=t['help_fert_bags'])
    input_pest = st.sidebar.number_input(f"{t['pest_label']} ({t['unit_bags']})", 0.0, 200000.0, pest_default_val, step=1.0, key=f"pest_{city}_{season_input}", help=t['help_pest_bags'])
    
    # Internal variables scaled back to Hectares & kg for ML compatibility...
    planting_area = input_area * 0.4047
    fertilizer_input = input_fert * 50.0
    pesticide_input = input_pest * 50.0
else:
    planting_area = st.sidebar.number_input(f"{t['area_label']} ({t['unit_hectares']})", 0.1, 10000000.0, default_area, step=0.5, key=f"area_{city}_{season_input}", help=t['help_area_ha'])
    fertilizer_input = st.sidebar.number_input(f"{t['fert_label']} ({t['unit_kg']})", 0.0, 100000000.0, default_fert, step=1000.0, key=f"fert_{city}_{season_input}", help=t['help_fert_kg'])
    pesticide_input = st.sidebar.number_input(f"{t['pest_label']} ({t['unit_kg']})", 0.0, 10000000.0, default_pest, step=50.0, key=f"pest_{city}_{season_input}", help=t['help_pest_kg'])

# Sidebar: Water Settings
st.sidebar.markdown(f"### {t['settings_water']}")
rainfall_input = st.sidebar.slider(t['rain_label'], 100, 3500, int(default_rain_avg), step=50, key=f"rain_slider_{city}_{season_input}", help=t['help_rain'])

st.sidebar.markdown("---")

# Sidebar: Weather API Override Settings
st.sidebar.markdown(f"### {t['settings_override']}")
override_weather = st.sidebar.checkbox(t['override_label'], value=False)
if override_weather:
    manual_temp = st.sidebar.slider(t['manual_temp'], 10.0, 50.0, float(default_temp_avg), step=0.5, help=t['help_manual_temp'])
    manual_humidity = st.sidebar.slider(t['manual_hum'], 10, 100, int(default_hum_avg), help=t['help_manual_hum'])
    manual_current_rain = st.sidebar.slider(t['manual_rain'], 0.0, 50.0, 0.0, step=0.5, help=t['help_manual_rain'])
else:
    manual_temp = 0.0
    manual_humidity = 0
    manual_current_rain = 0.0

# 9. Setting up recommendation click trigger...
btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    predict_button = st.button(t['btn_recommend'], use_container_width=True)

if predict_button:
    if not city:
        st.error(t['err_city'])
    else:
        with st.spinner(t['loading']):
            model, scaler, feature_cols = load_artifacts()
            model_clf, scaler_clf, feature_cols_clf, le = load_clf_artifacts()
            
            # Retrieving weather parameters...
            if override_weather:
                temp = manual_temp
                humidity = manual_humidity
                current_rain = manual_current_rain
            else:
                temp, humidity, current_rain = get_weather(city)
            
            # --- 1. Recommendation Prediction via Classifier ---
            row_clf = {
                'year': 2026,
                'avg_temp_c': temp,
                'total_rainfall_mm': rainfall_input,
                'avg_humidity_percent': humidity,
                'N': n,
                'P': p,
                'K': k,
                'pH': ph,
                'area': planting_area,
                'fertilizer': fertilizer_input,
                'pesticide': pesticide_input
            }
            # Assigning classifier season and city dummies...
            for se in seasons:
                row_clf[f'season_{se}'] = 1.0 if se == season_input else 0.0
            for ci in cities:
                row_clf[f'city_{ci}'] = 1.0 if ci == city else 0.0
                
            df_clf = pd.DataFrame([row_clf])
            df_clf_eng = engineer_features(df_clf)
            # Reindexing to match the exact features the classifier scaler was fit on...
            df_clf_eng = df_clf_eng.reindex(columns=list(scaler_clf.feature_names_in_), fill_value=0)
            df_clf_scaled = pd.DataFrame(scaler_clf.transform(df_clf_eng), columns=df_clf_eng.columns)

            
            pred_clf_encoded = model_clf.predict(df_clf_scaled)[0]
            recommended = le.inverse_transform([pred_clf_encoded])[0]
            
            # --- Agronomic Rule-Based Alignment (Post-Processing) ---
            # Flagging general seasonal crop cycle alerts for real-world precision
            has_override_msg = False
            if season_input.lower() == "kharif" and recommended.lower() == "rice":
                has_override_msg = True
            
            # --- 2. Yield & Economics Prediction via Regressor ---
            # Evaluating predicted yields for all candidate crops...
            candidate_rows = []
            for crop in crops:
                row = {
                    'year': 2026,
                    'avg_temp_c': default_temp_avg,
                    'total_rainfall_mm': rainfall_input,
                    'avg_humidity_percent': default_hum_avg,
                    'N': n,
                    'P': p,
                    'K': k,
                    'pH': ph
                }
                # Assigning regressor one-hot dummies...
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
            df_eng = engineer_features(df_candidates)
            # Reindexing to match the exact features the regressor scaler was fit on...
            df_eng = df_eng.reindex(columns=list(scaler.feature_names_in_), fill_value=0)
            df_scaled = pd.DataFrame(scaler.transform(df_eng), columns=df_eng.columns)

            
            predicted_yields = model.predict(df_scaled)
            
            # Calculating revenues...
            results = []
            for idx_c, crop in enumerate(crops):
                py = predicted_yields[idx_c]
                price = CROP_PRICES.get(crop.lower().strip(), 30000)
                rev = py * price
                results.append((crop, py, rev, price))
                
            # Lookup recommended crop details from regressor results...
            rec_details = [r for r in results if r[0].lower().strip() == recommended.lower().strip()]
            if rec_details:
                _, yield_val, profit, price = rec_details[0]
            else:
                price = CROP_PRICES.get(recommended.lower().strip(), 30000)
                yield_val = 1.0
                profit = price * yield_val
                
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
            
            # Sort regressor results for alternative crops leaderboard...
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Formatting financial displays based on selected units...
            if use_acres:
                display_area = planting_area / 0.4047
                area_label_str = t['unit_acres']
                
                # Conversion to per-acre values...
                yield_val_display = yield_val * 0.4047 # Yield in MT per Acre
                price_display = price # Price remains per MT
                yield_sub_text = f"Total Production: {(yield_val_display * display_area):.1f} MT"
                price_sub_text = f"Market Rate: INR {price_display:,}/MT"
            else:
                display_area = planting_area
                area_label_str = t['unit_hectares']
                yield_val_display = yield_val
                price_display = price
                yield_sub_text = f"Total Production: {(yield_val_display * display_area):.1f} MT"
                price_sub_text = f"Market Rate: INR {price_display:,}/MT"
                
            col_left, col_right = st.columns([3, 2], gap="large")
            
            with col_left:
                st.markdown(f"""
                <div class="prediction-header">
                    <h2 style='color:#065F46; margin:0;'>{t['rec_header']}: {recommended.upper()}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                if has_override_msg:
                    st.info(f"{t['alignment_header']}: {t['alignment_msg']}")
                
                # Dynamic Soil Health compatibility score...
                st.markdown(get_soil_health_score(n, p, k, ph, recommended, crop_ideals), unsafe_allow_html=True)
                
                # Suitability and tips based on dynamic totals...
                st.markdown(get_suitability_check(recommended, n, p, k, ph, rainfall_input, crop_ideals), unsafe_allow_html=True)
                st.markdown(get_amendment_tips(n, p, k, ph, recommended, crop_ideals), unsafe_allow_html=True)
                st.markdown(fert_html, unsafe_allow_html=True)
            
            with col_right:
                # Displaying live metrics alongside seasonal averages in the card...
                display_city = city + " (Override)" if override_weather else city
                st.markdown(get_weather_card(temp, humidity, current_rain, default_temp_avg, default_hum_avg, rainfall_input, display_city), unsafe_allow_html=True)
                
                st.markdown(f"<h4 style='margin:1.5rem 0 0.5rem 0; color:#047857 !important;'>{t['economics_header']} ({display_area:.1f} {area_label_str}):</h4>", unsafe_allow_html=True)
                
                profit_color = "#10B981" if net_profit >= 0 else "#EF4444"
                profit_text = t['net_profit'] if net_profit >= 0 else t['net_loss']
                
                st.markdown(f"""
                <div style='display:flex; flex-direction:column; gap:1rem;'>
                    <div class="metric-card" style='background-color:#F8FAFC !important;'>
                        <small style='color:#64748B !important;'>{t['pred_yield']}</small>
                        <h2 style='margin: 0.25rem 0; color:#047857 !important;'>{yield_val_display:.2f} MT/{area_label_str[0].lower()}</h2>
                        <span style='color:#94A3B8 !important; font-size:0.85rem;'>{yield_sub_text}</span>
                    </div>
                    <div class="metric-card" style='background-color:#F8FAFC !important;'>
                        <small style='color:#64748B !important;'>{t['est_revenue']}</small>
                        <h2 style='margin: 0.25rem 0; color:#047857 !important;'>INR {gross_revenue:,.2f}</h2>
                        <span style='color:#94A3B8 !important; font-size:0.85rem;'>{price_sub_text}</span>
                    </div>
                    <div class="metric-card" style='background-color:#F8FAFC !important;'>
                        <small style='color:#64748B !important;'>{t['total_cost']}</small>
                        <h2 style='margin: 0.25rem 0; color:#475569 !important;'>INR {total_input_cost:,.2f}</h2>
                        <span style='color:#94A3B8 !important; font-size:0.85rem;'>Fertilizers + Seeds/Machinery/Labor</span>
                    </div>
                    <div class="metric-card" style='border-left: 4px solid {profit_color}; background-color:#F8FAFC !important;'>
                        <small style='color:#64748B !important;'>{profit_text}</small>
                        <h2 style='margin: 0.25rem 0; color:{profit_color} !important;'>INR {net_profit:,.2f}</h2>
                        <span style='color:#94A3B8 !important; font-size:0.85rem;'>Return on Investment</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- Row 2: Calendar & Risk Analysis ---
            st.markdown("---")
            col_left2, col_right2 = st.columns([3, 2], gap="large")
            
            with col_left2:
                st.markdown(get_crop_calendar(recommended, season_input), unsafe_allow_html=True)
                
            with col_right2:
                # --- 3. Price Volatility & Market Risk Calculations ---
                pessimistic_rev = gross_revenue * 0.80
                optimistic_rev = gross_revenue * 1.20
                
                pessimistic_profit = pessimistic_rev - total_input_cost
                optimistic_profit = optimistic_rev - total_input_cost
                
                st.markdown(f"#### {t['risk_header']}")
                
                # Color code warnings based on pessimistic outcomes...
                if pessimistic_profit < 0:
                    risk_alert = t['risk_high_alert'].format(loss=abs(pessimistic_profit))
                    risk_bg = "#FEF2F2"
                    risk_border = "#FEE2E2"
                    risk_text_color = "#991B1B"
                else:
                    risk_alert = t['risk_low_alert'].format(profit=pessimistic_profit)
                    risk_bg = "#F0FDF4"
                    risk_border = "#DCFCE7"
                    risk_text_color = "#166534"
                    
                st.markdown(f"""
                <div class="custom-html-card" style='background-color:{risk_bg}; border: 1px solid {risk_border}; padding:1.2rem; border-radius:12px; margin-top:0.75rem; color:{risk_text_color} !important;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:0.75rem; font-size:0.85rem;'>
                        <div><strong style='color:{risk_text_color} !important;'>{t['risk_pessimistic']}</strong><br/><span style='font-size:1.1rem; font-weight:bold; color:{risk_text_color} !important;'>INR {pessimistic_profit:,.2f}</span></div>
                        <div style='text-align:right;'><strong style='color:{risk_text_color} !important;'>{t['risk_optimistic']}</strong><br/><span style='font-size:1.1rem; font-weight:bold; color:{risk_text_color} !important;'>INR {optimistic_profit:,.2f}</span></div>
                    </div>
                    <div style='border-top:1px dashed {risk_border}; padding-top:0.5rem; font-size:0.85rem; font-weight:500; color:{risk_text_color} !important;'>
                        {risk_alert}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- Row 3: Soil Nutrient Comparison & Agronomic Advisory ---
            st.markdown("---")
            col_left3, col_right3 = st.columns([3, 2], gap="large")
            
            with col_left3:
                st.markdown(f"<h4 style='color:#047857 !important;'>{t['comparison_title'].format(crop=recommended.title())}</h4>", unsafe_allow_html=True)
                ideals = crop_ideals.get(recommended)
                if ideals:
                    df_compare = pd.DataFrame({
                        t['your_soil_label']: [n, p, k],
                        t['ideal_soil_label']: [ideals['N'], ideals['P'], ideals['K']]
                    }, index=['N', 'P', 'K'])
                    st.bar_chart(df_compare, height=250)
                    
            with col_right3:
                st.markdown(f"<h4 style='color:#047857 !important;'>{t['advisory_title']}</h4>", unsafe_allow_html=True)
                alerts = []
                if humidity > 80 and temp > 25:
                    alerts.append(f"<li>{t['advisory_pest_risk']}</li>")
                if rainfall_input > 1500:
                    alerts.append(f"<li>{t['advisory_water_risk']}</li>")
                if ph < 5.5:
                    alerts.append(f"<li>{t['advisory_acid_risk']}</li>")
                elif ph > 8.0:
                    alerts.append(f"<li>{t['advisory_alk_risk']}</li>")
                
                if not alerts:
                    alerts.append(f"<li>{t['advisory_optimal']}</li>")
                    
                st.markdown(f"""
                <div class="custom-html-card" style='background-color:#FFFBEB; border: 1px solid #FEF3C7; padding:1.2rem; border-radius:12px; color:#92400E !important;'>
                    <ul style='margin:0; padding-left:1.2rem; line-height:1.6; color:#92400E !important;'>
                        {"".join(alerts)}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Spanning full width below the columns: Top Alternative Crops
            st.markdown("---")
            st.markdown(f"<h3 style='color:#065F46; margin-top:1.5rem; margin-bottom:1rem;'>{t['alternative_crops']}</h3>", unsafe_allow_html=True)
            
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
                    is_rec = (c.lower().strip() == recommended.lower().strip())
                    bg = "#F0FDF4" if is_rec else "white"
                    border = "#DCFCE7" if is_rec else "#E2E8F0"
                    
                    # Yield scaling display...
                    alt_yield_disp = y * 0.4047 if use_acres else y
                    html_list += f"<div style='background: {bg}; border: 1px solid {border}; padding: 0.5rem 0.75rem; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; color: #1E293B !important;'><div><strong style='color:#1E293B;'>{rank+1}. {c.capitalize()}</strong><br/><span style='color:#64748B; font-size:0.75rem;'>Yield: {alt_yield_disp:.2f} MT/{area_label_str[0].lower()}</span></div><div style='text-align: right;'><strong style='color:#059669;'>INR {r:,.0f}</strong><br/><span style='color:#94A3B8; font-size:0.7rem;'>INR {p:,}/MT</span></div></div>"
                html_list += "</div>"
                st.markdown(html_list, unsafe_allow_html=True)
                
            # --- Dynamic Leaderboard Analysis & Farmer Advisory ---
            best_rev_crop = results[0][0].capitalize()
            best_rev_val = results[0][2]
            
            results_by_yield = sorted(results, key=lambda x: x[1], reverse=True)
            best_yield_crop = results_by_yield[0][0].capitalize()
            best_yield_val = results_by_yield[0][1]
            if use_acres:
                best_yield_val_disp = best_yield_val * 0.4047
                yield_unit = t['unit_acres']
            else:
                best_yield_val_disp = best_yield_val
                yield_unit = t['unit_hectares']
                
            st.markdown(f"""
            <div class="custom-html-card" style='background-color:#F0FDF4; border: 1px solid #DCFCE7; padding:1.2rem; border-radius:12px; margin-top:1rem; color:#166534 !important;'>
                <strong style='color:#166534 !important;'>{t['leaderboard_advisory_title']}:</strong>
                <ul style='margin:0.5rem 0 0 0; padding-left:1.2rem; line-height:1.6; color:#166534 !important;'>
                    <li>{t['leaderboard_revenue_opt'].format(crop=best_rev_crop, revenue=best_rev_val)}</li>
                    <li>{t['leaderboard_yield_opt'].format(crop=best_yield_crop, yield_val=best_yield_val_disp, unit=yield_unit)}</li>
                    <li>{t['leaderboard_diversification']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)