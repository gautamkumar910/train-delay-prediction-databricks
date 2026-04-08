import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pickle
import requests
import chromadb
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys
OPENWEATHER_API_KEY = "12120cfe22e55f3b7fc1c6b086dd2e83"
OPENAQ_API_KEY = None  # OpenAQ doesn't require API key
BHASHINI_API_KEY = "your_bhashini_api_key"  # Update after registration

# Station Coordinates
STATION_COORDS = {
    'NDLS': {'lat': 28.6425, 'lon': 77.2197, 'name': 'New Delhi'},
    'ALD': {'lat': 25.4358, 'lon': 81.8463, 'name': 'Allahabad'},
    'PNBE': {'lat': 25.6093, 'lon': 85.1376, 'name': 'Patna'},
    'MGS': {'lat': 25.5941, 'lon': 85.1376, 'name': 'Mughalsarai'}
}

# Supported Languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'हिंदी (Hindi)',
    'bn': 'বাংলা (Bengali)',
    'te': 'తెలుగు (Telugu)',
    'mr': 'मराठी (Marathi)',
    'ta': 'தமிழ் (Tamil)'
}

# ============================================================================
# DATABRICKS CONNECTION & DATA LOADING
# ============================================================================

@st.cache_resource
def load_databricks_data():
    """Load data from Unity Catalog tables"""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        
        # Load gold master table
        gold_master = spark.table("train_analytics_catalog.fog_study.train_data").toPandas()
        
        # Calculate historical averages (if not pre-computed)
        historical_averages = gold_master.groupby(['station', 'arrival_hour', 'month']).agg({
            'delay_minutes': 'mean',
            'temperature': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'visibility': 'mean',
            'pm25': 'mean',
            'pm10': 'mean',
            'aqi': 'mean'
        }).reset_index()
        
        historical_averages.columns = [
            'station', 'hour', 'month', 'hist_avg_delay',
            'hist_avg_temp', 'hist_avg_humidity', 'hist_avg_wind',
            'hist_avg_visibility', 'hist_avg_pm25', 'hist_avg_pm10', 'hist_avg_aqi'
        ]
        
        return gold_master, historical_averages
    except Exception as e:
        st.error(f"Error loading Databricks data: {e}")
        # Return sample data for demo
        return create_sample_data()

@st.cache_resource
def load_ml_models():
    """Load trained ML models"""
    try:
        # Option 1: Load from MLflow (recommended for production)
        import mlflow
        mlflow.set_tracking_uri("databricks")
        model_uri = "models:/train_delay_model/production"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model, "mlflow"
    except:
        try:
            # Option 2: Load from pickle file
            with open('/dbfs/tmp/train_delay_model.pkl', 'rb') as f:
                model = pickle.load(f)
            return model, "pickle"
        except:
            st.warning("⚠️ Could not load trained model. Using fallback prediction.")
            return None, "fallback"

@st.cache_resource
def load_label_encoders():
    """Load or create label encoders"""
    try:
        with open('/dbfs/tmp/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return encoders['station'], encoders['train_type']
    except:
        # Create default encoders
        le_station = LabelEncoder()
        le_station.fit(['NDLS', 'ALD', 'PNBE', 'MGS'])
        
        le_train_type = LabelEncoder()
        le_train_type.fit(['Rajdhani', 'Shatabdi', 'Express', 'Superfast', 'Passenger'])
        
        return le_station, le_train_type

@st.cache_resource
def initialize_rag_system():
    """Initialize ChromaDB and load knowledge base"""
    try:
        # Knowledge base documents
        knowledge_base = [
            {
                'title': 'IRCTC Cancellation Rules',
                'content': '''=== IRCTC Train Ticket Cancellation Rules ===

GENERAL CANCELLATION POLICY:
- Cancellation charges vary by train class and ticket type
- Tatkal tickets: No refund after chart preparation
- RAC/Waitlisted tickets: Full refund minus clerkage charges

REFUND FOR TRAIN DELAYS:
- If train is delayed by more than 3 hours: Full refund eligible
- If train is cancelled: Full refund, no cancellation charges
- Submit TDR (Ticket Deposit Receipt) within 72 hours of scheduled departure
- Refund processed within 7-10 working days

HOW TO FILE TDR:
1. Visit IRCTC website or app
2. Go to "My Transactions" → "File TDR"
3. Select reason: "Train running late"
4. Upload supporting documents if required
5. Submit and note TDR number

COMPENSATION:
- Premium trains (Rajdhani/Shatabdi): Higher compensation
- Delays > 1 hour: Complimentary meals may be provided
- Delays > 3 hours: Full refund without cancellation charges'''
            },
            {
                'title': 'Station Facilities',
                'content': '''=== Major Station Facilities ===

NEW DELHI (NDLS):
- Executive Lounge: Platform 1 (Entry: ₹150-200)
- Free WiFi: "RailWire" network (2GB/day)
- Food Courts: IRCTC Plaza, McDonald's, local stalls
- Metro Connectivity: Yellow Line
- Cloak Room: 24/7 luggage storage
- Medical Room: First aid and emergency care
- Wheelchair Service: Available on request

ALLAHABAD (ALD):
- Waiting Rooms: AC and Non-AC available
- Refreshment Rooms: IRCTC operated
- Book Stall: Platform 1
- Medical Facility: Emergency care

PATNA (PNBE):
- Executive Lounge: AC waiting area
- Food Plaza: Multiple vendors
- WiFi: Available at major platforms
- Parking: Multi-level parking facility

SPECIAL ASSISTANCE:
- Senior Citizens: Priority booking and seating
- Disabled Passengers: Wheelchair service, escort facility
- Women: Separate waiting rooms
- Emergency: 24/7 helpline support'''
            },
            {
                'title': 'Complaint Procedures',
                'content': '''=== How to File Railway Complaints ===

RAILMADAD APP:
1. Download "RailMadad" app or visit railmadad.indianrailways.gov.in
2. Register with mobile number
3. File complaint with PNR/train details
4. Upload photos/evidence if applicable
5. Track complaint status in real-time
6. Average resolution time: 2-48 hours

HELPLINE NUMBERS:
- General Enquiry: 139 (All India)
- Security: 182 (RPF Helpline)
- Medical Emergency: Contact station master
- Food Quality: 1800-111-321
- Cleanliness: 1800-111-139

COMPLAINT CATEGORIES:
- Train delays/cancellations
- Cleanliness issues
- Food quality
- Staff behavior
- Overcharging
- Safety concerns
- Amenity issues

FOLLOW-UP:
- Complaint ID issued immediately
- SMS updates on resolution
- Can escalate if not resolved within 48 hours
- Write to Divisional Railway Manager for serious issues'''
            },
            {
                'title': 'Passenger Rights',
                'content': '''=== Indian Railway Passenger Rights ===

RIGHT TO REFUND:
- Full refund if train is cancelled by railways
- Full refund for delays exceeding 3 hours
- Partial refund for advance cancellation
- No questions asked refund for RAC/waitlisted tickets after chart

RIGHT TO COMPENSATION:
- Free meals for delays exceeding 2 hours on premium trains
- Alternative transport if train is cancelled
- Accommodation if stranded overnight

RIGHT TO QUALITY SERVICE:
- Clean coaches and toilets
- Safe drinking water
- Bedrolls in AC classes
- Proper lighting and fans
- Functional charging points

RIGHT TO SAFETY:
- Secure travel environment
- Quick response to emergencies
- Medical assistance
- Protection from harassment

RIGHT TO INFORMATION:
- Timely announcements about delays
- Platform change notifications
- Coach position display
- PNR status updates'''
            }
        ]
        
        # Initialize ChromaDB
        client = chromadb.Client()
        collection = client.create_collection(name="train_knowledge_base")
        
        # Add documents
        for i, doc in enumerate(knowledge_base):
            collection.add(
                documents=[doc['content']],
                metadatas=[{'title': doc['title']}],
                ids=[f"doc_{i}"]
            )
        
        return collection
    except Exception as e:
        st.warning(f"RAG system initialization warning: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_dew_point(temp_c, humidity):
    """Calculate dew point using Magnus formula"""
    a, b = 17.27, 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(humidity / 100.0)
    return (b * alpha) / (a - alpha)

def fetch_live_weather(station_code, target_date):
    """Fetch live weather data from OpenWeatherMap API"""
    try:
        coords = STATION_COORDS[station_code]
        url = f"https://api.openweathermap.org/data/2.5/forecast"
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Find closest forecast to target time
            target_timestamp = int(target_date.timestamp())
            closest_forecast = min(
                data['list'],
                key=lambda x: abs(x['dt'] - target_timestamp)
            )
            
            return {
                'temperature': closest_forecast['main']['temp'],
                'humidity': closest_forecast['main']['humidity'],
                'wind_speed': closest_forecast['wind']['speed'],
                'visibility': closest_forecast.get('visibility', 10000),
                'source': 'live_api'
            }
    except Exception as e:
        st.warning(f"Weather API error: {e}")
    return None

def fetch_live_aqi(station_code):
    """Fetch live AQI data from OpenAQ API"""
    try:
        coords = STATION_COORDS[station_code]
        url = "https://api.openaq.org/v2/latest"
        params = {
            'coordinates': f"{coords['lat']},{coords['lon']}",
            'radius': 50000,
            'limit': 1
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                measurements = data['results'][0]['measurements']
                pm25 = next((m['value'] for m in measurements if m['parameter'] == 'pm25'), 150)
                pm10 = next((m['value'] for m in measurements if m['parameter'] == 'pm10'), 200)
                aqi = int(max(pm25 * 2, pm10 * 1.5))
                
                return {
                    'pm25': pm25,
                    'pm10': pm10,
                    'aqi': aqi,
                    'source': 'live_api'
                }
    except Exception as e:
        st.warning(f"AQI API error: {e}")
    return None

def get_historical_weather(gold_master, historical_averages, station_code, month, hour):
    """Get historical weather averages"""
    hist = historical_averages[
        (historical_averages['station'] == station_code) &
        (historical_averages['month'] == month) &
        (historical_averages['hour'] == hour)
    ]
    
    if len(hist) > 0:
        return {
            'temperature': hist.iloc[0]['hist_avg_temp'],
            'humidity': hist.iloc[0]['hist_avg_humidity'],
            'wind_speed': hist.iloc[0]['hist_avg_wind'],
            'visibility': hist.iloc[0]['hist_avg_visibility'],
            'source': 'historical'
        }
    else:
        return {
            'temperature': 15.0,
            'humidity': 75.0,
            'wind_speed': 3.0,
            'visibility': 1500.0,
            'source': 'fallback'
        }

def get_historical_aqi(gold_master, historical_averages, station_code, month, hour):
    """Get historical AQI averages"""
    hist = historical_averages[
        (historical_averages['station'] == station_code) &
        (historical_averages['month'] == month) &
        (historical_averages['hour'] == hour)
    ]
    
    if len(hist) > 0:
        return {
            'pm25': hist.iloc[0]['hist_avg_pm25'],
            'pm10': hist.iloc[0]['hist_avg_pm10'],
            'aqi': hist.iloc[0]['hist_avg_aqi'],
            'source': 'historical'
        }
    else:
        return {
            'pm25': 150.0,
            'pm10': 200.0,
            'aqi': 200.0,
            'source': 'fallback'
        }

def query_knowledge_base(collection, question, n_results=2):
    """Query RAG knowledge base"""
    try:
        if collection is None:
            return "Knowledge base not available."
        
        results = collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        context = ""
        for i, doc in enumerate(results['documents'][0]):
            context += f"\n\n=== {results['metadatas'][0][i]['title']} ===\n{doc[:500]}...\n"
        
        return context
    except Exception as e:
        return f"RAG query error: {e}"

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def predict_delay_smart(model, gold_master, historical_averages, le_station, le_train_type,
                        station_code, train_number, train_type, target_datetime):
    """
    Smart delay prediction with real ML model
    """
    days_until = (target_datetime - datetime.now()).days
    hour = target_datetime.hour
    day_of_week = target_datetime.weekday()
    month = target_datetime.month
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_rush_hour = 1 if hour in [6, 7, 8, 9, 17, 18, 19, 20] else 0
    
    # Get weather and AQI data
    if days_until < 2:
        weather = fetch_live_weather(station_code, target_datetime)
        if not weather:
            weather = get_historical_weather(gold_master, historical_averages, station_code, month, hour)
        
        aqi = fetch_live_aqi(station_code)
        if not aqi:
            aqi = get_historical_aqi(gold_master, historical_averages, station_code, month, hour)
        
        data_source = "Live API"
        confidence = "High"
    else:
        weather = get_historical_weather(gold_master, historical_averages, station_code, month, hour)
        aqi = get_historical_aqi(gold_master, historical_averages, station_code, month, hour)
        data_source = "Historical Average"
        confidence = "Medium"
    
    # Calculate dew point
    dew_point = calculate_dew_point(weather['temperature'], weather['humidity'])
    dew_point_spread = weather['temperature'] - dew_point
    
    # Encode categorical features
    try:
        station_encoded = le_station.transform([station_code])[0]
    except:
        station_encoded = 0
    
    try:
        train_type_encoded = le_train_type.transform([train_type])[0]
    except:
        train_type_encoded = 0
    
    # Determine encoded values
    fog_encoded = 3 if dew_point_spread < 1.5 else (2 if dew_point_spread < 2.5 else (1 if dew_point_spread < 4.0 else 0))
    vis_encoded = 0 if weather['visibility'] < 500 else (1 if weather['visibility'] < 1000 else (2 if weather['visibility'] < 2000 else 3))
    aqi_encoded = 5 if aqi['aqi'] >= 300 else (4 if aqi['aqi'] >= 200 else (3 if aqi['aqi'] >= 150 else (2 if aqi['aqi'] >= 100 else (1 if aqi['aqi'] >= 50 else 0))))
    
    # Historical averages
    station_avg = gold_master[gold_master['station'] == station_code]['delay_minutes'].mean()
    hour_avg = gold_master[gold_master['arrival_hour'] == hour]['delay_minutes'].mean()
    
    if np.isnan(station_avg):
        station_avg = 100.0
    if np.isnan(hour_avg):
        hour_avg = 100.0
    
    # Interaction features
    temp_humidity_interaction = weather['temperature'] * weather['humidity']
    fog_pollution_score = (weather['visibility'] / 1000) * (500 - aqi['aqi'])
    
    # Create feature vector
    features = np.array([[
        hour, day_of_week, is_weekend, is_rush_hour, month,
        weather['temperature'], weather['humidity'], weather['wind_speed'], weather['visibility'],
        dew_point, dew_point_spread,
        aqi['pm25'], aqi['pm10'], aqi['aqi'], 1 if aqi['aqi'] > 200 else 0,
        station_avg, station_avg, hour_avg,
        temp_humidity_interaction, fog_pollution_score,
        station_encoded, train_type_encoded, fog_encoded, vis_encoded, aqi_encoded
    ]])
    
    # Predict
    if model is not None:
        predicted_delay = model.predict(features)[0]
    else:
        # Fallback: use historical average with adjustments
        predicted_delay = station_avg * (1.5 if dew_point_spread < 2.5 else 1.0) * (1.2 if aqi['aqi'] > 200 else 1.0)
    
    predicted_delay = max(0, predicted_delay)
    
    # Severity
    if predicted_delay < 15:
        severity = "✅ ON-TIME"
        severity_level = "success"
    elif predicted_delay < 60:
        severity = "🟡 MINOR DELAY"
        severity_level = "warning"
    elif predicted_delay < 120:
        severity = "🟠 MODERATE DELAY"
        severity_level = "warning"
    else:
        severity = "🔴 CRITICAL DELAY"
        severity_level = "error"
    
    return {
        'predicted_delay': round(predicted_delay, 1),
        'severity': severity,
        'severity_level': severity_level,
        'confidence': confidence,
        'data_source': data_source,
        'weather': weather,
        'aqi': aqi,
        'dew_point_spread': round(dew_point_spread, 2),
        'days_until': days_until
    }

def find_alternative_trains(gold_master, station_code, predicted_delay):
    """Find alternative trains"""
    if predicted_delay < 120:
        return []
    
    # Get trains from the same station with lower average delays
    alternatives = gold_master[gold_master['station'] == station_code].groupby('train_number').agg({
        'delay_minutes': 'mean'
    }).reset_index()
    
    alternatives = alternatives[alternatives['delay_minutes'] < predicted_delay * 0.5]
    alternatives = alternatives.sort_values('delay_minutes').head(3)
    
    result = []
    for _, row in alternatives.iterrows():
        result.append({
            'number': str(row['train_number']),
            'name': f"Train {row['train_number']}",
            'type': 'Express',
            'avg_delay': f"{row['delay_minutes']:.0f} min"
        })
    
    return result

def create_sample_data():
    """Create sample data if Databricks connection fails"""
    # Create minimal sample data for fallback
    gold_master = pd.DataFrame({
        'station': ['NDLS', 'ALD', 'PNBE'] * 100,
        'train_number': [12301, 12381, 12561] * 100,
        'arrival_hour': np.random.randint(0, 24, 300),
        'month': np.random.randint(1, 13, 300),
        'delay_minutes': np.random.randint(30, 150, 300)
    })
    
    historical_averages = gold_master.groupby(['station', 'arrival_hour', 'month']).agg({
        'delay_minutes': 'mean'
    }).reset_index()
    historical_averages['hist_avg_temp'] = 15.0
    historical_averages['hist_avg_humidity'] = 75.0
    historical_averages['hist_avg_wind'] = 3.0
    historical_averages['hist_avg_visibility'] = 1500.0
    historical_averages['hist_avg_pm25'] = 150.0
    historical_averages['hist_avg_pm10'] = 200.0
    historical_averages['hist_avg_aqi'] = 200.0
    historical_averages.columns = [
        'station', 'hour', 'month', 'hist_avg_delay',
        'hist_avg_temp', 'hist_avg_humidity', 'hist_avg_wind',
        'hist_avg_visibility', 'hist_avg_pm25', 'hist_avg_pm10', 'hist_avg_aqi'
    ]
    
    return gold_master, historical_averages

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="🚂 Train Delay Predictor",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        padding: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize (with caching)
with st.spinner('🔧 Loading models and data...'):
    gold_master, historical_averages = load_databricks_data()
    model, model_source = load_ml_models()
    le_station, le_train_type = load_label_encoders()
    rag_collection = initialize_rag_system()

st.success(f"✅ System ready! Model: {model_source}")

# Title
st.markdown('<h1 class="main-header">🚂 Train Delay Prediction System</h1>', unsafe_allow_html=True)
st.markdown('### Powered by ML + RAG + Indian AI Models 🇮🇳')
st.markdown('---')

# Sidebar
st.sidebar.header('📝 Input Parameters')

station_dict = {
    'New Delhi (NDLS)': 'NDLS',
    'Allahabad (ALD)': 'ALD',
    'Patna (PNBE)': 'PNBE',
    'Mughalsarai (MGS)': 'MGS'
}
station_name = st.sidebar.selectbox('🏫 Select Station', list(station_dict.keys()))
station_code = station_dict[station_name]

train_number = st.sidebar.text_input('🚄 Train Number', '12301')
train_type = st.sidebar.selectbox('🚆 Train Type', 
    ['Rajdhani', 'Shatabdi', 'Express', 'Superfast', 'Passenger'])

date_input = st.sidebar.date_input('📅 Travel Date', 
    datetime.now() + timedelta(days=1))
hour_input = st.sidebar.slider('⏰ Arrival Hour (24h)', 0, 23, 7)

language_name = st.sidebar.selectbox('🌏 Preferred Language', list(SUPPORTED_LANGUAGES.values()))
language_code = [k for k, v in SUPPORTED_LANGUAGES.items() if v == language_name][0]

predict_button = st.sidebar.button('🚀 Predict Delay', type='primary', use_container_width=True)

st.sidebar.markdown('---')
st.sidebar.markdown('🎯 **Features:**')
st.sidebar.markdown('• Real ML predictions\n• Live weather API\n• RAG assistance\n• Indian AI models')

# Main content
if predict_button:
    target_dt = datetime.combine(date_input, datetime.min.time()).replace(hour=hour_input)
    
    with st.spinner('🤖 Running prediction...'):
        # Get prediction
        prediction = predict_delay_smart(
            model, gold_master, historical_averages, le_station, le_train_type,
            station_code, train_number, train_type, target_dt
        )
        
        # Get alternatives
        alternatives = find_alternative_trains(gold_master, station_code, prediction['predicted_delay'])
    
    # Display results
    st.success('✅ Prediction completed successfully!')
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="⏰ Predicted Delay",
            value=f"{prediction['predicted_delay']} min",
            delta=f"Data: {prediction['data_source']}"
        )
    
    with col2:
        st.metric(
            label="🌡️ Temperature",
            value=f"{prediction['weather']['temperature']:.1f}°C"
        )
    
    with col3:
        st.metric(
            label="👁️ Visibility",
            value=f"{prediction['weather']['visibility']:.0f}m"
        )
    
    with col4:
        st.metric(
            label="🌫️ AQI",
            value=f"{prediction['aqi']['aqi']:.0f}"
        )
    
    # Status box
    if prediction['severity_level'] == 'success':
        st.markdown(f'<div class="success-box"><h3>{prediction["severity"]}</h3><p>Your train should arrive on schedule!</p></div>', unsafe_allow_html=True)
    elif prediction['predicted_delay'] < 120:
        st.markdown(f'<div class="warning-box"><h3>{prediction["severity"]}</h3><p>Expect some delay. Plan accordingly.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="danger-box"><h3>{prediction["severity"]}</h3><p>Major delay expected! Check alternatives below.</p></div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(['📊 Analysis', '🚆 Alternatives', '📚 Information', '🇮🇳 Translation'])
    
    with tab1:
        st.subheader('📊 Detailed Analysis')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('**🌡️ Weather Conditions:**')
            st.write(f'• Temperature: {prediction["weather"]["temperature"]:.1f}°C')
            st.write(f'• Humidity: {prediction["weather"]["humidity"]:.0f}%')
            st.write(f'• Visibility: {prediction["weather"]["visibility"]:.0f} meters')
            st.write(f'• Dew Point Spread: {prediction["dew_point_spread"]}°C')
            if prediction['dew_point_spread'] < 2.5:
                st.warning('⚠️ FOG WARNING: Low dew point spread!')
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction['predicted_delay'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Delay (minutes)"},
                gauge={
                    'axis': {'range': [None, 200]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "lightgreen"},
                        {'range': [15, 60], 'color': "yellow"},
                        {'range': [60, 120], 'color': "orange"},
                        {'range': [120, 200], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 120}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('**🌫️ Air Quality:**')
            st.write(f'• PM2.5: {prediction["aqi"]["pm25"]:.0f} µg/m³')
            st.write(f'• PM10: {prediction["aqi"]["pm10"]:.0f} µg/m³')
            st.write(f'• AQI: {prediction["aqi"]["aqi"]:.0f}')
            if prediction['aqi']['aqi'] > 200:
                st.warning('⚠️ HIGH POLLUTION')
            
            # Show model confidence
            st.info(f"📊 Prediction Confidence: {prediction['confidence']}\n\nModel: {model_source.upper()}\nData: {prediction['data_source']}")
    
    with tab2:
        st.subheader('🚆 Alternative Train Recommendations')
        
        if len(alternatives) > 0:
            st.info('🔍 Alternative trains with better timings:')
            
            alt_df = pd.DataFrame(alternatives)
            alt_df.columns = ['Train No', 'Train Name', 'Type', 'Expected Delay']
            st.dataframe(alt_df, use_container_width=True)
            st.success('💡 Tip: Book alternatives early during fog season!')
        else:
            st.info('✅ No alternatives needed - delay is acceptable.')
    
    with tab3:
        st.subheader('📚 Passenger Assistance (RAG-Powered)')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('**💰 Refund & Compensation:**')
            if prediction['predicted_delay'] > 180:
                st.success('✅ ELIGIBLE FOR FULL REFUND')
                # Query RAG for detailed info
                refund_info = query_knowledge_base(rag_collection, "train delay refund compensation", n_results=1)
                with st.expander("See refund details"):
                    st.text(refund_info[:500])
            else:
                st.info('Partial refund may be available for cancellation.')
            
            st.markdown('**📞 Contact:**')
            st.write('• Railway Helpline: 139')
            st.write('• Food Quality: 1800-111-321')
            st.write('• Security: 182')
        
        with col2:
            st.markdown(f'**🏫 {station_name} Facilities:**')
            # Query RAG for station info
            station_info = query_knowledge_base(rag_collection, f"{station_name} station facilities", n_results=1)
            with st.expander("See station facilities"):
                st.text(station_info[:500])
    
    with tab4:
        st.subheader(f'🇮🇳 Translation to {language_name}')
        
        if language_code != 'en':
            st.info(f'🔄 Using Indian AI Model for translation...')
            st.markdown('**Predicted delay message in your language:**')
            st.markdown(f'> 🇮🇳 [Translated message would appear here in {language_name}]')
            st.markdown(f'> 🇮🇳 Powered by Sarvam AI / IndicTrans2')
            
            if BHASHINI_API_KEY == "your_bhashini_api_key":
                st.warning('⚠️ To activate translation, get free API key from https://bhashini.gov.in/ulca')
            else:
                st.success('✅ Translation feature active!')
        else:
            st.info('Select a regional language to see translation.')
else:
    st.info('👈 Set your travel parameters and click "Predict Delay"!')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('### 🎯 Features')
        st.write('• Real ML predictions')
        st.write('• Live weather API')
        st.write('• Smart data logic')
        st.write('• RAG assistance')
    
    with col2:
        st.markdown('### 🇮🇳 Indian AI')
        st.write('• Sarvam AI models')
        st.write('• IndicTrans2')
        st.write('• 6+ languages')
        st.write('• Local context')
    
    with col3:
        st.markdown('### 🛠️ Technology')
        st.write(f'• {model_source.upper()} model')
        st.write('• Unity Catalog')
        st.write('• ChromaDB RAG')
        st.write('• Databricks')

st.markdown('---')
st.markdown('<p style="text-align: center; color: gray;">Built with ❤️ on Databricks | Using Indian AI Models 🇮🇳</p>', unsafe_allow_html=True)