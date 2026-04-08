import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
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

# Title
st.markdown('<h1 class="main-header">🚂 Train Delay Prediction System</h1>', unsafe_allow_html=True)
st.markdown('### Powered by ML + RAG + Indian AI Models 🇮🇳')
st.markdown('---')

# Sidebar - Input Section
st.sidebar.header('📝 Input Parameters')

# Station selection
station_dict = {
    'New Delhi (NDLS)': 'NDLS',
    'Allahabad (ALD)': 'ALD',
    'Patna (PNBE)': 'PNBE',
    'Mughalsarai (MGS)': 'MGS'
}
station_name = st.sidebar.selectbox('🏫 Select Station', list(station_dict.keys()))
station_code = station_dict[station_name]

# Train details
train_number = st.sidebar.text_input('🚄 Train Number', '12301')
train_type = st.sidebar.selectbox('🚆 Train Type', 
    ['Rajdhani', 'Shatabdi', 'Express', 'Superfast', 'Passenger'])

# Date and time
date_input = st.sidebar.date_input('📅 Travel Date', 
    datetime.now() + timedelta(days=1))
hour_input = st.sidebar.slider('⏰ Arrival Hour (24h)', 0, 23, 7)

# Language preference
language_map = {
    'English': 'en',
    'Hindi (हिंदी)': 'hi',
    'Bengali (বাংলা)': 'bn',
    'Telugu (తెలుగు)': 'te',
    'Marathi (मराठी)': 'mr',
    'Tamil (தமிழ்)': 'ta'
}
language_name = st.sidebar.selectbox('🌏 Preferred Language', list(language_map.keys()))
language_code = language_map[language_name]

# Predict button
predict_button = st.sidebar.button('🚀 Predict Delay', type='primary', use_container_width=True)

st.sidebar.markdown('---')
st.sidebar.markdown('🎯 **Features:**')
st.sidebar.markdown('• Real-time weather API\n• ML predictions\n• RAG assistance\n• Indian AI models')

# Main content area
if predict_button:
    # Create target datetime
    target_dt = datetime.combine(date_input, datetime.min.time()).replace(hour=hour_input)
    days_ahead = (target_dt - datetime.now()).days
    
    with st.spinner('🤖 Running prediction...'):
        # Simulate prediction (replace with actual model call)
        import time
        time.sleep(1)
        
        # Mock results (replace with actual prediction from your ML model)
        predicted_delay = np.random.randint(15, 180)
        temperature = np.random.uniform(10, 25)
        humidity = np.random.uniform(60, 95)
        visibility = np.random.uniform(500, 3000)
        aqi = np.random.randint(100, 300)
        
        # Display results
        st.success('✅ Prediction completed successfully!')
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="⏰ Predicted Delay",
                value=f"{predicted_delay} min",
                delta=f"{predicted_delay - 30} vs avg" if predicted_delay > 30 else "Better than avg"
            )
        
        with col2:
            st.metric(
                label="🌡️ Temperature",
                value=f"{temperature:.1f}°C"
            )
        
        with col3:
            st.metric(
                label="👁️ Visibility",
                value=f"{visibility:.0f}m"
            )
        
        with col4:
            st.metric(
                label="🌫️ AQI",
                value=f"{aqi}"
            )
        
        # Status box
        if predicted_delay < 15:
            st.markdown('<div class="success-box"><h3>✅ ON TIME</h3><p>Minimal delay expected. Your train should arrive on schedule!</p></div>', unsafe_allow_html=True)
        elif predicted_delay < 60:
            st.markdown('<div class="warning-box"><h3>🟡 MINOR DELAY</h3><p>Slight delay expected. Plan for extra 30-60 minutes.</p></div>', unsafe_allow_html=True)
        elif predicted_delay < 120:
            st.markdown('<div class="warning-box"><h3>🟠 MODERATE DELAY</h3><p>Significant delay expected. Consider checking alternatives.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="danger-box"><h3>🔴 CRITICAL DELAY</h3><p>Major delay expected! Please check alternative trains below.</p></div>', unsafe_allow_html=True)
        
        # Tabs for detailed information
        tab1, tab2, tab3, tab4 = st.tabs(['📊 Analysis', '🚆 Alternatives', '📚 Information', '🇮🇳 Translation'])
        
        with tab1:
            st.subheader('📊 Detailed Analysis')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('**🌡️ Weather Conditions:**')
                st.write(f'• Temperature: {temperature:.1f}°C')
                st.write(f'• Humidity: {humidity:.0f}%')
                st.write(f'• Visibility: {visibility:.0f} meters')
                if visibility < 1000:
                    st.warning('⚠️ FOG WARNING: Low visibility!')
                
                # Gauge chart for delay
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_delay,
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
                st.write(f'• PM2.5: {aqi * 0.5:.0f} µg/m³')
                st.write(f'• PM10: {aqi * 0.8:.0f} µg/m³')
                st.write(f'• AQI: {aqi}')
                if aqi > 200:
                    st.warning('⚠️ HIGH POLLUTION')
                
                # Historical trend (mock data)
                trend_df = pd.DataFrame({
                    'Hour': range(0, 24),
                    'Avg Delay': [np.random.randint(20, 100) for _ in range(24)]
                })
                fig2 = px.line(trend_df, x='Hour', y='Avg Delay', 
                              title='Historical Delay Pattern',
                              markers=True)
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader('🚆 Alternative Train Recommendations')
            
            if predicted_delay > 120:
                st.info('🔍 Finding alternative trains with better timings...')
                
                # Mock alternative trains
                alternatives_df = pd.DataFrame({
                    'Train No': ['12302', '12561', '12303'],
                    'Train Name': ['Howrah Rajdhani', 'Swatantrata Senani Exp', 'Poorva Express'],
                    'Departure': ['08:30', '10:15', '12:00'],
                    'Expected Delay': ['25 min', '45 min', '30 min'],
                    'Status': ['✅ Available', '✅ Available', '⚠️ Limited']
                })
                
                st.dataframe(alternatives_df, use_container_width=True)
                st.success('💡 Tip: Book alternative trains early during fog season!')
            else:
                st.info('✅ No alternatives needed - delay is acceptable.')
        
        with tab3:
            st.subheader('📚 Passenger Assistance (RAG-Powered)')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('**💰 Refund & Compensation:**')
                if predicted_delay > 180:
                    st.success('✅ ELIGIBLE FOR FULL REFUND')
                    st.write('• Delays > 3 hours qualify')
                    st.write('• Submit TDR within 72 hours')
                    st.write('• No cancellation charges')
                    st.write('• Refund in 7-10 days')
                else:
                    st.info('Partial refund may be available for cancellation.')
                
                st.markdown('**📞 Contact:**')
                st.write('• Railway Helpline: 139')
                st.write('• Food Quality: 1800-111-321')
                st.write('• Security: 182')
            
            with col2:
                st.markdown(f'**🏫 {station_name} Facilities:**')
                st.write('• Executive Lounge')
                st.write('• Free WiFi')
                st.write('• Food Courts')
                st.write('• Waiting Rooms (AC/Non-AC)')
                st.write('• Medical Facility')
                st.write('• Wheelchair Access')
                
                st.markdown('**♿ Special Assistance:**')
                st.write('• Senior citizens priority')
                st.write('• Wheelchair booking')
                st.write('• Medical emergency support')
        
        with tab4:
            st.subheader(f'🇮🇳 Translation to {language_name}')
            
            if language_code != 'en':
                st.info(f'🔄 Using Indian AI Model for translation...')
                st.markdown('**Predicted delay message in your language:**')
                st.markdown(f'> 🇮🇳 [Translated message would appear here in {language_name}]')
                st.markdown(f'> 🇮🇳 Powered by Sarvam AI / IndicTrans2')
                
                st.success('✅ Translation feature demonstrates use of Indian-built AI models!')
            else:
                st.info('Select a regional language from sidebar to see translation.')
else:
    # Welcome screen
    st.info('👈 Set your travel parameters in the sidebar and click "Predict Delay" to get started!')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('### 🎯 Features')
        st.write('• Real-time predictions')
        st.write('• Weather integration')
        st.write('• ML-powered accuracy')
        st.write('• RAG assistance')
    
    with col2:
        st.markdown('### 🇮🇳 Indian AI')
        st.write('• Sarvam AI models')
        st.write('• IndicTrans2 translation')
        st.write('• 10+ languages')
        st.write('• Local context')
    
    with col3:
        st.markdown('### 🛠️ Technology')
        st.write('• Random Forest + XGBoost')
        st.write('• ChromaDB RAG')
        st.write('• OpenWeatherMap API')
        st.write('• Databricks Platform')

# Footer
st.markdown('---')
st.markdown('<p style="text-align: center; color: gray;">Built with ❤️ on Databricks | Using Indian AI Models 🇮🇳</p>', unsafe_allow_html=True)