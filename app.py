import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
import numpy as np
import boto3  # AWS SDK for Python
import io     # For handling data streams

S3_BUCKET_NAME = 'your-fire-data-bucket'  # <<< REPLACE with your actual S3 bucket name
S3_FILE_KEY = 'historical_fire_data.csv'  # <<< Ensure this file exists in your bucket

st.set_page_config(
    page_title="Agni-Rakshak Fire Forecast",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data_from_s3():
    """Fetches historical fire data from an AWS S3 bucket or falls back to sample data."""
    st.sidebar.info(f"Attempting to load training data from S3...")
    
    # Fallback/Sample Data (Used if S3 fails or is not configured)
    sample_data = pd.DataFrame({
        'temperature': [35, 28, 40, 25, 30], 'humidity': [20, 60, 15, 70, 50],
        'wind_speed': [15, 5, 20, 3, 10], 'slope': [30, 10, 45, 5, 20],
        'vegetation_type': [1, 2, 1, 3, 2], 'fire_occurred': [1, 0, 1, 0, 0]
    })
    
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=S3_FILE_KEY)
        data = pd.read_csv(io.BytesIO(obj['Body'].read()))
        st.sidebar.success(f"✅ Loaded {len(data)} records from S3.")
        
        # Simple data validation: ensure required columns exist
        required_cols = ['temperature', 'humidity', 'wind_speed', 'slope', 'vegetation_type', 'fire_occurred']
        if not all(col in data.columns for col in required_cols):
             st.sidebar.error(f"S3 Data is missing required columns. Using sample data.")
             return sample_data
             
        return data

    except Exception as e:
        st.sidebar.error(f"⚠️ S3 Connection Error: Using Sample Data for demo.")
        st.sidebar.caption(f"Details: {e}")
        return sample_data

with st.sidebar:
    st.header("⚙️ System Controls")
    
    data = load_data_from_s3() # Load data here
    
    forecast_date = st.date_input("Forecast Date", value="today")
    region = st.selectbox("Target Region", ["Northern India (Demo)", "Western Ghats", "Custom"])
    
    st.markdown("---")
    st.subheader("Model Details")
    st.info(f"Training Data Size: {len(data)} records.")
    st.caption("Model: Random Forest Classifier (Trained)")
    st.metric("Risk Threshold", "50%", help="Probability score above which an area is classified as HIGH risk.")
    st.markdown("---")

from sklearn.ensemble import RandomForestClassifier
X = data.drop('fire_occurred', axis=1)
y = data['fire_occurred']

if len(data) > 1:
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    new_data = pd.DataFrame({
        'temperature': [33, 26, 42, 29], # Added two more points for better demo
        'humidity': [25, 65, 12, 55],
        'wind_speed': [12, 4, 18, 8],
        'slope': [35, 8, 50, 15],
        'vegetation_type': [1, 2, 1, 3]
    })
    risk_proba = model.predict_proba(new_data)[:, 1]
    locations = [(26.8467, 80.9462), (30.7, 76.8), (22.25, 80.5), (25.5, 75.0)] # 4 points

else:
    # Fallback if training data is insufficient
    st.error("Cannot train model: Insufficient historical data.")
    risk_proba = np.array([0.0, 0.0])
    locations = [(0, 0)] # Dummy location

st.title("🔥 Agni-Rakshak: AI-Powered Forest Fire Risk Forecast")
st.markdown(f"### Real-Time Risk Assessment for **{region}** on **{forecast_date}**")
st.divider()

st.subheader("Dashboard Summary")
col1, col2, col3, col4 = st.columns(4)

total_locations = len(risk_proba)
avg_risk = risk_proba.mean() * 100 if total_locations > 0 else 0
high_risk_count = (risk_proba >= 0.5).sum()
max_risk = risk_proba.max() * 100 if total_locations > 0 else 0

col1.metric("Total Locations Analyzed", total_locations)
col2.metric("High-Risk Zones (≥50%)", high_risk_count, delta=f"{total_locations - high_risk_count} Low-Risk")
col3.metric("Average Risk Score", f"{avg_risk:.2f}%", delta_color="off")
col4.metric("Maximum Predicted Risk", f"{max_risk:.2f}%")

st.markdown("---")
st.subheader("Interactive Forecast Map")
st.markdown("🔴 **High Risk** (Probability ≥ 50%) | 🟢 **Low Risk** (Probability < 50%)")

if locations and locations[0] != (0, 0):
    center_lat = np.mean([loc[0] for loc in locations])
    center_lon = np.mean([loc[1] for loc in locations])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB dark_matter")
else:
    m = folium.Map(location=[28.8, 78.5], zoom_start=5, tiles="CartoDB dark_matter")

for i, loc in enumerate(locations):
    if i < len(risk_proba): # Check bounds
        probability_percent = f"{risk_proba[i] * 100:.2f}%"
        risk_level = "High" if risk_proba[i] >= 0.5 else "Low"
        color = "red" if risk_level == "High" else "green"
        icon_name = 'fire' if risk_level == 'High' else 'cloud'
        
        folium.Marker(
            location=loc,
            popup=f"**Risk:** {risk_level}<br>**Probability:** {probability_percent}",
            icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
        ).add_to(m)

st_folium(m, width=1200, height=600)

st.markdown("---")
st.subheader("Raw Prediction Data (for detailed inspection)")

results_df = new_data.copy()
results_df['Latitude'] = [loc[0] for loc in locations]
results_df['Longitude'] = [loc[1] for loc in locations]
results_df['Risk Probability (%)'] = risk_proba * 100
results_df['Risk Level'] = results_df['Risk Probability (%)'].apply(lambda x: 'HIGH' if x >= 50 else 'LOW')

cols = ['Latitude', 'Longitude', 'Risk Level', 'Risk Probability (%)'] + X.columns.tolist()
results_df = results_df[cols]

st.dataframe(
    results_df.style.background_gradient(cmap='Reds', subset=['Risk Probability (%)']), 
    use_container_width=True
)