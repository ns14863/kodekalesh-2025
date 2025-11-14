AGNI-RAKSHAK: AI-POWERED FOREST FIRE FORECASTING SYSTEM
Project Status: Functional Prototype (MVP) - Streamlit 

---PROJECT SUMMARY

Agni-Rakshak is a comprehensive, cloud-native application designed to provide
real-time risk forecasting for forest fires. It integrates geospatial,
meteorological, and terrain data to generate actionable risk probabilities.

The solution is built around a robust Machine Learning model served via a
Streamlit dashboard, featuring secure data sourcing via AWS S3.

2. TECHNOLOGY STACK

Frontend/App: Streamlit (Interactive dashboard and UI presentation)

Machine Learning: scikit-learn (RandomForestClassifier) (Core prediction engine)

Data Visualization: Folium, streamlit-folium (Renders dynamic, interactive risk maps)

Cloud Integration: AWS Boto3 (Securely connects to Amazon S3 - AWS Builder Pattern)

Data Handling: Pandas, NumPy (Data cleaning and statistical processing)

3. KEY FEATURES

Risk Probability Score: Provides a precise percentage risk rather than a simple binary (Yes/No) prediction.

AWS Builder Integration: Securely loads historical training data from an Amazon S3 bucket using Boto3 and environment variables.

Interactive Geospatial Visualization: Displays predicted high-risk zones (Red) and low-risk zones (Green) on a customizable map interface.

Intuitive Dashboard: Presents key metrics (Average Risk, High-Risk Count) and raw prediction data for complete transparency.

4. SETUP AND EXECUTION GUIDE

4.1. Prerequisites

You must have Python 3.x and the required libraries installed:
pip install streamlit streamlit-folium pandas numpy scikit-learn boto3

4.2. AWS Configuration (CRITICAL STEP)

The app requires temporary AWS credentials to connect to S3. You must run the following commands in your terminal before running the Streamlit app:

# Example for Linux/macOS
export AWS_DEFAULT_REGION=us-west-2
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...


4.3. Data Setup

S3 Bucket: Upload your historical training data (historical_fire_data.csv) to your AWS S3 bucket.

Code Configuration: Open app.py and update the S3 configuration variables:
S3_BUCKET_NAME = 'your-fire-data-bucket'
S3_FILE_KEY = 'historical_fire_data.csv'

4.4. Run the Application

Execute the Python script using Streamlit:
streamlit run app.py

5. FUTURE IMPROVEMENTS

Real-Time API: Integrate with an external weather API (e.g., IMD, OpenWeatherMap) to fetch new data dynamically.

GeoPandas Utilization: Fully implement geopandas to load and display forest boundary shapefiles and predict risk across continuous regions.

Feature Importance: Calculate and visualize the feature importance to explain why the model made a specific prediction.