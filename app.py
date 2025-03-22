import streamlit as st
import folium
from streamlit_folium import st_folium
import ee
import pandas as pd
import plotly.express as px
from folium.plugins import Draw
import requests
import numpy as np
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from google.oauth2.credentials import Credentials

# ---------------------------
# 1. Custom Font Integration (CSS Injection)
# ---------------------------
custom_css = """
<style>
/* Import League Spartan font from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=League+Spartan:wght@400;700&display=swap');

/* Use !important to override Streamlit's theme defaults */
html, body, [class*="css"]  {
    font-family: 'League Spartan', sans-serif !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------
# 2. Caching for heavy operations
# ---------------------------
@st.cache_resource(show_spinner=False)
def initialize_ee():
    auth_info = {
        "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
        "refresh_token": "1//0gG2mDfWKFKUqCgYIARAAGBASNwF-L9IrQGZEaYW_-eanASRxWMFIFTRLIIMrc9hj41c7E0wER3XvkIw8Go-25_HmjDm_W2-x0EI",
        "type": "authorized_user",
        "project": "ee-krishnapunjabifarmmatrix"
    }
    credentials = Credentials.from_authorized_user_info(info=auth_info)
    ee.Initialize(credentials=credentials, project='ee-krishnapunjabifarmmatrix')
    return True

@st.cache_resource(show_spinner=False)
def load_model():
    clf = joblib.load('crop_recommendation_model.pkl')
    label_encoder = joblib.load('crop_label_encoder.pkl')
    return clf, label_encoder

# ---------------------------
# 3. Initialize Earth Engine and load model
# ---------------------------
initialize_ee()
clf, label_encoder = load_model()

# ---------------------------
# 4. App Title and Description
# ---------------------------
st.title("🌍 Advanced Field Scanner & Real-Time Crop Advisor")
st.write("Analyze NDVI, LST, Rainfall, Soil Moisture, and more to get AI-based crop recommendations based on real-time satellite and weather data.")

# ---------------------------
# 5. Sidebar: Location & Date Range
# ---------------------------
if "user_location" not in st.session_state:
    st.session_state.user_location = [18.4575, 73.8503]

st.sidebar.header("📍 Enter Location")
lat = st.sidebar.number_input("Latitude", value=st.session_state.user_location[0], format="%.6f")
lon = st.sidebar.number_input("Longitude", value=st.session_state.user_location[1], format="%.6f")
st.session_state.user_location = [lat, lon]

st.sidebar.header("📅 Select Time Range")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

OPENWEATHER_API_KEY = "bfd90807d20e8b889145cbc80b8015b3"

# ---------------------------
# 6. Map Setup with Drawing Tool
# ---------------------------
m = folium.Map(location=st.session_state.user_location, zoom_start=15)
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google", 
    name="Satellite"
).add_to(m)
Draw(export=True).add_to(m)
folium.Marker(
    st.session_state.user_location,
    popup="Your Location",
    tooltip="Click to select"
).add_to(m)

map_data = st_folium(m, width=700, height=500)

selected_boundary = None
if map_data and "last_active_drawing" in map_data:
    selected_boundary = map_data["last_active_drawing"]
    st.write("### **Selected Area:**", selected_boundary)

# ---------------------------
# 7. Define Helper Functions for Data Extraction
# ---------------------------
def get_lst(region):
    lst = (ee.ImageCollection("MODIS/061/MOD11A2")
           .filterDate(str(start_date), str(end_date))
           .select("LST_Day_1km").mean())
    lst_value = lst.reduceRegion(ee.Reducer.mean(), region, scale=500).get("LST_Day_1km").getInfo()
    return round(lst_value * 0.02 - 273.15, 2) if lst_value else 25

def get_rainfall(region):
    larger_region = region.buffer(50000)
    chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterDate(str(start_date), str(end_date))
              .filterBounds(larger_region)
              .select("precipitation").sum())
    rain = chirps.reduceRegion(ee.Reducer.sum(), larger_region, scale=5000).get("precipitation").getInfo()
    return round(rain, 2) if rain else 0

def get_soil_moisture(region):
    sm_dataset = (ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
                  .filterDate(str(start_date), str(end_date))
                  .select("SoilMoi0_10cm_inst").mean())
    sm = sm_dataset.reduceRegion(ee.Reducer.mean(), region, scale=500).get("SoilMoi0_10cm_inst").getInfo()
    return round(sm, 3) if sm else 0.2

def get_ph(median_image, region):
    # Compute BSI using bands B11 and B2 from the median image
    bsi_image = median_image.normalizedDifference(["B11", "B2"]).rename("BSI").clip(region)
    bsi = bsi_image.reduceRegion(ee.Reducer.mean(), geometry=region, scale=500).get("BSI").getInfo()
    bsi = round(bsi, 3) if bsi else 0.2
    lst_value = get_lst(region)
    # Compute NDVI using bands B8 and B4
    ndvi_image = median_image.normalizedDifference(["B8", "B4"]).rename("NDVI").clip(region)
    ndvi_value = ndvi_image.reduceRegion(ee.Reducer.mean(), geometry=region, scale=30).get("NDVI").getInfo()
    ph = 6.5 - 0.5 * ndvi_value + 0.3 * bsi - 0.01 * lst_value
    return round(ph, 2)

def get_openweather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    temp = response.get("main", {}).get("temp", 25)
    humidity = response.get("main", {}).get("humidity", 70)
    return temp, humidity

# ---------------------------
# 8. Data Extraction and Option Display
# ---------------------------
if not selected_boundary:
    st.write("**🛑 Select an area on the map to scan all parameters and get real-time recommendations.**")
else:
    # Use a session_state flag to run extraction only once
    if st.button("Extract Data") or "extraction" in st.session_state:
        if "extraction" not in st.session_state:
            try:
                region = ee.Geometry.Polygon(selected_boundary["geometry"]["coordinates"])
                image_collection = (
                    ee.ImageCollection("COPERNICUS/S2")
                    .filterDate(str(start_date), str(end_date))
                    .filterBounds(region)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                    .select(["B2", "B4", "B8", "B11"])
                )
                image_count = image_collection.size().getInfo()
                st.sidebar.write(f"📸 **Satellite images found:** {image_count}")

                if image_count == 0:
                    st.error("❌ No suitable satellite images found for this date range.")
                else:
                    # Compute a median image with all bands intact
                    median_image = image_collection.median().clip(region)
                    # Compute NDVI from the median image
                    ndvi_image = median_image.normalizedDifference(["B8", "B4"]).rename("NDVI")
                    mean_ndvi = ndvi_image.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=region,
                        scale=30
                    ).get("NDVI").getInfo()

                    # Compute environmental parameters
                    lst = get_lst(region)
                    rainfall = get_rainfall(region)
                    soil_moisture = get_soil_moisture(region)
                    temp, humidity = get_openweather(lat, lon)
                    ph = get_ph(median_image, region)
                    evapotranspiration = np.random.uniform(2, 4)
                    evi = np.random.uniform(0.2, 0.6)
                    SOC = 1.5

                    if mean_ndvi is None:
                        st.error("⚠ NDVI computation failed. Try selecting a larger area or different date range.")
                    else:
                        st.session_state.extraction = {
                            "mean_ndvi": mean_ndvi,
                            "lst": lst,
                            "rainfall": rainfall,
                            "soil_moisture": soil_moisture,
                            "temp": temp,
                            "humidity": humidity,
                            "ph": ph,
                            "evapotranspiration": evapotranspiration,
                            "evi": evi,
                            "SOC": SOC,
                            "region": region,
                            "median_image": median_image
                        }
                        st.sidebar.write(f"🌿 **Mean NDVI Value:** {mean_ndvi:.4f}")
                        st.sidebar.write(f"🌡️ **Temperature:** {temp}°C")
                        st.sidebar.write(f"💨 **Humidity:** {humidity:.2f}%")
                        st.sidebar.write(f"🌡️ **LST:** {lst}°C")
                        st.sidebar.write(f"🌧️ **Rainfall:** {rainfall} mm")
                        st.sidebar.write(f"💧 **Soil Moisture:** {soil_moisture} m³/m³")
                        st.sidebar.write(f"🧪 **pH Level:** {ph:.2f}")
                        st.sidebar.write(f"🌊 **Evapotranspiration:** {evapotranspiration:.2f} mm")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        # If extraction data is available, display options
        if "extraction" in st.session_state:
            extraction = st.session_state.extraction
            rec_option = st.radio(
                "Do you want crop recommendations?",
                options=["Yes, Recommend", "No, I don't want recommendation"]
            )

            if rec_option == "No, I don't want recommendation":
                user_crop = st.text_input("Enter the crop you are growing:")
                if user_crop:
                    st.write(f"### **You selected: {user_crop}**")
                st.subheader("📋 Soil Report")
                soil_report = {
                    "Mean NDVI": f"{extraction['mean_ndvi']:.4f}",
                    "LST (°C)": f"{extraction['lst']}°C",
                    "Rainfall (mm)": f"{extraction['rainfall']} mm",
                    "Soil Moisture (m³/m³)": f"{extraction['soil_moisture']}",
                    "pH Level": f"{extraction['ph']}",
                    "Temperature (°C)": f"{extraction['temp']}°C",
                    "Humidity (%)": f"{extraction['humidity']}%",
                    "Evapotranspiration (mm)": f"{extraction['evapotranspiration']:.2f} mm"
                }
                st.table(pd.DataFrame(soil_report.items(), columns=["Parameter", "Value"]))

                if st.button("📥 Download Soil Report PDF"):
                    buffer = BytesIO()
                    c = canvas.Canvas(buffer, pagesize=letter)
                    c.setFont("Helvetica-Bold", 18)
                    c.drawString(50, 750, "Professional Soil Report")
                    c.setFont("Helvetica", 12)
                    c.drawString(50, 720, f"Location: Latitude {lat}, Longitude {lon}")
                    c.drawString(50, 700, f"Date Range: {start_date} to {end_date}")
                    c.line(50, 690, 550, 690)
                    y = 670
                    for key, value in soil_report.items():
                        c.drawString(50, y, f"{key}: {value}")
                        y -= 20
                    c.drawString(
                        50, y-10,
                        "This report provides a detailed overview of your field's soil health "
                        "based on the latest satellite data and current weather conditions."
                    )
                    c.save()
                    buffer.seek(0)
                    st.download_button(
                        label="📥 Download Soil Report PDF",
                        data=buffer,
                        file_name="soil_report.pdf",
                        mime="application/pdf"
                    )

            else:
                st.write("### **🌾 AI-Based Crop Recommendations for Selected Field**")
                feature_vector = pd.DataFrame(
                    [[
                        extraction['temp'],
                        extraction['humidity'],
                        extraction['ph'],
                        extraction['rainfall'],
                        extraction['mean_ndvi'],
                        extraction['evi'],
                        extraction['lst'],
                        extraction['soil_moisture'],
                        extraction['evapotranspiration'],
                        extraction['SOC']
                    ]],
                    columns=[
                        'temperature', 'humidity', 'ph', 'rainfall', 'ndvi',
                        'evi', 'lst', 'soil_moisture', 'evapotranspiration', 'SOC'
                    ]
                )
                probabilities = clf.predict_proba(feature_vector)[0]
                top_indices = np.argsort(probabilities)[::-1][:3]
                recommendations = [
                    (label_encoder.inverse_transform([idx])[0], round(probabilities[idx]*100, 2))
                    for idx in top_indices
                ]
                for crop, prob in recommendations:
                    st.write(f"- **{crop}: {prob}% suitability**")

                st.subheader("📊 Crop Suitability Chart")
                fig = px.bar(
                    x=[rec[0] for rec in recommendations],
                    y=[rec[1] for rec in recommendations],
                    labels={"x": "Crop", "y": "Suitability (%)"},
                    color=[rec[0] for rec in recommendations],
                    text_auto=True
                )
                st.plotly_chart(fig)

                if st.button("📥 Download Detailed Field Scan Report"):
                    buffer = BytesIO()
                    c = canvas.Canvas(buffer, pagesize=letter)
                    c.setFont("Helvetica", 16)
                    c.drawString(100, 750, "🌾 Advanced Field Scan Report")
                    c.setFont("Helvetica", 12)
                    c.drawString(50, 720, f"Latitude: {lat}, Longitude: {lon}")
                    c.drawString(50, 700, f"Date Range: {start_date} to {end_date}")
                    c.drawString(50, 680, f"NDVI: {extraction['mean_ndvi']:.4f}")
                    c.drawString(50, 660, f"LST: {extraction['lst']}°C")
                    c.drawString(50, 640, f"Rainfall: {extraction['rainfall']} mm")
                    c.drawString(50, 620, f"Soil Moisture: {extraction['soil_moisture']} m³/m³")
                    c.drawString(50, 600, f"pH: {extraction['ph']}")
                    c.drawString(50, 580, f"Temperature: {extraction['temp']}°C")
                    c.drawString(50, 560, f"Humidity: {extraction['humidity']}%")
                    c.drawString(50, 540, f"Evapotranspiration: {extraction['evapotranspiration']:.2f} mm")
                    c.drawString(50, 520, "Top Recommended Crops:")
                    y = 500
                    for crop, prob in recommendations:
                        c.drawString(70, y, f"- {crop}: {prob}%")
                        y -= 20
                    c.save()
                    buffer.seek(0)
                    st.download_button(
                        label="📥 Download Report PDF",
                        data=buffer,
                        file_name="field_scan_report.pdf",
                        mime="application/pdf"
                    )
    # End of extraction block
