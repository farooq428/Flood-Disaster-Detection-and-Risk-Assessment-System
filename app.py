from flask import Flask, render_template, request
import joblib, json, requests, numpy as np
from datetime import datetime

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

with open('pakistan_cities.json', 'r', encoding='utf-8') as f:
    ALL_CITIES = sorted(json.load(f))

API_KEY = "133d554dfa4a38e05d6c80341e71458d"

def fetch_flood_risk(city, lat=None, lon=None):
    name = city
    elevation = 100.0
    
    # 1. Geocoding for small villages
    if not (lat and lon):
        query = city.strip()
        om_url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json"
        try:
            om_r = requests.get(om_url).json()
            if om_r.get("results"):
                lat = om_r["results"][0]["latitude"]
                lon = om_r["results"][0]["longitude"]
                name = om_r["results"][0]["name"]
        except Exception:
            pass

    # 2. Dynamic Elevation Data
    if lat and lon:
        try:
            e_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
            e_r = requests.get(e_url).json()
            if "elevation" in e_r and e_r["elevation"]:
                elevation = e_r["elevation"][0]
        except Exception:
            pass

    # 3. Weather Data (OpenWeatherMap)
    if lat and lon:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    else:
        query = city.strip()
        url = f"https://api.openweathermap.org/data/2.5/weather?q={query},PK&appid={API_KEY}&units=metric"
        
    try:
        w_r = requests.get(url).json()
    except Exception:
        return None
    
    if str(w_r.get("cod")) != "200":
        return None
        
    if "name" in w_r and w_r["name"]:
        name = w_r['name']
    
    # If clicked on map, name might be "Lat: ..., Lon: ...", we should fetch a proper reverse geocode or stick to it
    if name.startswith("Lat:"):
        try:
            rg_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
            headers = {'User-Agent': 'FloodRiskApp/1.0'}
            rg_r = requests.get(rg_url, headers=headers).json()
            if 'address' in rg_r:
                addr = rg_r['address']
                name = addr.get('village', addr.get('town', addr.get('city', addr.get('county', name))))
        except Exception:
            pass

    lat = w_r['coord']['lat']
    lon = w_r['coord']['lon']
    temp = w_r['main']['temp']
    humidity = w_r['main']['humidity']
    wind = w_r['wind']['speed']
    
    # Check rainfall (if any)
    rainfall = 0
    if 'rain' in w_r and '1h' in w_r['rain']:
        rainfall = w_r['rain']['1h']
        
    # 5-Day Forecast Data for Chart
    f_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    f_res = requests.get(f_url).json()
    forecast = []
    if 'list' in f_res:
        forecast = [{"day": e['dt'], "temp": e['main']['temp'], "rain": e.get('rain', {}).get('3h', 0)} for e in f_res['list'][::8]]

    # 2. Flood Data (Open-Meteo)
    flood_url = f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge,river_discharge_mean,river_discharge_median,river_discharge_max,river_discharge_min&forecast_days=14"
    f_r = requests.get(flood_url).json()
    
    max_forecast_discharge = 0
    mean_discharge = 0
    historical_max = 0
    historical_min = 0
    median_discharge = 0
    
    if "daily" in f_r:
        daily_flood = f_r["daily"]
        discharges = [x for x in daily_flood.get("river_discharge", [])[:5] if x is not None]
        if discharges:
            max_forecast_discharge = max(discharges)
        
        mean_d = [x for x in daily_flood.get("river_discharge_mean", [])[:5] if x is not None]
        if mean_d:
            mean_discharge = sum(mean_d) / len(mean_d)
            
        max_d = [x for x in daily_flood.get("river_discharge_max", [])[:5] if x is not None]
        if max_d:
            historical_max = max(max_d)
            
        min_d = [x for x in daily_flood.get("river_discharge_min", [])[:5] if x is not None]
        if min_d:
            historical_min = min(min_d)
            
        med_d = [x for x in daily_flood.get("river_discharge_median", [])[:5] if x is not None]
        if med_d:
            median_discharge = sum(med_d) / len(med_d)
            
    features = {
        'elevation': elevation,
        'river_discharge': max_forecast_discharge,
        'river_discharge_mean': mean_discharge,
        'river_discharge_median': median_discharge,
        'river_discharge_max': historical_max,
        'river_discharge_min': historical_min
    }
    
    feature_values = [features.get(f, 0) for f in feature_names]
    
    return feature_values, name, temp, humidity, wind, rainfall, forecast, lat, lon

@app.route("/", methods=["GET", "POST"])
def home():
    city = request.form.get("city", "Lahore")
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    
    result = fetch_flood_risk(city, lat, lon)
    today = datetime.now().strftime("%A, %d %B")

    if not result:
        return render_template("index.html", error=True, cities=ALL_CITIES, city=city, today=today, map_lat=30.3753, map_lon=69.3451)

    feats, name, temp, humidity, wind, rainfall, forecast, r_lat, r_lon = result
    
    # AI Prediction
    scaled_feats = scaler.transform([feats])
    impact_class = model.predict(scaled_feats)[0]
    probs = model.predict_proba(scaled_feats)[0]
    
    # Adjust logic: if rainfall is high, override model for safety
    if rainfall > 10.0:
        impact_class = 2
        probs = [0.0, 0.1, 0.9]
    elif rainfall > 2.0 and impact_class == 0:
        impact_class = 1
        probs = [0.2, 0.7, 0.1]
        
    if impact_class == 2:
        risk_percentage = round(probs[2] * 100, 1)
        impact_text = "Severe Flood Risk - Heavy Rain/Discharge"
        color_class = "high-risk"
    elif impact_class == 1:
        risk_percentage = round((probs[1] + probs[2]) * 100, 1)
        impact_text = "Moderate Risk - Waterlogging Possible"
        color_class = "mod-risk"
    else:
        risk_percentage = round(probs[1] * 100, 1)
        impact_text = "Safe - No Immediate Threat"
        color_class = "low-risk"
        
    if impact_class == 0 and risk_percentage > 30: risk_percentage = 30.0
    if impact_class == 1 and risk_percentage < 30: risk_percentage = 35.0
    if impact_class == 2 and risk_percentage < 70: risk_percentage = 75.0

    return render_template("index.html", 
                           city=name, 
                           temp=round(temp), 
                           humidity=round(humidity), 
                           wind=round(wind),
                           rainfall=rainfall,
                           forecast=forecast, 
                           cities=ALL_CITIES, 
                           today=today,
                           risk_percentage=risk_percentage,
                           impact_text=impact_text,
                           color_class=color_class,
                           river_discharge=round(feats[1], 2),
                           map_lat=r_lat,
                           map_lon=r_lon)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)