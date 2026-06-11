"""
app.py – Pakistan Flood Risk & Disaster Prediction System
Uses Open-Meteo (free, no API key) for all weather + flood data.
Provides 15-day per-day flood risk predictions using the trained ML model.
"""

from flask import Flask, render_template, request
import joblib, json, requests, numpy as np
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
model         = joblib.load("model.pkl")
scaler        = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ── Cities list ───────────────────────────────────────────────────────────────
with open("pakistan_cities.json", "r", encoding="utf-8") as f:
    ALL_CITIES = sorted(json.load(f))

PKT = timezone(timedelta(hours=5))  # Pakistan Standard Time

def is_monsoon_season():
    return 6 <= datetime.now(PKT).month <= 9

def geocode_nominatim(name: str):
    """Geocode using Nominatim with Pakistan country filter."""
    try:
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={requests.utils.quote(name)}&countrycodes=pk"
            f"&format=json&limit=1&addressdetails=1"
        )
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "FloodRiskPK/2.0"}).json()
        if r:
            addr = r[0].get("address", {})
            display = (addr.get("city") or addr.get("town") or
                       addr.get("village") or addr.get("county") or name)
            return float(r[0]["lat"]), float(r[0]["lon"]), display
    except Exception:
        pass
    # Fallback: Open-Meteo geocoding
    try:
        url = (
            f"https://geocoding-api.open-meteo.com/v1/search"
            f"?name={requests.utils.quote(name + ' Pakistan')}&count=1&language=en&format=json"
        )
        r = requests.get(url, timeout=10).json()
        if r.get("results"):
            res = r["results"][0]
            return res["latitude"], res["longitude"], res.get("name", name)
    except Exception:
        pass
    return None, None, name

def reverse_geocode(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        r = requests.get(url, timeout=8,
                         headers={"User-Agent": "FloodRiskPK/2.0"}).json()
        addr = r.get("address", {})
        return (addr.get("village") or addr.get("town") or addr.get("city")
                or addr.get("county") or f"{float(lat):.4f}N, {float(lon):.4f}E")
    except Exception:
        return f"{float(lat):.4f}N, {float(lon):.4f}E"

def get_elevation(lat, lon):
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}",
            timeout=8).json()
        elev = r.get("elevation", [100.0])
        return float(elev[0]) if elev else 100.0
    except Exception:
        return 100.0

def predict_risk_day(elevation, precipitation, rain_7d, wind,
                     discharge, dis_mean, dis_max, discharge_ratio):
    """Predict flood risk for a single day."""
    features = {
        "elevation":            elevation,
        "precipitation":        precipitation,
        "rain_sum_7d":          rain_7d,
        "wind_speed":           wind,
        "river_discharge":      discharge,
        "river_discharge_mean": dis_mean,
        "river_discharge_max":  dis_max,
        "discharge_ratio":      discharge_ratio,
    }
    feat_vec = [features.get(f, 0.0) for f in feature_names]
    scaled   = scaler.transform([feat_vec])
    cls      = int(model.predict(scaled)[0])
    probs    = model.predict_proba(scaled)[0].tolist()
    return cls, probs

RISK_META = {
    0: {"label": "Safe",     "color": "#10b981", "css": "low-risk",  "icon": "✅",
        "advice": "No immediate flood threat. Stay informed about local forecasts."},
    1: {"label": "Moderate", "color": "#f59e0b", "css": "mod-risk",  "icon": "⚠️",
        "advice": "Waterlogging possible. Avoid underpasses and low-lying areas."},
    2: {"label": "Severe",   "color": "#ef4444", "css": "high-risk", "icon": "🚨",
        "advice": "High flood risk! Evacuate flood plains. Move to higher ground immediately."},
}

def risk_percentage(cls, probs):
    if cls == 0:
        return min(round(probs[1] * 100 + probs[2] * 50, 1), 29.9)
    elif cls == 1:
        return max(min(round((probs[1] + probs[2]) * 100, 1), 69.9), 30.0)
    else:
        return max(round(probs[2] * 100, 1), 70.0)

def fetch_all(city_name, lat=None, lon=None):
    display_name = city_name
    elevation    = 100.0

    if lat and lon:
        try:
            lat, lon = float(lat), float(lon)
            elevation    = get_elevation(lat, lon)
            display_name = reverse_geocode(lat, lon)
        except ValueError:
            lat = lon = None

    if not (lat and lon):
        lat, lon, display_name = geocode_nominatim(city_name)
        if lat is None:
            return None
        elevation = get_elevation(lat, lon)

    # ── 15-day forecast (past 7 days for rolling rain_7d context) ─────────────
    # Only VALID Open-Meteo daily variables
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,wind_speed_10m_max,temperature_2m_max,temperature_2m_min"
        f"&hourly=relative_humidity_2m"
        f"&past_days=7&forecast_days=15&timezone=Asia%2FKarachi"
    )
    flood_url = (
        f"https://flood-api.open-meteo.com/v1/flood"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=river_discharge,river_discharge_mean,river_discharge_max,"
        f"river_discharge_min,river_discharge_median"
        f"&past_days=7&forecast_days=15"
    )

    try:
        wr = requests.get(weather_url, timeout=15).json()
        fr = requests.get(flood_url,   timeout=15).json()
    except Exception:
        return None

    if "daily" not in wr:
        return None

    wd = wr["daily"]
    dates      = wd.get("time", [])
    precip_raw = wd.get("precipitation_sum",  [0] * len(dates))
    wind_raw   = wd.get("wind_speed_10m_max", [0] * len(dates))
    temp_max   = wd.get("temperature_2m_max", [30] * len(dates))
    temp_min   = wd.get("temperature_2m_min", [20] * len(dates))

    # ── Hourly humidity → daily average ───────────────────────────────────────
    hourly_data = wr.get("hourly", {})
    hourly_times = hourly_data.get("time", [])
    hourly_hum   = hourly_data.get("relative_humidity_2m", [])
    # Build date→avg humidity map
    hum_by_date = {}
    for i, ts in enumerate(hourly_times):
        d = ts[:10]  # "YYYY-MM-DD"
        val = float(hourly_hum[i]) if i < len(hourly_hum) and hourly_hum[i] is not None else 60.0
        hum_by_date.setdefault(d, []).append(val)
    hum_daily = {d: sum(v)/len(v) for d, v in hum_by_date.items()}

    # ── Flood data (may be missing for areas without river gauges) ─────────────
    fd = fr.get("daily", {}) if "daily" in fr else {}
    dates_f  = fd.get("time", [])
    disc_arr = fd.get("river_discharge",        [])
    dmean    = fd.get("river_discharge_mean",   [])
    dmax_arr = fd.get("river_discharge_max",    [])
    dmin_arr = fd.get("river_discharge_min",    [])
    dmed_arr = fd.get("river_discharge_median", [])

    def safe_float(arr, i, default=0.0):
        return float(arr[i]) if i < len(arr) and arr[i] is not None else default

    # Build flood lookup by date
    flood_by_date = {}
    for i, fd_date in enumerate(dates_f):
        flood_by_date[fd_date] = {
            "discharge": safe_float(disc_arr, i),
            "dis_mean":  safe_float(dmean,    i, 0.001),
            "dis_max":   safe_float(dmax_arr, i, 0.001),
            "dis_min":   safe_float(dmin_arr, i),
            "dis_med":   safe_float(dmed_arr, i),
        }

    # Rolling 7-day precipitation
    precip_arr  = np.array([p if p is not None else 0.0 for p in precip_raw], dtype=float)
    rain_7d_all = np.array([
        float(np.sum(precip_arr[max(0, i-6):i+1]))
        for i in range(len(precip_arr))
    ])

    today_str     = datetime.now(PKT).strftime("%Y-%m-%d")
    forecast_days = []

    for i, date_str in enumerate(dates):
        if date_str < today_str:
            continue
        if len(forecast_days) >= 15:
            break

        p   = float(precip_raw[i]) if precip_raw[i] is not None else 0.0
        w   = float(wind_raw[i])   if wind_raw[i]   is not None else 0.0
        tx  = float(temp_max[i])   if temp_max[i]   is not None else 30.0
        tn  = float(temp_min[i])   if temp_min[i]   is not None else 20.0
        r7  = float(rain_7d_all[i])
        hum = round(hum_daily.get(date_str, 65.0), 1)

        fld  = flood_by_date.get(date_str, {})
        d    = fld.get("discharge", 0.0)
        dm   = fld.get("dis_mean",  0.001)
        dmax = fld.get("dis_max",   0.001)
        dmin = fld.get("dis_min",   0.0)
        dmed = fld.get("dis_med",   0.0)
        dratio = d / max(dm, 0.001) if d > 0 else 0.0

        cls, probs = predict_risk_day(elevation, p, r7, w, d, dm, dmax, dratio)
        pct        = risk_percentage(cls, probs)
        meta       = RISK_META[cls]

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            day_label  = dt.strftime("%a")
            date_label = dt.strftime("%d %b")
        except Exception:
            day_label  = "—"
            date_label = date_str

        forecast_days.append({
            "date":          date_str,
            "day":           day_label,
            "date_label":    date_label,
            "is_today":      (date_str == today_str),
            "temp_max":      round(tx, 1),
            "temp_min":      round(tn, 1),
            "humidity":      hum,
            "precipitation": round(p, 1),
            "rain_7d":       round(r7, 1),
            "wind":          round(w, 1),
            "discharge":     round(d, 1),
            "dis_mean":      round(dm, 1),
            "dis_max":       round(dmax, 1),
            "dis_min":       round(dmin, 1),
            "dis_median":    round(dmed, 1),
            "discharge_ratio": round(dratio, 2),
            "risk_pct":      pct,
            "risk_class":    cls,
            "risk_label":    meta["label"],
            "risk_color":    meta["color"],
            "risk_css":      meta["css"],
            "risk_icon":     meta["icon"],
            "advice":        meta["advice"],
        })

    if not forecast_days:
        return None

    return {
        "name":       display_name,
        "lat":        lat,
        "lon":        lon,
        "elevation":  round(elevation, 0),
        "is_monsoon": is_monsoon_season(),
        "today":      forecast_days[0],
        "forecast":   forecast_days,
    }

@app.route("/", methods=["GET", "POST"])
def home():
    city    = request.form.get("city", "Lahore")
    lat     = request.form.get("lat")
    lon     = request.form.get("lon")
    now_str = datetime.now(PKT).strftime("%A, %d %B %Y")

    data = fetch_all(city, lat, lon)

    if not data:
        return render_template("index.html", error=True,
                               cities=ALL_CITIES, city=city,
                               today_str=now_str,
                               map_lat=30.3753, map_lon=69.3451)

    return render_template("index.html", error=False,
                           cities=ALL_CITIES, city=data["name"],
                           today_str=now_str, data=data,
                           map_lat=data["lat"], map_lon=data["lon"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)