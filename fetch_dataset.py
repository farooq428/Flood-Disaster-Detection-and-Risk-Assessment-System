"""
fetch_dataset.py  –  Final Working Version
==========================================
Uses Open-Meteo geocoding (which worked originally) with a Pakistan bounding
box check (lat 23-38, lon 60-77) to reject wrong country matches.
Uses the Archive API with ONLY valid daily variables.

Features: elevation, precipitation, rain_sum_7d, wind_speed,
          river_discharge, river_discharge_mean, river_discharge_max, discharge_ratio
"""

import requests, pandas as pd, numpy as np, os, time
from datetime import datetime, timedelta

END_DATE   = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=91)).strftime("%Y-%m-%d")
print(f"Date range: {START_DATE} to {END_DATE}")

# Pakistan bounding box
PK_LAT_MIN, PK_LAT_MAX = 23.5, 37.5
PK_LON_MIN, PK_LON_MAX = 60.5, 77.5

CITIES = [
    "Lahore", "Faisalabad", "Rawalpindi", "Gujranwala", "Multan",
    "Sialkot", "Bahawalpur", "Sargodha", "Rahim Yar Khan", "Jhang",
    "Sahiwal", "Muzaffargarh", "Dera Ghazi Khan", "Khanewal", "Layyah",
    "Mianwali", "Pakpattan", "Hafizabad", "Chiniot", "Narowal",
    "Karachi", "Sukkur", "Larkana", "Nawabshah", "Jacobabad",
    "Shikarpur", "Mirpur Khas", "Dadu", "Khairpur", "Badin",
    "Sanghar", "Qambar", "Ghotki", "Shahdadkot", "Kotri",
    "Peshawar", "Mardan", "Abbottabad", "Mansehra", "Swat",
    "Charsadda", "Nowshera", "Kohat", "Bannu", "Dera Ismail Khan",
    "Quetta", "Turbat", "Khuzdar", "Loralai", "Sibi",
]

def geocode_pk(city):
    """Geocode via Open-Meteo, validate result is within Pakistan bounding box."""
    try:
        url = (
            f"https://geocoding-api.open-meteo.com/v1/search"
            f"?name={requests.utils.quote(city)}&count=5&language=en&format=json"
        )
        r = requests.get(url, timeout=10).json()
        for res in r.get("results", []):
            lat  = res["latitude"]
            lon  = res["longitude"]
            elev = float(res.get("elevation") or 100.0)
            if PK_LAT_MIN <= lat <= PK_LAT_MAX and PK_LON_MIN <= lon <= PK_LON_MAX:
                return lat, lon, elev
    except Exception:
        pass
    return None, None, 100.0

def fetch_json(url, timeout=20, retries=2):
    for attempt in range(retries + 1):
        try:
            return requests.get(url, timeout=timeout).json()
        except Exception as e:
            if attempt < retries:
                time.sleep(3)
            else:
                return {}

data = []
print(f"\nFetching data for {len(CITIES)} Pakistan cities...\n")

for idx, city in enumerate(CITIES):
    print(f"  [{idx+1}/{len(CITIES)}] {city}...", end=" ", flush=True)
    try:
        lat, lon, elevation = geocode_pk(city)
        if lat is None:
            print("SKIP: not found in Pakistan")
            continue
        print(f"lat={lat:.2f}, lon={lon:.2f}, elev={elevation:.0f}m")

        # Archive weather API – VALID daily variables only
        wr = fetch_json(
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={START_DATE}&end_date={END_DATE}"
            f"&daily=precipitation_sum,wind_speed_10m_max"
            f"&timezone=Asia%2FKarachi"
        )

        # Flood / discharge API
        fr = fetch_json(
            f"https://flood-api.open-meteo.com/v1/flood"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=river_discharge,river_discharge_mean,river_discharge_max,river_discharge_min"
            f"&past_days=91&forecast_days=1"
        )

        if "daily" not in wr:
            print(f"    WARNING: No weather archive data for {city}")
            continue

        wd       = wr["daily"]
        dates_w  = wd.get("time", [])
        p_raw    = wd.get("precipitation_sum",  [0] * len(dates_w))
        w_raw    = wd.get("wind_speed_10m_max", [0] * len(dates_w))

        # Flood lookup by date
        flood_by_date = {}
        if "daily" in fr:
            fd      = fr["daily"]
            dates_f = fd.get("time", [])
            disc    = fd.get("river_discharge",        [])
            dmean   = fd.get("river_discharge_mean",   [])
            dmax    = fd.get("river_discharge_max",    [])
            for i, d in enumerate(dates_f):
                flood_by_date[d] = {
                    "d":    float(disc[i])  if i < len(disc)  and disc[i]  is not None else 0.0,
                    "dm":   float(dmean[i]) if i < len(dmean) and dmean[i] is not None else 0.001,
                    "dmax": float(dmax[i])  if i < len(dmax)  and dmax[i]  is not None else 0.001,
                }

        # Rolling 7-day rain
        p_arr   = np.array([p if p is not None else 0.0 for p in p_raw], dtype=float)
        r7_arr  = np.array([float(np.sum(p_arr[max(0,i-6):i+1])) for i in range(len(p_arr))])

        city_rows = 0
        has_disc  = False
        for i, date in enumerate(dates_w):
            p   = float(p_raw[i]) if p_raw[i] is not None else 0.0
            w   = float(w_raw[i]) if w_raw[i] is not None else 0.0
            r7  = float(r7_arr[i])

            fld    = flood_by_date.get(date, {})
            d      = fld.get("d",    0.0)
            dm     = fld.get("dm",   0.001)
            dmax   = fld.get("dmax", 0.001)
            dratio = d / max(dm, 0.001) if d > 0 else 0.0

            if d > 0.1:
                has_disc = True

            # Pakistan NDMA-calibrated flood labeling
            if d > 0.1:
                # Discharge data available
                if dratio > 2.5 or (p > 40 and dratio > 1.5) or r7 > 120:
                    label = 2
                elif dratio > 1.4 or p > 20 or r7 > 60:
                    label = 1
                else:
                    label = 0
            else:
                # Precipitation-only fallback
                if p > 50 or r7 > 150:
                    label = 2
                elif p > 20 or r7 > 60:
                    label = 1
                else:
                    label = 0

            data.append({
                "city":                 city,
                "elevation":            elevation,
                "precipitation":        round(p, 2),
                "rain_sum_7d":          round(r7, 2),
                "wind_speed":           round(w, 2),
                "river_discharge":      round(d, 4),
                "river_discharge_mean": round(dm, 4),
                "river_discharge_max":  round(dmax, 4),
                "discharge_ratio":      round(dratio, 4),
                "flood_impact":         label,
            })
            city_rows += 1

        print(f"    +{city_rows} rows (discharge={'yes' if has_disc else 'no'})")
        time.sleep(0.5)

    except Exception as e:
        print(f"    ERROR: {e}")
        time.sleep(2)

os.makedirs("dataset", exist_ok=True)
df = pd.DataFrame(data)
print(f"\n{'='*55}")
print(f"Dataset: {len(df)} rows from {df['city'].nunique() if len(df)>0 else 0}/{len(CITIES)} cities")
if len(df) > 0:
    vc = df["flood_impact"].value_counts().sort_index()
    print(f"Classes: Safe={vc.get(0,0)}  Moderate={vc.get(1,0)}  Severe={vc.get(2,0)}")
print(f"{'='*55}")
df.to_csv("dataset/pak_flood_data.csv", index=False)
print(f"Saved: dataset/pak_flood_data.csv")
