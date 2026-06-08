import requests

city = "Lahore"
geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
geo_r = requests.get(geo_url).json()

lat = geo_r["results"][0]["latitude"]
lon = geo_r["results"][0]["longitude"]

weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
w_r = requests.get(weather_url).json()
print("Current Weather:", w_r.get('current'))

flood_url = f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge,river_discharge_mean,river_discharge_median,river_discharge_max,river_discharge_min&forecast_days=14"
f_r = requests.get(flood_url).json()
daily_flood = f_r.get("daily", {})
print("Flood Data river_discharge:", daily_flood.get("river_discharge", [])[:5])
