import requests

API_KEY = "133d554dfa4a38e05d6c80341e71458d"
lat = 28.5134
lon = 71.0772

url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
w_r = requests.get(url).json()
print("Weather response:", w_r)
print("cod type:", type(w_r.get("cod")))

flood_url = f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge,river_discharge_mean,river_discharge_median,river_discharge_max,river_discharge_min&forecast_days=14"
f_r = requests.get(flood_url).json()
print("Flood response:", "daily" in f_r)
