import requests

queries = ["Garhmor", "Garh More", "Garh Maharaja", "Chak 44", "Peshawar"]

for q in queries:
    headers = {'User-Agent': 'FloodRiskApp/1.0'}
    url = f"https://nominatim.openstreetmap.org/search?q={q}, Pakistan&format=json&limit=1"
    r = requests.get(url, headers=headers).json()
    if r:
        print(f"Nominatim: {q} -> Found!", r[0]['lat'], r[0]['lon'])
    else:
        print(f"Nominatim: {q} -> Not Found")

    om_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1&language=en&format=json"
    om_r = requests.get(om_url).json()
    if om_r.get("results"):
        print(f"Open-Meteo: {q} -> Found!")
    else:
        print(f"Open-Meteo: {q} -> Not Found")
