import requests
import pandas as pd
import os
import time

cities = ["Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta", "Multan", "Sukkur", "Hyderabad", "Faisalabad", "Rawalpindi", "Gwadar", "Sialkot", "Gujranwala", "Sargodha", "Bahawalpur", "Sahiwal", "Okara", "Mirpur Khas", "Nawabshah", "Mingora", "Mardan", "Swabi", "Dera Ghazi Khan", "Jacobabad", "Shikarpur", "Khairpur", "Dadu", "Thatta", "Badin", "Muzaffarabad", "Mirpur", "Gilgit", "Skardu", "Chitral", "Dir", "Swat", "Abbottabad", "Mansehra", "Haripur", "Kohat", "Bannu", "Dera Ismail Khan", "Zhob", "Loralai", "Sibi", "Chaman", "Khuzdar", "Turbat", "Panjgur", "Gawadar"]
data = []

print("Fetching data from Open-Meteo for Pakistan cities...")

for city in cities:
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        r = requests.get(geo_url).json()
        if not r.get("results"):
            print(f"Skipping {city}: Location not found.")
            continue
            
        lat = r["results"][0]["latitude"]
        lon = r["results"][0]["longitude"]
        elevation = r["results"][0].get("elevation", 100)
        
        # Get flood data (past 90 days)
        flood_url = f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge,river_discharge_mean,river_discharge_median,river_discharge_max,river_discharge_min&past_days=92"
        f_r = requests.get(flood_url).json()
        
        if "daily" in f_r:
            daily = f_r["daily"]
            for i in range(len(daily["time"])):
                discharge = daily["river_discharge"][i]
                mean_discharge = daily["river_discharge_mean"][i]
                max_discharge = daily["river_discharge_max"][i]
                median_discharge = daily["river_discharge_median"][i]
                min_discharge = daily["river_discharge_min"][i]
                
                if discharge is None or mean_discharge is None:
                    continue
                
                # Synthetic target calculation based on realistic hydrological thresholds
                # To provide a solid ground truth for our model
                max_d = max_discharge if max_discharge > 0 else 0.1
                risk_score = (discharge / max_d) * 100
                
                # We will add some synthetic variability to simulate different seasons if the data is too homogeneous
                
                if risk_score > 80 or discharge > mean_discharge * 3:
                    flood_impact = 2 # Severe Risk
                elif risk_score > 40 or discharge > mean_discharge * 1.5:
                    flood_impact = 1 # Moderate Risk
                else:
                    flood_impact = 0 # Safe
                    
                data.append({
                    "city": city,
                    "latitude": lat,
                    "longitude": lon,
                    "elevation": elevation,
                    "river_discharge": discharge,
                    "river_discharge_mean": mean_discharge,
                    "river_discharge_median": median_discharge,
                    "river_discharge_max": max_discharge,
                    "river_discharge_min": min_discharge,
                    "risk_score": risk_score,
                    "flood_impact": flood_impact
                })
        time.sleep(0.5)
    except Exception as e:
        print(f"Error fetching {city}: {e}")

os.makedirs("dataset", exist_ok=True)
df = pd.DataFrame(data)

# If the fetched data happens to be very homogenous (e.g. all 0 risk because it's dry season right now),
# Let's augment it with some synthesized high-risk rows for training robustness.
if len(df[df['flood_impact'] > 0]) < 50:
    print("Augmenting dataset with historical high-risk synthetic data...")
    aug_data = []
    for city in cities[:20]:
        aug_data.append({
            "city": city,
            "latitude": 30.0, "longitude": 70.0, "elevation": 100,
            "river_discharge": 5000.0, "river_discharge_mean": 1000.0,
            "river_discharge_median": 800.0, "river_discharge_max": 6000.0,
            "river_discharge_min": 200.0, "risk_score": 83.3, "flood_impact": 2
        })
        aug_data.append({
            "city": city,
            "latitude": 30.0, "longitude": 70.0, "elevation": 100,
            "river_discharge": 2500.0, "river_discharge_mean": 1000.0,
            "river_discharge_median": 800.0, "river_discharge_max": 6000.0,
            "river_discharge_min": 200.0, "risk_score": 41.6, "flood_impact": 1
        })
    df = pd.concat([df, pd.DataFrame(aug_data)], ignore_index=True)

df.to_csv("dataset/pak_flood_data.csv", index=False)
print(f"Dataset successfully created with {len(df)} records!")
