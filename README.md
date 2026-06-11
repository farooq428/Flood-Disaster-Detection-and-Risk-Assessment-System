# Flood Disaster & Risk Assessment System

A Flask-based flood disaster detection and risk assessment web app for Pakistani cities. The app fetches live weather data from OpenWeatherMap and river discharge/elevation data from Open-Meteo, transforms it into model features, and predicts the severity of flood risk using a trained machine learning model.

## Project Structure

- `app.py` - Flask application serving the dashboard and prediction API.
- `train_model.py` - Script to train a Random Forest classifier and save model artifacts.
- `requirements.txt` - Python dependencies required to run the app.
- `dataset/pak_flood_data.csv` - Historical flood dataset used to train the model.
- `pakistan_cities.json` - List of supported Pakistani cities for the UI dropdown.
- `templates/index.html` - HTML template for the dashboard.
- `static/style.css` - App styling.
- `model.pkl`, `scaler.pkl`, `feature_names.pkl` - Saved model artifacts loaded by `app.py`.

## Features

- Live weather and flood data lookup for Pakistani locations
- Precise Geocoding and dynamic elevation fetch using Open-Meteo
- Flood severity risk prediction using a trained Random Forest model
- Interactive Map for selecting remote villages and locations
- Responsive dashboard with temperature, humidity, wind, rainfall, river discharge, and a 5-day forecast chart
- Easy local deployment with Flask

## Prerequisites

- Python 3.10+ installed
- Git installed
- Internet access to fetch OpenWeatherMap data

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/Rainfall-Prediction0.1.git
cd Rainfall-Prediction0.1
```

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
```

- PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

- Command Prompt:

```cmd
.\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Verify the required files exist:

- `dataset/pak_flood_data.csv`
- `pakistan_cities.json`
- `model.pkl`
- `scaler.pkl`
- `feature_names.pkl`

If `model.pkl`, `scaler.pkl`, or `feature_names.pkl` are missing, run the model training script.

## Train the Model (Optional)

If the saved model files are not present, generate them with:

```bash
python train_model.py
```

This will create:

- `model.pkl`
- `scaler.pkl`
- `feature_names.pkl`

## Configure the OpenWeatherMap API Key

The app uses OpenWeatherMap for live weather and Open-Meteo for flood and geocoding data. By default, `app.py` contains a hardcoded OpenWeatherMap API key.

1. Get your own API key at: https://openweathermap.org/api
2. Open `app.py`
3. Replace the value of `API_KEY` with your key:

```python
API_KEY = "your_openweather_api_key"
```

> Note: Open-Meteo does not require an API key for the free tier. For production use, move your OpenWeatherMap API key to a `.env` file or environment variable instead of keeping it directly in `app.py`.

## Run the Web App

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Usage

- Choose a Pakistani city from the dropdown, type a city name, or **click anywhere on the map**.
- Click `Analyze Location`.
- The app will display:
  - Flood risk assessment and severity percentage
  - Current temperature, humidity, and rainfall
  - Current river discharge volume
  - 5-day temperature forecast chart

## Troubleshooting

- If the app shows `City data unavailable`, check the city name spelling.
- If OpenWeatherMap returns an error, verify your API key and internet connection.
- If the app fails to start, ensure all dependencies are installed and the virtual environment is active.

## Notes

- This project uses a Random Forest classifier trained on historical rainfall data.
- The UI is implemented using Flask templates and Chart.js for the forecast chart.

## Recommended GitHub Workflow

- Commit code changes, not generated files such as model artifacts or local environment folders.
- Keep API keys out of version control.

## License

You can add your preferred license here, for example MIT or Apache 2.0.
