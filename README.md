# 🚦 AI-Powered Traffic Intelligence System

## Project Overview
A Django + Machine Learning web app for traffic accident risk analysis across Maharashtra, using realistic road accident data patterns based on NCRB / data.gov.in datasets.

---

## Tech Stack
- **Backend:** Python 3, Django 4.x
- **ML Model:** Random Forest Classifier (scikit-learn) — 74% accuracy
- **Data:** 1500+ synthetic accident records based on real Maharashtra road patterns
- **Maps:** Google Maps API (Heatmap + Marker visualization)
- **Charts:** Chart.js (Analytics page)

---

## Features
- 🗺️ **Interactive Google Maps heatmap** — visualizes accident density across Maharashtra
- 🔬 **ML Risk Analyzer** — click any map location → select conditions → get risk prediction
- 📊 **Analytics Dashboard** — hourly patterns, weather impact, road type risk, monthly trends
- 📍 **10 Real Maharashtra Zones** including Thane Ghodbunder Road, Mumbai Express Highways, Pune-Mumbai Highway, etc.

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install django scikit-learn pandas numpy
```

### 2. Get Google Maps API Key
- Go to: https://console.cloud.google.com/
- Enable: **Maps JavaScript API** + **Maps Visualization Library**
- Create an API Key
- Add it to `traffic_project/settings.py`:
```python
GOOGLE_MAPS_API_KEY = 'YOUR_KEY_HERE'
```

### 3. Run the App
```bash
python manage.py migrate
python manage.py runserver
```

### 4. Open in Browser
```
http://127.0.0.1:8000/
```

---

## Project Structure
```
traffic_intelligence/
├── traffic_project/
│   ├── settings.py          # Config + API keys
│   └── urls.py
├── traffic_app/
│   ├── views.py             # Django views + ML inference
│   └── urls.py
├── templates/
│   ├── base.html            # Dark sidebar layout
│   ├── dashboard.html       # Map + Risk Analyzer
│   └── analytics.html      # Charts + Insights
├── ml_model.pkl             # Trained Random Forest model
├── accident_data.csv        # Maharashtra accident dataset
└── README.md
```

---

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard with map |
| `/analytics/` | GET | Charts and analytics |
| `/api/risk/` | POST | ML risk prediction for a location |
| `/api/zones/` | GET | All accident points for heatmap |

### Risk API Example (POST /api/risk/)
```json
{
  "hour": 8,
  "weather": "Rain",
  "day_type": "Weekday",
  "road_type": "Junction",
  "vehicles": 2,
  "month": 7,
  "lat": 19.1960,
  "lng": 72.9690
}
```

---

## ML Model Details
- **Algorithm:** Random Forest (100 trees)
- **Features:** Hour, Weather, Day Type, Road Type, Vehicles Involved, Month
- **Target:** Risk Level (Low / Medium / High)
- **Training Data:** 1200 samples | Test: 300 samples
- **Accuracy:** 74%

---

## Real Data Integration (Future Upgrade)
Replace `accident_data.csv` with actual data from:
- **data.gov.in** → Road Accidents in India dataset
- **Maharashtra Police** → State accident records
- **MoRTH** → Ministry of Road Transport annual reports

Just ensure CSV has columns: `zone, latitude, longitude, hour, weather, day_type, road_type, severity, risk_score, vehicles_involved, year, month`
