import json
import pickle
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt


def load_model():
    with open(settings.ML_MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def load_data():
    return pd.read_csv(settings.ACCIDENT_DATA_PATH)


def dashboard(request):
    df = load_data()
    zone_stats = df.groupby('zone').agg(
        total_accidents=('zone', 'count'),
        avg_risk=('risk_score', 'mean'),
        fatal_count=('severity', lambda x: (x == 'Fatal').sum()),
        lat=('latitude', 'mean'),
        lng=('longitude', 'mean'),
    ).reset_index()
    zone_stats['risk_level'] = zone_stats['avg_risk'].apply(
        lambda x: 'High' if x >= 0.65 else ('Medium' if x >= 0.4 else 'Low')
    )
    context = {
        'total_accidents': len(df),
        'high_risk_zones': len(zone_stats[zone_stats['risk_level'] == 'High']),
        'fatal_accidents': len(df[df['severity'] == 'Fatal']),
        'zones': json.dumps(zone_stats.to_dict('records')),
        'google_maps_key': settings.GOOGLE_MAPS_API_KEY,
        'hours': list(range(24)),
    }
    
    return render(request, 'dashboard.html', context)


def analytics(request):
    df = load_data()
    hourly = df.groupby('hour').size().reset_index(name='count')
    weather_dist = df.groupby('weather').size().reset_index(name='count')
    severity_dist = df.groupby('severity').size().reset_index(name='count')
    monthly = df.groupby('month').size().reset_index(name='count')
    road_risk = df.groupby('road_type').agg(avg_risk=('risk_score','mean'), count=('road_type','count')).reset_index()
    context = {
        'hourly_data': json.dumps(hourly.to_dict('records')),
        'weather_data': json.dumps(weather_dist.to_dict('records')),
        'severity_data': json.dumps(severity_dist.to_dict('records')),
        'monthly_data': json.dumps(monthly.to_dict('records')),
        'road_risk_data': json.dumps(road_risk.to_dict('records')),
        'total_records': len(df),
    }
    return render(request, 'analytics.html', context)


@csrf_exempt
def risk_analysis_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    data = json.loads(request.body)
    hour = int(data.get('hour', 12))
    weather = data.get('weather', 'Clear')
    day_type = data.get('day_type', 'Weekday')
    road_type = data.get('road_type', 'Urban Road')
    vehicles = int(data.get('vehicles', 2))
    month = int(data.get('month', 6))
    lat = float(data.get('lat', 19.2))
    lng = float(data.get('lng', 72.9))
    
    artifacts = load_model()
    model = artifacts['model']
    le_weather, le_day, le_road = artifacts['le_weather'], artifacts['le_day'], artifacts['le_road']
    
    try: w_enc = le_weather.transform([weather])[0]
    except: w_enc = 0
    try: d_enc = le_day.transform([day_type])[0]
    except: d_enc = 0
    try: r_enc = le_road.transform([road_type])[0]
    except: r_enc = 0
    
    features = np.array([[hour, w_enc, d_enc, r_enc, vehicles, month]])
    risk_class = model.predict(features)[0]
    risk_proba = model.predict_proba(features)[0]
    proba_dict = {str(c): round(float(p)*100, 1) for c, p in zip(model.classes_, risk_proba)}
    
    df = load_data()
    df['dist'] = ((df['latitude'] - lat)**2 + (df['longitude'] - lng)**2)**0.5
    nearby = df[df['dist'] < 0.05].head(10)
    nearby_accidents = nearby[['zone','severity','weather','road_type','hour','risk_score']].to_dict('records')
    
    color_map = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}
    
    night_hours = list(range(20,24))+list(range(0,6))
    if risk_class == 'High':
        rec = f"⚠️ HIGH RISK ZONE: {'Night driving — use high beam.' if hour in night_hours else 'Rush hour congestion likely.'} {'Reduce speed in wet conditions.' if weather == 'Rain' else ''}"
    elif risk_class == 'Medium':
        rec = f"🟡 MODERATE RISK: Drive cautiously. {'Rain detected — increase following distance.' if weather == 'Rain' else 'Standard precautions advised.'}"
    else:
        rec = "✅ LOW RISK: Normal driving conditions. Stay alert."
    
    return JsonResponse({
        'risk_level': str(risk_class),
        'risk_color': color_map.get(str(risk_class), '#95a5a6'),
        'probabilities': proba_dict,
        'nearby_accidents': nearby_accidents,
        'nearby_count': len(nearby),
        'recommendation': rec,
    })


def zone_data_api(request):
    df = load_data()
    points = df[['latitude','longitude','risk_score','severity','zone']].to_dict('records')
    zone_summary = df.groupby('zone').agg(
        count=('zone','count'), avg_risk=('risk_score','mean'),
        lat=('latitude','mean'), lng=('longitude','mean'),
        fatal=('severity', lambda x: (x=='Fatal').sum()),
    ).reset_index().to_dict('records')
    return JsonResponse({'points': points, 'zones': zone_summary})
    
