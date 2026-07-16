from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('analytics/', views.analytics, name='analytics'),
    path('api/risk/', views.risk_analysis_api, name='risk_api'),
    path('api/zones/', views.zone_data_api, name='zone_api'),
]
