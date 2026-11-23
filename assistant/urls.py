from django.urls import path

from . import views

urlpatterns = [
    path("health", views.HealthView.as_view(), name="health"),
    path("ask", views.AskView.as_view(), name="ask"),
    path("extract", views.ExtractView.as_view(), name="extract"),
    path("tokenize", views.TokenizeView.as_view(), name="tokenize"),
    path("generate", views.GenerateView.as_view(), name="generate"),
    path("report", views.ReportView.as_view(), name="report"),
]
