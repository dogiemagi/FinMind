from django.urls import path

from . import views

urlpatterns = [
    path("agent/run", views.RunView.as_view(), name="agent-run"),
    path("agent/approve", views.ApproveView.as_view(), name="agent-approve"),
]
