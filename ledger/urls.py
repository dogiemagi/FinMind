from django.urls import path

from . import views

urlpatterns = [
    path("ledger/summary", views.SummaryView.as_view(), name="ledger-summary"),
    path("ledger/overdue", views.OverdueView.as_view(), name="ledger-overdue"),
]
