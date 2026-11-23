from django.contrib import admin
from django.urls import include, path

from assistant.views import MetricsView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("assistant.urls")),
    path("api/", include("ledger.urls")),
    path("api/", include("agents.urls")),
    path("metrics", MetricsView.as_view(), name="metrics"),
    path("", include("web.urls")),
]
