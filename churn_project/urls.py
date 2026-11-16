from django.contrib import admin
from django.urls import path
from churn import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload_csv, name="upload_csv"),
    path("results/", views.results, name="results"),
    path("download/", views.download_results, name="download_results"),
]
