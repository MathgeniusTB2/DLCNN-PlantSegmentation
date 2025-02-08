from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('analysis_feed/', views.analysis_feed, name='analysis_feed'),
    path('capture/', views.capture_image, name='capture_image'),
    path('history/', views.get_history, name='get_history'),
    path('export/', views.export_results, name='export_results'),
]