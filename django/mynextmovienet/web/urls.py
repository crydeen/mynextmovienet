from django.urls import path

from . import views

urlpatterns = [
    path('information/', views.information_page, name='information_page'),
    path('recommendations/', views.get_recommendations, name="get_recommendations"),
    path('popular-movies/', views.show_popular_movies, name="show_popular_movies"),
    path('recommendation-page/', views.get_recommendations_page, name="get_recommendations_page"),
    path('train-model/', views.train_model, name='train_model')
]