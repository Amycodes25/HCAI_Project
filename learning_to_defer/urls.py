from django.urls import path

from . import views

app_name = "learning_to_defer"

urlpatterns = [
    path("", views.overview, name="overview"),
    path("baseline/", views.baseline, name="baseline"),
    path("simulated-expert/", views.simulated_expert,name="simulated_expert",),
    path("learning-to-defer/",views.learning_to_defer,name="learning_to_defer",),
    path("active-learning/",views.active_learning,name="active_learning",),
    path("human-expert/",views.human_expert,name="human_expert",),
    path("classify/", views.classify, name="classify"),
    ]

