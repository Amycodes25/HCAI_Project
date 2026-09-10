from django.urls import path

from . import views

app_name = "learning_to_defer"

urlpatterns = [
    path("", views.overview, name="overview"),
    path("baseline/", views.baseline, name="baseline"),
    path("simulated-expert/", views.simulated_expert, name="simulated_expert"),
    path("learning-to-defer/", views.learning_to_defer, name="learning_to_defer"),
    path("active-learning/", views.active_learning, name="active_learning"),
    path("user-study/",views.user_study_intro,name="user_study_intro"),
    path("user-study/start/",views.start_user_study,name="start_user_study",),
    path("user-study/run/",views.user_study,name="user_study",),
    path("human-expert/", views.human_expert, name="human_expert"),
    path("classify/", views.classify, name="classify"),
]
