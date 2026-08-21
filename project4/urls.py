from django.urls import path

from . import views

app_name = "project4"

urlpatterns = [
    # Landing page. The brief asks for exactly two things reachable from here:
    # the PDF, and the study itself.
    path("", views.overview, name="index"),

    # Write-ups of the work that the PDF also covers.
    path("features/", views.task_features, name="features"),
    path("preference-model/", views.task_preference_model, name="preference_model"),
    path("study-design/", views.task_study_design, name="study_design"),

    # The participant flow. This is what a real participant would be given, so
    # it deliberately does not carry the project navigation.
    path("study/", views.study_consent, name="study_consent"),
    path("study/instructions/", views.study_instructions, name="study_instructions"),
    path("study/task/", views.study_task, name="study_task"),
    path("study/questionnaire/", views.study_questionnaire, name="study_questionnaire"),
    path("study/thank-you/", views.study_debrief, name="study_debrief"),
]
