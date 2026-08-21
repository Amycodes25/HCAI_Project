from django.shortcuts import render


# Matriculation numbers are still to be collected; leave the value empty and the
# home template omits the separator rather than printing a placeholder.
STUDENTS = [
    {"name": "Darren", "matriculation": ""},
    {"name": "Chandana", "matriculation": ""},
    {"name": "Divya", "matriculation": ""},
    {"name": "Amritha", "matriculation": ""},
]

PROJECTS = [
    {
        "name": "Project 1: Supervised Learning Interface",
        "description": (
            "Upload a CSV, visualise it, and train several classifiers "
            "across a sweep of hyperparameter values."
        ),
        "url_name": "project1:index",
    },
    {
        "name": "Project 2: Explainability",
        "description": (
            "Interpretability and complexity on the Palmer Penguins dataset: "
            "regularisation trade-off, counterfactuals, PDP and ALE."
        ),
        "url_name": "project2:index",
    },
    {
        "name": "Project 3: Active Learning for Learning-to-Defer",
        "description": (
            "Train a news classifier and study when an AI should defer "
            "a decision to an expert."
        ),
        "url_name": "learning_to_defer:overview",
    },
    {
        "name": "Project 4: Preference Elicitation",
        "description": (
            "A movie recommender that learns your taste, and a user study "
            "comparing two ways of asking for your preferences."
        ),
        "url_name": "project4:index",
    },
]


def index(request):
    context = {
        "students": STUDENTS,
        "projects": PROJECTS,
    }
    return render(request, "home/index.html", context)
