from django.shortcuts import render


# The group, as the brief asks for it: names and matriculation numbers, set
# here in the view rather than written into the template.
STUDENTS = [
    {"name": "Darren Noronha", "matriculation": "637482"},
    {"name": "Chandana Putta", "matriculation": "637789"},
    {"name": "Divya Chidananda", "matriculation": "672434"},
    {"name": "Amritha Subramanian", "matriculation": "641269"},
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
            "Trade off accuracy against model complexity on the Palmer "
            "Penguins dataset, then read the chosen model through "
            "counterfactuals and feature effect plots."
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
            "Design a study comparing two ways of asking someone what they "
            "like, and the interface that would run it."
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
