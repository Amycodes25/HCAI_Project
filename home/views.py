# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.shortcuts import render


def index(request):
    students = [
        {"name": "Group member", "matriculation": "Add before submission"},
    ]
    
    projects = [
        {
            "name": "Project 3: Active Learning for Learning-to-Defer",
            "description": (
                "Train a news classifier and study when an AI should defer "
                "a decision to an expert."
            ),
            "url_name": "learning_to_defer:overview",
        },
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return render(request, "home/index.html", context)
