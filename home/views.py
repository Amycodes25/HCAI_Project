# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    # TODO: replace placeholders with real names + matriculation numbers.
    students = [
        {"name": "Member 1 (TBD)", "matriculation": "TBD"},
        {"name": "Member 2 (TBD)", "matriculation": "TBD"},
        {"name": "Member 3 (TBD)", "matriculation": "TBD"},
        {"name": "Member 4 (TBD)", "matriculation": "TBD"},
    ]

    projects = [
        {"name": "Project 1 — Supervised Learning Interface",
         "url_name": "project1:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))