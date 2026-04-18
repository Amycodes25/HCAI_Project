from django.http import HttpResponse
from django.template import loader


def index(request):
    """Landing page for Project 1.

    Placeholder until the upload + training views are implemented.
    """
    template = loader.get_template("project1/index.html")
    context = {
        "title": "Project 1 — Supervised Learning Interface",
        "subtitle": "Upload a CSV, visualise it, train models, compare results.",
    }
    return HttpResponse(template.render(context, request))
