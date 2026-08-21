"""Fit and cache the Project 2 models."""

from django.core.management.base import BaseCommand

from project2.ml import artifacts


class Command(BaseCommand):
    help = "Train the Project 2 model grid and write project2/artifacts/models.joblib"

    def handle(self, *args, **options):
        artifacts.build(verbose=True)
        self.stdout.write(self.style.SUCCESS("Project 2 artifacts rebuilt."))
