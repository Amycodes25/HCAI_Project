import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    """Restores the migration file for the schema already in use.

    This app's models (Dataset, TrainingRun, ModelResult) were already
    designed, applied, and in active use -- db.sqlite3 already has their
    tables, and django_migrations already records "project1.0001_initial" as
    applied -- but the migration file itself was missing from the project on
    disk. Without it, a fresh clone (a teammate's machine, or the grader's)
    would run `manage.py migrate` per the README, get no error, and simply
    never create these three tables. The first CSV upload would then fail
    with "no such table: project1_dataset".

    This file reproduces the schema exactly as it already exists locally, so
    `migrate` on this machine treats it as already applied (matching by app
    and migration name, not by file content) and does nothing here, while a
    fresh clone gets the tables it was always supposed to have.
    """

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Dataset",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("filename", models.CharField(max_length=255)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                ("n_rows", models.PositiveIntegerField()),
                ("n_columns", models.PositiveIntegerField()),
                ("column_names", models.JSONField()),
            ],
            options={
                "ordering": ["-uploaded_at"],
            },
        ),
        migrations.CreateModel(
            name="TrainingRun",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("algorithm", models.CharField(max_length=32)),
                ("algorithm_label", models.CharField(max_length=64)),
                ("parameter_name", models.CharField(max_length=64)),
                ("target", models.CharField(max_length=255)),
                ("test_size", models.FloatField()),
                ("score_name", models.CharField(max_length=32)),
                ("score_label", models.CharField(max_length=64)),
                ("best_parameter", models.CharField(max_length=64)),
                ("best_score", models.FloatField()),
                ("training_rows", models.PositiveIntegerField()),
                ("testing_rows", models.PositiveIntegerField()),
                (
                    "dataset",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="runs",
                        to="project1.dataset",
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="ModelResult",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("parameter_value", models.CharField(max_length=64)),
                ("score", models.FloatField()),
                (
                    "run",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="results",
                        to="project1.trainingrun",
                    ),
                ),
            ],
            options={
                "ordering": ["id"],
            },
        ),
    ]
