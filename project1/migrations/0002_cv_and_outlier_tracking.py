from django.db import migrations, models


class Migration(migrations.Migration):
    """Adds the fields the cross-validation and outlier-handling work needs.

    Unlike 0001, this migration is genuinely new and has not been applied
    anywhere yet -- running `manage.py migrate` after pulling this change is
    required, on every machine including this one, or TrainingRun.objects
    .create(...) will fail with "no such column: project1_trainingrun
    .held_out_test_score".
    """

    dependencies = [
        ("project1", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="trainingrun",
            name="held_out_test_score",
            field=models.FloatField(null=True, blank=True),
        ),
        migrations.AddField(
            model_name="trainingrun",
            name="cv_folds",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="trainingrun",
            name="outliers_excluded",
            field=models.PositiveIntegerField(default=0),
        ),
    ]
