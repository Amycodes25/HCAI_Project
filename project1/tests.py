"""Tests for Project 1 - Supervised Learning Interface.

Run with:

    python manage.py test project1

These cover what the brief asks the app to do (upload a CSV, visualise it, train
a model across a hyperparameter sweep) and the ways a user can hold it wrong.
Every test drives the real view through the Django test client, so they exercise
the same code path a browser does.
"""

import io

from django.test import TestCase
from django.urls import reverse

from .models import Dataset, ModelResult, TrainingRun

IRIS = b"""SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
7.6,3.0,6.6,2.1,virginica
"""


def csv_upload(content, name="data.csv"):
    handle = io.BytesIO(content)
    handle.name = name
    return handle


class Project1Base(TestCase):
    def setUp(self):
        self.url = reverse("project1:index")

    def upload(self, content=IRIS, name="iris.csv"):
        return self.client.post(
            self.url, {"action": "upload", "csv_file": csv_upload(content, name)}
        )

    def context_of(self, response):
        # response.context is a list of contexts; the view renders one template.
        return response.context

    def error_in(self, response):
        return response.context.get("error")


class PageLoads(Project1Base):
    def test_index_renders_without_a_dataset(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.error_in(response))

    def test_page_links_back_to_the_hub(self):
        response = self.client.get(self.url)
        self.assertContains(response, 'href="/"')


class Upload(Project1Base):
    def test_valid_csv_is_accepted_and_summarised(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.error_in(response))
        self.assertEqual(response.context["row_count"], 18)
        self.assertEqual(response.context["column_count"], 5)

    def test_last_column_is_suggested_as_the_target(self):
        # The brief states the last column is the label.
        response = self.upload()
        self.assertEqual(response.context["default_target"], "Species")

    def test_dataset_survives_across_requests(self):
        self.upload()
        response = self.client.get(self.url)
        self.assertEqual(response.context["row_count"], 18)

    def test_non_csv_is_rejected(self):
        response = self.client.post(
            self.url, {"action": "upload", "csv_file": csv_upload(IRIS, "notes.txt")}
        )
        self.assertIn("CSV", self.error_in(response))

    def test_missing_file_is_rejected(self):
        response = self.client.post(self.url, {"action": "upload"})
        self.assertIsNotNone(self.error_in(response))

    def test_empty_csv_is_rejected(self):
        response = self.upload(b"a,b,label\n")
        self.assertIsNotNone(self.error_in(response))

    def test_duplicate_column_names_are_rejected(self):
        response = self.upload(b"a,a,label\n1,2,0\n3,4,1\n")
        self.assertIn("duplicate", self.error_in(response).lower())

    def test_identifier_columns_are_filtered_out(self):
        response = self.upload(
            b"Id,SepalLengthCm,Species\n1,5.1,setosa\n2,4.9,setosa\n3,7.0,versicolor\n"
        )
        self.assertNotIn("Id", response.context["columns"])

    def test_loaded_filename_is_remembered_and_shown(self):
        """A file input renders empty again, so the page must say what is loaded."""
        response = self.upload(name="iris.csv")
        self.assertEqual(response.context["data_filename"], "iris.csv")
        self.assertContains(response, "iris.csv")

        # and it survives a later request that does not re-upload
        response = self.client.get(self.url)
        self.assertEqual(response.context["data_filename"], "iris.csv")

    def test_reset_forgets_the_filename_too(self):
        self.upload(name="iris.csv")
        response = self.client.post(self.url, {"action": "reset"})
        self.assertIsNone(response.context.get("data_filename"))

    def test_reset_clears_the_dataset(self):
        self.upload()
        response = self.client.post(self.url, {"action": "reset"})
        self.assertNotIn("row_count", response.context)


class Persistence(Project1Base):
    """The brief suggests Django models once several algorithms are offered."""

    def test_upload_records_the_dataset(self):
        self.upload(name="iris.csv")
        dataset = Dataset.objects.get()
        self.assertEqual(dataset.filename, "iris.csv")
        self.assertEqual(dataset.n_rows, 18)
        self.assertEqual(dataset.n_columns, 5)
        self.assertIn("Species", dataset.column_names)

    def test_uploaded_rows_are_not_stored(self):
        """Only metadata is kept; the data itself stays in the session."""
        self.upload()
        stored = " ".join(str(v) for v in Dataset.objects.get().__dict__.values())
        self.assertNotIn("5.1", stored)

    def test_training_records_a_run_and_every_result(self):
        self.upload()
        self.client.post(self.url, {
            "action": "train", "target": "Species",
            "model_name": "knn", "test_size": "0.3",
        })
        run = TrainingRun.objects.get()
        self.assertEqual(run.algorithm, "knn")
        self.assertEqual(run.dataset, Dataset.objects.get())
        self.assertEqual(run.best_score, max(r.score for r in run.results.all()))
        self.assertEqual(ModelResult.objects.count(), run.results.count())
        self.assertGreater(run.results.count(), 1)

    def test_runs_accumulate_so_families_can_be_compared(self):
        self.upload()
        for model in ("tree", "knn", "logistic"):
            self.client.post(self.url, {
                "action": "train", "target": "Species",
                "model_name": model, "test_size": "0.3",
            })
        response = self.client.get(self.url)
        self.assertEqual(len(response.context["previous_runs"]), 3)
        self.assertContains(response, "Everything tried on this dataset")

    def test_a_run_appears_in_the_same_response_that_created_it(self):
        """The history must not lag a request behind the run it records."""
        self.upload()
        first = self.client.post(self.url, {
            "action": "train", "target": "Species",
            "model_name": "tree", "test_size": "0.3",
        })
        self.assertEqual(len(first.context["previous_runs"]), 1)

        second = self.client.post(self.url, {
            "action": "train", "target": "Species",
            "model_name": "knn", "test_size": "0.3",
        })
        self.assertEqual(len(second.context["previous_runs"]), 2)

    def test_history_is_scoped_to_the_current_dataset(self):
        self.upload(name="first.csv")
        self.client.post(self.url, {
            "action": "train", "target": "Species",
            "model_name": "tree", "test_size": "0.3",
        })
        self.upload(name="second.csv")
        response = self.client.get(self.url)
        self.assertEqual(response.context["previous_runs"], [])

    def test_deleting_a_dataset_takes_its_runs_with_it(self):
        self.upload()
        self.client.post(self.url, {
            "action": "train", "target": "Species",
            "model_name": "tree", "test_size": "0.3",
        })
        Dataset.objects.get().delete()
        self.assertEqual(TrainingRun.objects.count(), 0)
        self.assertEqual(ModelResult.objects.count(), 0)


class Visualisation(Project1Base):
    def test_scatter_plot_is_produced(self):
        self.upload()
        response = self.client.post(self.url, {
            "action": "plot",
            "x_feature": "SepalLengthCm",
            "y_feature": "PetalLengthCm",
            "plot_target": "Species",
        })
        self.assertIsNone(self.error_in(response))
        self.assertTrue(response.context["scatter_plot"])

    def test_plot_before_upload_is_refused(self):
        response = self.client.post(self.url, {
            "action": "plot",
            "x_feature": "a", "y_feature": "b", "plot_target": "c",
        })
        self.assertIsNotNone(self.error_in(response))

    def test_same_feature_on_both_axes_is_refused(self):
        self.upload()
        response = self.client.post(self.url, {
            "action": "plot",
            "x_feature": "SepalLengthCm",
            "y_feature": "SepalLengthCm",
            "plot_target": "Species",
        })
        self.assertIn("different", self.error_in(response).lower())

    def test_non_numeric_axis_is_refused(self):
        self.upload()
        response = self.client.post(self.url, {
            "action": "plot",
            "x_feature": "Species",
            "y_feature": "SepalLengthCm",
            "plot_target": "Species",
        })
        self.assertIsNotNone(self.error_in(response))


class Training(Project1Base):
    def train(self, model_name="tree", target="Species", test_size="0.3"):
        return self.client.post(self.url, {
            "action": "train",
            "target": target,
            "model_name": model_name,
            "test_size": test_size,
        })

    def test_each_model_family_trains(self):
        self.upload()
        for name in ("logistic", "tree", "knn"):
            with self.subTest(model=name):
                response = self.train(model_name=name)
                self.assertIsNone(self.error_in(response))
                self.assertTrue(response.context["hyperparameter_results"])

    def test_sweep_reports_one_result_per_hyperparameter_value(self):
        self.upload()
        response = self.train(model_name="knn")
        results = response.context["hyperparameter_results"]
        self.assertGreater(len(results), 1)
        for row in results:
            self.assertIn("parameter", row)
            self.assertIn("accuracy", row)

    def test_best_result_is_the_maximum_of_the_sweep(self):
        self.upload()
        response = self.train(model_name="tree")
        results = response.context["hyperparameter_results"]
        self.assertEqual(
            response.context["best_accuracy"],
            max(row["accuracy"] for row in results),
        )

    def test_split_sizes_reflect_the_requested_test_size(self):
        self.upload()
        response = self.train(test_size="0.5")
        train_rows = response.context["training_rows"]
        test_rows = response.context["testing_rows"]
        self.assertEqual(train_rows + test_rows, 18)
        self.assertEqual(test_rows, 9)

    def test_accuracy_is_a_percentage(self):
        self.upload()
        response = self.train()
        for row in response.context["hyperparameter_results"]:
            self.assertGreaterEqual(row["accuracy"], 0)
            self.assertLessEqual(row["accuracy"], 100)

    def test_each_score_can_be_selected(self):
        self.upload()
        for score in ("accuracy", "f1_macro", "balanced_accuracy"):
            with self.subTest(score=score):
                response = self.client.post(self.url, {
                    "action": "train", "target": "Species",
                    "model_name": "tree", "test_size": "0.3",
                    "score_name": score,
                })
                self.assertIsNone(self.error_in(response))
                self.assertEqual(response.context["score_name"], score)

    def test_unknown_score_falls_back_rather_than_failing(self):
        self.upload()
        response = self.client.post(self.url, {
            "action": "train", "target": "Species", "model_name": "tree",
            "test_size": "0.3", "score_name": "nonsense",
        })
        self.assertIsNone(self.error_in(response))
        self.assertEqual(response.context["score_name"], "accuracy")

    def test_scores_disagree_on_an_imbalanced_target(self):
        """The reason the choice is offered at all."""
        # The rare class sits on top of the common one, so a classifier that
        # never predicts it still scores well on accuracy. That is exactly the
        # situation where balanced accuracy earns its place.
        lines = ["a,b,label"]
        lines += [f"{1 + i * 0.01},{2 + i * 0.01},common" for i in range(20)]
        lines += [f"{1 + i * 0.01},{2 + i * 0.01},rare" for i in range(4)]
        self.upload(("\n".join(lines) + "\n").encode())

        accuracy = self.client.post(self.url, {
            "action": "train", "target": "label", "model_name": "logistic",
            "test_size": "0.5", "score_name": "accuracy",
        }).context["best_accuracy"]
        balanced = self.client.post(self.url, {
            "action": "train", "target": "label", "model_name": "logistic",
            "test_size": "0.5", "score_name": "balanced_accuracy",
        }).context["best_accuracy"]

        self.assertNotEqual(accuracy, balanced)

    def test_results_include_a_sweep_plot_and_confusion_matrix(self):
        self.upload()
        response = self.train()
        self.assertTrue(response.context["sweep_plot"])
        self.assertTrue(response.context["confusion_plot"])

    def test_training_before_upload_is_refused(self):
        response = self.train()
        self.assertIsNotNone(self.error_in(response))

    def test_single_class_target_is_refused(self):
        self.upload(b"a,b,label\n1,2,x\n3,4,x\n5,6,x\n7,8,x\n")
        response = self.train(target="label")
        self.assertIn("two classes", self.error_in(response))

    def test_identifier_like_target_is_refused(self):
        # A target that is almost all distinct values is not a class label.
        rows = b"a,b,label\n" + b"".join(
            f"{i},{i * 2},unique{i}\n".encode() for i in range(20)
        )
        self.upload(rows)
        response = self.train(target="label")
        self.assertIsNotNone(self.error_in(response))

    def test_invalid_target_is_refused(self):
        self.upload()
        response = self.train(target="NoSuchColumn")
        self.assertIsNotNone(self.error_in(response))

    def test_categorical_features_are_handled(self):
        # island is a string column; it must be encoded, not crash the pipeline.
        data = (
            b"island,mass,label\n"
            b"Biscoe,3200,a\nDream,3400,a\nBiscoe,3600,a\nTorgersen,3100,a\n"
            b"Dream,5000,b\nBiscoe,5200,b\nDream,5400,b\nTorgersen,5100,b\n"
        )
        self.upload(data)
        response = self.train(target="label")
        self.assertIsNone(self.error_in(response))
        self.assertIn("island", response.context["categorical_model_features"])

    def test_missing_values_are_imputed_not_fatal(self):
        data = (
            b"a,b,label\n1,2,x\n,3,x\n2,,x\n3,4,x\n"
            b"10,11,y\n11,,y\n,12,y\n13,14,y\n"
        )
        self.upload(data)
        response = self.train(target="label")
        self.assertIsNone(self.error_in(response))


class CrossValidation(Project1Base):
    """The brief asks who selects the hyperparameters; the decision record
    says cross-validation is the system's job. These check that a sweep is
    actually chosen by cross-validated performance on the training set, and
    that the held-out test score is measured separately, not used to pick
    the winner (see evaluate_model_hyperparameters in views.py)."""

    def train(self, model_name="tree", target="Species", test_size="0.3"):
        return self.client.post(self.url, {
            "action": "train", "target": target,
            "model_name": model_name, "test_size": test_size,
        })

    def test_a_normal_sized_sweep_uses_cross_validation(self):
        self.upload()  # the 18-row IRIS fixture, three balanced classes
        response = self.client.post(self.url, {
            "action": "train", "target": "Species",
            "model_name": "tree", "test_size": "0.3",
        })
        self.assertIsNone(self.error_in(response))
        self.assertTrue(response.context["used_cross_validation"])
        self.assertGreaterEqual(response.context["cv_folds"], 2)

    def test_held_out_test_score_is_reported_alongside_the_cv_score(self):
        self.upload()
        response = self.client.post(self.url, {
            "action": "train", "target": "Species",
            "model_name": "tree", "test_size": "0.3",
        })
        test_score = response.context["best_test_accuracy"]
        self.assertIsNotNone(test_score)
        self.assertGreaterEqual(test_score, 0)
        self.assertLessEqual(test_score, 100)

    def test_cv_folds_and_test_score_are_persisted_on_the_run(self):
        self.upload()
        response = self.client.post(self.url, {
            "action": "train", "target": "Species",
            "model_name": "knn", "test_size": "0.3",
        })
        run = TrainingRun.objects.get()
        self.assertEqual(run.cv_folds, response.context["cv_folds"])
        self.assertEqual(run.held_out_test_score, response.context["best_test_accuracy"])

    def test_too_little_data_falls_back_instead_of_crashing(self):
        """A handful of rows cannot support k-fold cross-validation. The run
        should still complete -- just without cross-validation -- rather than
        raising, and the page should say so."""
        data = b"a,b,label\n1,2,x\n2,3,x\n10,11,y\n11,12,y\n"
        self.upload(data)
        response = self.train(target="label", model_name="tree", test_size="0.5")
        self.assertIsNone(self.error_in(response))
        self.assertFalse(response.context["used_cross_validation"])


class FeatureSelection(Project1Base):
    """Decision: input features are the user's call, starting from a system
    suggestion (identifier-like and near-unique columns pre-excluded)."""

    IDENTIFIER_DATA = (
        b"id,length,width,label\n"
        b"1,5.0,3.0,x\n2,5.2,3.1,x\n3,5.1,2.9,x\n4,5.3,3.2,x\n"
        b"5,7.0,3.5,y\n6,7.2,3.6,y\n7,7.1,3.4,y\n8,7.3,3.7,y\n"
    )

    def test_no_selection_falls_back_to_the_system_suggestion(self):
        self.upload(self.IDENTIFIER_DATA, name="flowers.csv")
        response = self.train(target="label")
        self.assertIsNone(self.error_in(response))
        self.assertCountEqual(
            response.context["used_features"], ["length", "width"]
        )

    def test_user_can_drop_a_suggested_feature(self):
        self.upload(self.IDENTIFIER_DATA, name="flowers.csv")
        response = self.client.post(self.url, {
            "action": "train", "target": "label", "model_name": "tree",
            "test_size": "0.3", "features": ["width"],
        })
        self.assertIsNone(self.error_in(response))
        self.assertEqual(response.context["used_features"], ["width"])

    def test_user_can_add_back_an_ignored_column(self):
        self.upload(self.IDENTIFIER_DATA, name="flowers.csv")
        response = self.client.post(self.url, {
            "action": "train", "target": "label", "model_name": "tree",
            "test_size": "0.3", "features": ["id", "length", "width"],
        })
        self.assertIsNone(self.error_in(response))
        self.assertIn("id", response.context["used_features"])

    def train(self, model_name="tree", target="label", test_size="0.3"):
        return self.client.post(self.url, {
            "action": "train", "target": target,
            "model_name": model_name, "test_size": test_size,
        })


class OutlierHandling(Project1Base):
    """Decision: suspicious/outlier rows are flagged by the system, and
    excluding them from training is the user's choice, not a default."""

    # Twelve ordinary values clustered around 10, split across two balanced
    # classes, plus one point at 500 -- verified with pandas' IQR rule (in
    # the same way flag_outlier_rows does it) to flag exactly that one row.
    OUTLIER_DATA = (
        b"value,label\n"
        b"9,a\n10,a\n10,a\n11,a\n9,a\n10,a\n"
        b"10,b\n11,b\n9,b\n10,b\n10,b\n11,b\n"
        b"500,a\n"
    )

    def train(self, exclude=False):
        payload = {
            "action": "train", "target": "label", "model_name": "tree",
            "test_size": "0.3",
        }
        if exclude:
            payload["exclude_outliers"] = "on"
        return self.client.post(self.url, payload)

    def test_outlier_is_flagged_but_kept_by_default(self):
        self.upload(self.OUTLIER_DATA, name="outliers.csv")
        response = self.train(exclude=False)
        self.assertIsNone(self.error_in(response))
        self.assertEqual(response.context["outliers_flagged"], 1)
        self.assertEqual(response.context["outliers_excluded"], 0)
        self.assertEqual(
            response.context["training_rows"] + response.context["testing_rows"], 13
        )

    def test_excluding_flagged_outliers_removes_the_row(self):
        self.upload(self.OUTLIER_DATA, name="outliers.csv")
        response = self.train(exclude=True)
        self.assertIsNone(self.error_in(response))
        self.assertEqual(response.context["outliers_excluded"], 1)
        self.assertEqual(
            response.context["training_rows"] + response.context["testing_rows"], 12
        )

    def test_outliers_excluded_is_persisted_on_the_run(self):
        self.upload(self.OUTLIER_DATA, name="outliers.csv")
        self.train(exclude=True)
        run = TrainingRun.objects.get()
        self.assertEqual(run.outliers_excluded, 1)


class MissingValueStrategy(Project1Base):
    """Decision: missing values are handled automatically by default, and
    that is adjustable -- a user can ask to drop incomplete rows instead."""

    GAPPY_DATA = (
        b"a,b,label\n1,2,x\n,3,x\n2,,x\n3,4,x\n"
        b"10,11,y\n11,,y\n,12,y\n13,14,y\n"
    )

    def train(self, missing_strategy=None):
        payload = {
            "action": "train", "target": "label", "model_name": "tree",
            "test_size": "0.3",
        }
        if missing_strategy:
            payload["missing_strategy"] = missing_strategy
        return self.client.post(self.url, payload)

    def test_automatic_strategy_keeps_every_row(self):
        self.upload(self.GAPPY_DATA, name="gappy.csv")
        response = self.train()
        self.assertIsNone(self.error_in(response))
        self.assertEqual(
            response.context["training_rows"] + response.context["testing_rows"], 8
        )

    def test_drop_strategy_removes_incomplete_rows(self):
        self.upload(self.GAPPY_DATA, name="gappy.csv")
        response = self.train(missing_strategy="drop")
        self.assertIsNone(self.error_in(response))
        # Four of the eight rows have a gap in "a" or "b"; only the four
        # complete rows should remain once they are dropped.
        self.assertEqual(
            response.context["training_rows"] + response.context["testing_rows"], 4
        )
