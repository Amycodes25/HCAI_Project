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

    def test_reset_clears_the_dataset(self):
        self.upload()
        response = self.client.post(self.url, {"action": "reset"})
        self.assertNotIn("row_count", response.context)


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
