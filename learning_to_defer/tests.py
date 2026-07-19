from django.test import TestCase
from django.urls import reverse

from .services.train_baseline import build_pipeline, combine_text


class PageTests(TestCase):
    def test_overview_loads(self):
        response = self.client.get(reverse("learning_to_defer:overview"))
        self.assertEqual(response.status_code, 200)

    def test_baseline_loads_without_artifacts(self):
        response = self.client.get(reverse("learning_to_defer:baseline"))
        self.assertContains(response, "Baseline not trained yet")


class PipelineTests(TestCase):
    def test_title_and_description_are_combined(self):
        self.assertEqual(
            combine_text({"title": "Market rises", "description": "Stocks gain"}),
            "Market rises Stocks gain",
        )

    def test_pipeline_outputs_probabilities(self):
        model = build_pipeline(min_df=1)
        texts = [
            "government election president", "football team wins match",
            "stock market company profit", "software computer technology",
            "minister country talks", "basketball season game",
            "shares bank earnings", "internet device science",
        ]
        labels = [0, 1, 2, 3, 0, 1, 2, 3]
        model.fit(texts, labels)
        self.assertEqual(model.predict_proba(["team wins game"]).shape, (1, 4))
