"""Tests for Project 4 - Preference elicitation.

    python manage.py test project4

Two halves: the model (Tasks 1 and 2), and the participant flow (Task 4).
"""

import numpy as np
from django.test import TestCase
from django.urls import reverse

from . import views
from .ml import features, preference
from .models import StudySession


class FeatureRepresentation(TestCase):
    """Task 1."""

    def test_dataset_loads(self):
        self.assertGreater(features.movie_count(), 4000)

    def test_representation_is_small_enough_to_estimate(self):
        # The point of the design: w must be identifiable from ~10 interactions.
        self.assertLessEqual(features.n_features(), 30)

    def test_design_matrix_shape_matches(self):
        X = features.design_matrix()
        self.assertEqual(X.shape, (features.movie_count(), features.n_features()))

    def test_no_missing_values(self):
        self.assertFalse(np.isnan(features.design_matrix()).any())

    def test_genre_columns_are_binary(self):
        genres = features.design_matrix()[:, :features.N_GENRES]
        self.assertTrue(np.all((genres == 0) | (genres == 1)))

    def test_films_can_carry_several_genres(self):
        # Multi-hot, not one-hot: the justification for the choice depends on it.
        genres = features.design_matrix()[:, :features.N_GENRES]
        self.assertGreater(genres.sum(axis=1).max(), 1)

    def test_continuous_columns_are_standardised(self):
        block = features.design_matrix()[
            :, features.N_GENRES:features.N_GENRES + len(features.CONTINUOUS_SPECS)
        ]
        self.assertTrue(np.allclose(block.mean(axis=0), 0, atol=1e-8))
        self.assertTrue(np.allclose(block.std(axis=0), 1, atol=1e-8))

    def test_sampling_is_reproducible_and_distinct(self):
        a = [m["index"] for m in features.sample_movies(10, seed=(0, 0))]
        b = [m["index"] for m in features.sample_movies(10, seed=(0, 0))]
        c = [m["index"] for m in features.sample_movies(10, seed=(0, 1))]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(set(a)), 10)

    def test_description_has_what_the_interface_shows(self):
        movie = features.describe(0)
        for key in ("index", "title", "year", "genres", "duration", "score"):
            self.assertIn(key, movie)


class PreferenceModel(TestCase):
    """Task 2."""

    def setUp(self):
        self.X = features.design_matrix()
        self.d = features.n_features()

    def test_reduces_to_bradley_terry_at_n_two(self):
        # The property that makes it an extension rather than a different model.
        rng = np.random.default_rng(0)
        w = rng.normal(size=self.d)
        xi, xj = self.X[10], self.X[500]

        plackett_luce = np.exp(
            preference.ranking_log_likelihood(w, [np.vstack([xi, xj])])
        )
        bradley_terry = preference.pairwise_probability(xi, xj, w)

        self.assertAlmostEqual(plackett_luce, bradley_terry, places=12)

    def test_ranking_probabilities_over_all_orderings_sum_to_one(self):
        from itertools import permutations

        rng = np.random.default_rng(3)
        w = rng.normal(size=self.d)
        X = self.X[[1, 2, 3, 4]]

        total = sum(
            np.exp(preference.ranking_log_likelihood(w, [X[list(order)]]))
            for order in permutations(range(4))
        )
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(1)
        rankings = [self.X[rng.choice(len(self.X), 6, replace=False)] for _ in range(3)]
        w = rng.normal(scale=0.3, size=self.d)

        _, analytic = preference._objective(w, rankings, 1.0)

        numeric = np.zeros_like(w)
        h = 1e-6
        for i in range(len(w)):
            up, down = w.copy(), w.copy()
            up[i] += h
            down[i] -= h
            numeric[i] = (
                preference._objective(up, rankings, 1.0)[0]
                - preference._objective(down, rankings, 1.0)[0]
            ) / (2 * h)

        self.assertLess(np.abs(analytic - numeric).max(), 1e-5)

    def test_fit_returns_zero_without_observations(self):
        self.assertTrue(np.allclose(preference.fit([], self.d), 0))

    def test_fit_recovers_a_known_preference(self):
        """A simulated user's w should be recoverable from their rankings."""
        rng = np.random.default_rng(7)
        w_true = rng.normal(size=self.d)
        w_true /= np.linalg.norm(w_true)

        rankings = []
        for _ in range(40):
            rows = self.X[rng.choice(len(self.X), 10, replace=False)]
            order = np.argsort(-(rows @ w_true))  # a decisive user
            rankings.append(rows[order])

        w_hat = preference.fit(rankings, self.d)
        cosine = w_hat @ w_true / (np.linalg.norm(w_hat) * np.linalg.norm(w_true))
        self.assertGreater(cosine, 0.8)

    def test_pairwise_probability_is_symmetric(self):
        rng = np.random.default_rng(5)
        w = rng.normal(size=self.d)
        xi, xj = self.X[3], self.X[400]
        self.assertAlmostEqual(
            preference.pairwise_probability(xi, xj, w)
            + preference.pairwise_probability(xj, xi, w),
            1.0, places=12,
        )


class ProjectPages(TestCase):
    def test_every_project_page_loads(self):
        for name in ("index", "features", "preference_model", "study_design"):
            with self.subTest(page=name):
                response = self.client.get(reverse(f"project4:{name}"))
                self.assertEqual(response.status_code, 200)

    def test_landing_page_offers_the_two_required_actions(self):
        # The brief prescribes exactly these two.
        response = self.client.get(reverse("project4:index"))
        self.assertContains(response, "report.pdf")
        self.assertContains(response, reverse("project4:study_consent"))

    def test_pages_link_back_to_the_hub(self):
        response = self.client.get(reverse("project4:index"))
        self.assertContains(response, 'href="/"')


class ParticipantFlow(TestCase):
    """Task 4."""

    def start(self):
        return self.client.post(reverse("project4:study_consent"), {"consent": "yes"})

    def test_study_cannot_start_without_consent(self):
        response = self.client.post(reverse("project4:study_consent"), {})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "agree")
        self.assertEqual(StudySession.objects.count(), 0)

    def test_consent_creates_a_pseudonymous_session(self):
        self.start()
        session = StudySession.objects.get()
        self.assertEqual(len(session.participant), 12)
        self.assertEqual(sorted(session.condition_order), ["pairwise", "ranking"])

    def test_pages_redirect_to_consent_when_not_started(self):
        for name in ("study_instructions", "study_task", "study_questionnaire",
                     "study_debrief"):
            with self.subTest(page=name):
                response = self.client.get(reverse(f"project4:{name}"))
                self.assertRedirects(response, reverse("project4:study_consent"))

    def test_condition_order_alternates_across_participants(self):
        # Counterbalancing has to work across participants, not within one.
        firsts = []
        for _ in range(4):
            self.client.cookies.clear()
            self.start()
        firsts = [s.condition_order[0] for s in StudySession.objects.all()]
        self.assertEqual(firsts, ["pairwise", "ranking", "pairwise", "ranking"])

    def _answer_trial(self, response):
        """Answer one trial the way a browser without JavaScript would."""
        html = response.content.decode()
        shown = html.split('name="shown" value="')[1].split('"')[0]
        indices = shown.split(",")
        data = {"shown": shown, "elapsed_ms": "1500"}
        if "movie-choice" in html:
            data["chosen"] = indices[0]
        else:
            for position, index in enumerate(indices, start=1):
                data[f"pos_{index}"] = position
        return self.client.post(reverse("project4:study_task"), data)

    def test_full_flow_without_javascript(self):
        self.start()
        self.client.post(reverse("project4:study_instructions"), {})

        answered = 0
        for _ in range(40):
            response = self.client.get(reverse("project4:study_task"))
            if response.status_code == 302:
                if "questionnaire" in response["Location"]:
                    break
                self.client.post(reverse("project4:study_instructions"), {})
                continue
            self._answer_trial(response)
            answered += 1

        # 8 trials per interface, plus the 6 held-out validation trials.
        self.assertEqual(answered, 2 * views.TRIALS_PER_CONDITION + views.VALIDATION_TRIALS)

        self.client.post(reverse("project4:study_questionnaire"),
                         {"effort_1": "3", "effort_2": "5", "preferred": "1"})
        response = self.client.get(reverse("project4:study_debrief"))
        self.assertEqual(response.status_code, 200)

        session = StudySession.objects.get()
        self.assertTrue(session.completed)
        self.assertEqual(len(session.estimated_w), features.n_features())

    def test_refreshing_the_trial_page_after_finishing_moves_on(self):
        self.start()
        session_data = self.client.session["project4"]
        session_data["stage"] = len(session_data["order"])  # every block done
        self.client.session["project4"] = session_data
        session = self.client.session
        session["project4"] = session_data
        session.save()

        response = self.client.get(reverse("project4:study_task"))
        self.assertRedirects(response, reverse("project4:study_questionnaire"),
                             fetch_redirect_response=False)

    def test_validation_block_runs_last_and_is_pairwise(self):
        """The held-out block is what makes the primary measure computable."""
        self.start()
        order = self.client.session["project4"]["order"]
        self.assertEqual(order[-1], views.VALIDATION)
        self.assertEqual(len(order), 3)

    def test_validation_answers_are_excluded_from_the_fit(self):
        self.start()
        self.client.post(reverse("project4:study_instructions"), {})
        for _ in range(40):
            response = self.client.get(reverse("project4:study_task"))
            if response.status_code == 302:
                if "questionnaire" in response["Location"]:
                    break
                self.client.post(reverse("project4:study_instructions"), {})
                continue
            self._answer_trial(response)

        responses = self.client.session["project4"]["responses"]
        held_out = [r for r in responses if r["design"] == views.VALIDATION]
        self.assertEqual(len(held_out), views.VALIDATION_TRIALS)
        # Every held-out trial is a pairwise choice, so it can score any estimate.
        for response in held_out:
            self.assertEqual(len(response["ordering"]), 2)

    def test_debrief_reports_the_primary_measure_per_interface(self):
        self.start()
        self.client.post(reverse("project4:study_instructions"), {})
        for _ in range(40):
            response = self.client.get(reverse("project4:study_task"))
            if response.status_code == 302:
                if "questionnaire" in response["Location"]:
                    break
                self.client.post(reverse("project4:study_instructions"), {})
                continue
            self._answer_trial(response)
        self.client.post(reverse("project4:study_questionnaire"), {"preferred": "1"})

        response = self.client.get(reverse("project4:study_debrief"))
        self.assertContains(response, "Held-out agreement")
        # One row per elicitation interface, not for the validation block.
        self.assertEqual(len(response.context["per_design"]), 2)
        for row in response.context["per_design"]:
            self.assertIsNotNone(row["agreement"])

    def test_debrief_clears_the_session(self):
        self.start()
        self.client.get(reverse("project4:study_debrief"))
        response = self.client.get(reverse("project4:study_debrief"))
        self.assertRedirects(response, reverse("project4:study_consent"))
