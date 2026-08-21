"""Run the simulated pilot and cache its results.

Slow enough that it should not run per request, so it writes an artifact the
pages and the report read, in the same pattern as Project 2's model cache.

    python manage.py run_project4_pilot
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from project4.ml import pilot

OUTPUT = Path(__file__).resolve().parents[2] / "artifacts" / "pilot.json"


class Command(BaseCommand):
    help = "Simulate the pilot study and write project4/artifacts/pilot.json"

    def add_arguments(self, parser):
        parser.add_argument("--participants", type=int, default=60)

    def handle(self, *args, **options):
        n = options["participants"]

        self.stdout.write(f"simulating {n} participants per time budget...")
        results = pilot.run(n_participants=n)

        self.stdout.write("checking sensitivity to the assumed ranking duration...")
        results["sensitivity"] = pilot.sensitivity(n_participants=max(20, n // 2))

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

        for row in results["budgets"]:
            self.stdout.write(
                f"  {row['minutes']}min: pairwise {row['pairwise_mean']:.1%}, "
                f"ranking {row['ranking_mean']:.1%}, "
                f"n = {row['n_required']}"
            )
        self.stdout.write(self.style.SUCCESS(f"Wrote {OUTPUT}"))
