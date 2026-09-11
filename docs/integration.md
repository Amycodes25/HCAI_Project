# Integration notes

What the work looked like before the projects were brought together, what
changed, and what is still worth doing. Written for the group, so that everyone
can see what happened to their own project and disagree with it if they want to.

Everything here is taken from the repository history rather than memory. Where
something is uncertain it says so.

---

## Where we started

- Work was split across **two repositories**, both containing a full copy of the
  course skeleton:
  - `Amycodes25/HCAI_Project` — branch `Chandana` (Project 1), branch `Amritha`
    (Project 3). Branches `Darren` and `Divya` contained only the initial commit.
  - `darrennoronha75/HCAI-Project-SoSe26` — Project 2, plus an older copy of
    Project 1.
- **Project 4 did not exist** in any form.
- The two Django projects had diverged in four files: `pbl/settings.py`,
  `pbl/urls.py`, `home/views.py`, `requirements.txt`. Everything else in the
  skeleton was byte-identical, because both branches descend from the same
  commit (`1254c3d`).
- The brief requires **one Django project, one launch page, one Git repository**.
  Two repositories and two launch pages did not meet that.

### Authorship, stated plainly

- Project 1 is Chandana's, in one commit (`751301e`, 19 July).
- Project 3 is Amritha's.
- Project 2's code was committed by `H P <hp@example.com>`. Divya's GitHub
  account made four commits the same day, all file creation and deletion with no
  code. The two identities are most likely the same person committing locally
  with an unconfigured Git identity, but **the repository does not prove that**,
  and it is worth confirming rather than assuming.
- Project 4 was written during integration.

---

## Shared setup

**Was**

- Two repositories, each with its own `home` page listing one project.
- The site root `http://127.0.0.1:8000/` returned **404** on Chandana's branch —
  home was mounted at `/home/`. The brief tells the reader to open the bare root.
- No shared stylesheet beyond the skeleton's, which nobody had modified.
- `requirements.txt` differed between branches; one pinned, one loose.

**Now**

- One Django project, six apps, one launch page at `/` listing all four projects.
- `static/theme.css` — shared palette, typography and page components.
- `static/dashboard.css` — shared sidebar and page shell.
- `home/templatetags/assets.py` provides `{% vstatic %}`, which stamps every CSS
  and JS URL with the file's modification time. Without it browsers served stale
  files: a correctly styled page appeared broken because the browser was running
  a cached stylesheet from an earlier version.
- `requirements.txt` (runtime) and `requirements-dev.txt` (only needed to
  regenerate committed artifacts) are separate. Nothing the site serves needs
  the dev packages.
- Deleted as dead: `notebooks/` (a README describing notebooks that were never
  written), `templates/hub_nav.html`, `project4/templates/project4/index.html`.

---

## Project 1 — Supervised Learning Interface

Chandana's. The largest set of changes: **+2066 lines, −577**, across 11 files.

**Was**

- 656 lines in `views.py`, 914 in one template.
- The template did **not** extend `base.html`, and there were **no CSS files** —
  416 lines of styling sat inline in the template. The page titled itself
  "Machine Learning Studio" and looked like a different website from the rest.
- `models.py` contained a comment saying models "will be defined here".
- `tests.py` was the 4-line Django stub.
- Accuracy was the only score.

**Now**

*Bugs fixed*

- The duplicate-column-name check ran `df.columns.duplicated()` **after**
  `read_csv`. Pandas silently renames a repeated column to `name.1`, so the check
  could never be true. The header is now read before pandas sees it.
- After uploading, the file input showed "No file chosen" again and nothing said
  what was loaded. The filename is kept and displayed.

*Brief compliance*

- Extends `base.html`; styling moved to `static/project1/style.css`. The brief
  names both paths as important.
- **Score selector** — accuracy, F1-macro, balanced accuracy. Task 4 asks "is the
  user in charge … of the choice of the score?" and only accuracy existed.
- **Django models** — `Dataset`, `TrainingRun`, `ModelResult`, with a migration.
  The brief says: "If you decide to implement several learning algorithms, you
  might have to define django models for the algorithms and the variables."
  Three algorithms were offered and there were none.

*Added*

- Confusion matrix for the best configuration, and a score-against-hyperparameter
  plot.
- "What you chose, and what we chose for you" — the control split, visible in the
  interface rather than only in a document.
- "Everything tried on this dataset" — every sweep is kept, so two model families
  can be compared with each other and not only within one sweep.
- `sample_data/iris.csv` and `sample_data/penguins.csv`, so the app can be tried
  without hunting for a file. Penguins is mixed numeric and categorical, which
  exercises the one-hot path Iris does not.
- Three near-identical training branches collapsed into one loop over a table of
  model specifications.
- Figures restyled from matplotlib defaults to the site palette.
- Forms post with `fetch` and the page is swapped in place, so an action no
  longer discards the page. They remain ordinary POST forms and work with
  JavaScript disabled.
- **39 tests**, up from 0.
- `docs/project1-design.md` had twelve `_TBD_` cells; it now records the
  decisions actually taken, including those not taken and why.

---

## Project 2 — Explainability

Committed by `H P`; see the authorship note above. Rewritten, because it was a
Gradio application and the brief requires Django.

**Kept exactly as it was**

- Both model families, and the complexity measures: number of leaves for the
  tree, ‖W‖₁ for logistic regression.
- The leaf grid (2–40), the C grid (`logspace(-3, 2, 30)`), `test_size=0.2`,
  `random_state=42`.
- The counterfactual method: local sampling, filter to the target class, rank by
  MAD-weighted L1 distance, widen the search if nothing is found.
- The PDP and ALE approach.

**The ML design was sound. What follows is the framework change and defects.**

**Was → Now**

- Gradio application, 451 lines in one file plus 507 in another (≈490 of which
  were unreachable Streamlit code after a `raise SystemExit`) → a Django app with
  the ML in `project2/ml/`.
- `multi_class="multinomial"` → removed. It was **removed in scikit-learn 1.7**,
  and `train_lregs()` runs at import, so on a current install the app did not
  start at all.
- `penalty="l1"` → removed. Deprecated in 1.8 and **silently overridden** by the
  default `l1_ratio`, so it fitted L2 while appearing to ask for L1.
- `island` encoded as an ordinal 0/1/2 → one-hot. Three unordered values were
  being given a fictitious ordering, in a project about reading model
  coefficients.
- `year` perturbed as a continuous quantity → treated as categorical. It records
  a field season, so counterfactuals were suggesting a penguin be measured in a
  different year.
- Counterfactuals re-randomised `island` and flipped `sex` on **every** draw, so
  no candidate could keep the original values and none was ever sparse → each
  draw now perturbs a random subset. Output changed from several features at once
  to, typically, "bill length 38.9 → 47.0".
- The README claimed an exact softmax derivative for logistic regression's ALE;
  the code ran the same finite-difference path for both models → the closed-form
  Jacobian is implemented and agrees with central differences to 1e-8. The brief
  asks exactly this question.
- `plot_tree` PNG → server-rendered SVG. One-hot splits read "Island is Dream?"
  instead of `categorical__island_Dream <= 0.5`, which was both ugly and
  backwards.
- Trees were fitted on standardised features, so thresholds displayed in standard
  deviations ("≤ 0.4") → fitted on raw values ("≤ 207.5 mm"). Accuracy unchanged,
  since trees are invariant to monotone rescaling.
- Models trained at import → 69 pre-fitted into a 27 KB joblib artifact. Dataset
  committed, so no network access at runtime.
- Every control reloaded the whole page → only the affected regions refresh.
  Changing the penguin went from 0.75s to 0.04s.
- The λ slider ran 0–1, defaulting to 0.01, unexplained. Above about 0.05 both
  families have already collapsed to their simplest model, so most of that range
  did nothing → starts at 0 (no penalty, most accurate model) and reports what
  the trade costs: "3.0 points of accuracy to remove 5 leaves".

---

## Project 3 — Active Learning for Learning-to-Defer

Amritha's. **Deliberately left almost alone: 93 insertions, 250 deletions** — and 209 of
those deletions are a single dead file. It was the most
complete project in the group and the closest to the deadline is the worst time
to refactor working code.

**Changed**

- `datasets` was imported at module scope in five `services/*.py` files. Because
  `tests.py` imports `train_baseline`, `manage.py test` **failed on a clean
  install from `requirements.txt`**. The import moved inside each `main()`, which
  is the only place it is used.
- "← Hub" added to the task navigation; "← HCAI · Group 29" added to both
  dashboard sidebars.
- Templates switched to `{% vstatic %}` for cache-busting.
- Its palette tokens now alias the shared ones instead of repeating the same hex
  values, so the two cannot drift apart.
- Deleted `services/simulated_expert.html` — 209 lines of HTML inside a Python
  package, referenced by nothing.

**Not touched**

- All machine learning, all metrics, the six-page report, and every page layout.

**Attempted and reverted**

- Migrating it onto the shared dashboard shell, which would have removed 22
  duplicated CSS classes. A before-and-after comparison of computed styles showed
  it broke the overview: the task grid is a deliberate six-column layout with
  cards that span, and the shared three-column grid destroyed it. Reverted.

---

## Project 4 — Preference Elicitation

Did not exist. Written during integration, so there is nothing to compare
against.

- IMDB 5000 dataset committed; 22-component feature representation.
- Plackett–Luce extension of Bradley–Terry, with MAP estimation of *w* by L-BFGS
  using a hand-derived gradient.
- A study design, and a pilot on 250 simulated participants that gives the
  sample size the real study would need.
- The participant interface: consent, instructions, both elicitation designs in
  counterbalanced order, a held-out validation block, questionnaire, debrief.
- A 7-page PDF report, generated from the implementation so its numbers cannot
  drift from the code.
- 34 tests.

---

## Where things stand

| | Project 1 | Project 2 | Project 3 | Project 4 |
|---|---|---|---|---|
| Python (excl. migrations) | 1353 | 1206 | 2304 | 1899 |
| CSS | 361 | 175 | 1110 | 295 |
| Templates | 2 | 5 | 8 | 12 |
| Tests | 39 | **0** | 4 | 34 |

---

## What is still worth doing

Ordered by what it costs against what it gains.

### Required, and not yet done

- **Matriculation numbers on the launch page.** This is Task 1 of Project 1 and
  the only requirement in the whole submission that is still unmet. Nothing
  substitutes for having them. They go in one list in `home/views.py`.

### Worth doing, in order

- **Tests for Project 2.** It has none, and it is the project that was rewritten
  most completely, so it is the one nobody else has reviewed. Roughly an hour,
  scoped to the numerical claims: the exact ALE gradient, counterfactual
  sparsity, and that raising λ never selects a more complex model.
- **Design notes on the Project 2 page.** The brief asks two questions directly —
  how categorical and binary features should be noised, and which model permits
  exact derivatives. Both are answered correctly in code and only one is
  explained on screen. "PDP" does not appear as text anywhere on the page; it is
  inside an image.
- **Project 3, Task 5 closing the loop.** Its own report names this: the labels a
  human provides are used for feedback and scoring but never fed back into the
  competence model. Amritha's call, not ours.
- **Project 3, Table 2 of the report** has misaligned rows, on page 2.

### Optional, if there is time

- **PDP against ALE on correlated features.** The course covers what PDPs do
  when features are correlated. Bill depth and flipper length are strongly correlated
  through species, so the two curves may visibly disagree — and you already plot
  them side by side. If they agree, that is also a result. Not yet checked.
- **Permutation feature importance in Project 1.** The course covers it, the
  original design notes list it, and it fits Project 1's own remit of comparing
  models. LIME would not — Project 2 is the explainability project, and putting
  explanation methods in Project 1 blurs a line the briefs draw clearly.
- **Migrating Project 3 onto the shared shell**, which removes about 140 lines of
  duplicated CSS. Sensible after submission, when breaking it costs nothing.

### Checked and not recommended

- **Explanation stability against model complexity.** The idea was to sweep λ and
  show that counterfactuals become less reproducible as the model grows, matching
  the result that post-hoc methods are uninformative when *f* is not
  simple. Tested: 2 leaves gave 1 distinct answer in 10 runs, 3 leaves gave 2,
  5 gave 2, 10 gave 1. No trend. Palmer Penguins is nearly separable and the
  trees stay small, so there is not enough complexity for the effect to appear.
- **Jupyter notebooks backing each project.** No brief asks for them, they would
  duplicate code that already exists in `project2/ml/` and `project4/ml/`, and
  duplicated code drifts. The 77 tests are stronger evidence that the maths works.
  A `notebooks/` README promising two notebooks already existed once and they were
  never written.

---

## Open questions for the group

- Who is `H P <hp@example.com>`? If it is Divya committing locally, the Git
  identity should be fixed before submission — a commit from `example.com` in a
  graded repository looks careless.
- `main` in `Amycodes25/HCAI_Project` is still the initial commit. At some point
  branch `Darren` needs to merge into it. Doing that through a pull request lets
  everyone see how the merge conflicts were resolved rather than discovering it.
- The merge resolved two conflicts in favour of one person's work: the launch
  page took Amritha's card layout over Chandana's, and Project 1's inline styling
  was replaced so all four projects share one appearance. Both are defensible and
  both are reversible.
