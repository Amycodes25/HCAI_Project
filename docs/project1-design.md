# Project 1 — Supervised Learning Interface: design notes

Living document. Capture decisions here as you make them, so the final write-up
and the UI stay in sync.

## Problem statement

Build a Django app inside the HCAI-PBL project that lets a user:

1. Upload a CSV (feature names in row 0, label in the last column — see PDF §2.4).
2. Visualise the data (scatter of two chosen features, coloured by class).
3. Train one or more ML models with a hyperparameter sweep and compare results.

Course brief: `../88_Source_Material/2026/Projects/HCAI-project_01 (4).pdf` *(vault path; add copy under `docs/` when convenient)*.

## Scope decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Classification, regression, or both? | _TBD_ | Iris is classification. Starting classification-only keeps the UI simple; regression can be a stretch goal. |
| Problem-type detection | _TBD_ | Auto-detect (heuristic on the label column) vs. user picks in the upload form. |
| Supported algorithms | _TBD_ | Candidates: LogReg, kNN, Decision Tree, Random Forest, SVM. Pick 3 to start. |
| Hyperparameter sweep | _TBD_ | Single axis per algorithm (e.g. k for kNN, max_depth for DT) and plot score vs. hyperparameter. |
| Scoring | _TBD_ | Classification: accuracy, F1-macro, confusion matrix. Regression: MSE, R². |

## What the user controls vs. what's automatic (PDF Task 4)

This is the **human-centric** axis the professor will grade on. Fill in as you decide.

| Step | User | Automatic | Notes |
|------|------|-----------|-------|
| CSV upload | ✓ | | |
| Target column | _TBD_ | _TBD_ | Auto-pick last column per PDF, or let user confirm? |
| Train/test split ratio | _TBD_ | _TBD_ | |
| Model choice | _TBD_ | _TBD_ | |
| Hyperparameter range | _TBD_ | _TBD_ | |
| Scoring metric | _TBD_ | _TBD_ | |
| Model selection from sweep | _TBD_ | _TBD_ | |

## Django models (sketch)

```python
# project1/models.py  (draft — iterate)

class Dataset(models.Model):
    uploaded_at = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255)
    feature_names = models.JSONField()
    n_rows = models.IntegerField()
    problem_type = models.CharField(choices=[("classification", ...), ("regression", ...)])

class TrainingRun(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    algorithm = models.CharField(max_length=64)
    hyperparam_name = models.CharField(max_length=64)
    hyperparam_values = models.JSONField()
    test_size = models.FloatField(default=0.2)
    created_at = models.DateTimeField(auto_now_add=True)

class ModelResult(models.Model):
    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE)
    hyperparam_value = models.JSONField()
    train_score = models.FloatField()
    test_score = models.FloatField()
    confusion_matrix = models.JSONField(null=True, blank=True)  # classification only
```

## Views / URL plan

| URL | View | Purpose |
|-----|------|---------|
| `/project1/` | `index` | Landing, links to upload |
| `/project1/upload/` | `upload_dataset` | POST CSV → `Dataset` row |
| `/project1/dataset/<id>/` | `dataset_detail` | Preview table + scatter viz |
| `/project1/dataset/<id>/train/` | `configure_training` | Pick algorithm + hyperparam range |
| `/project1/run/<id>/` | `run_detail` | Results: sweep plot, confusion matrix, feature importance |

## Human-centric polish (grade differentiator)

Things we can add that go beyond "sklearn wrapper":

- [ ] Feature importance panel (permutation importance, works for any sklearn model)
- [ ] Metric switcher — re-rank models live by accuracy / F1 / etc.
- [ ] Show the confusion matrix, not just one score
- [ ] Clear UI callout for "what you chose" vs "what we chose for you"
- [ ] Brief written justification in this doc for each automation decision

## Open questions for the team

1. Do we want one big "train everything" button or one model at a time?
2. Persist datasets across sessions, or session-scoped?
3. One CSS file per app, or one global stylesheet?
