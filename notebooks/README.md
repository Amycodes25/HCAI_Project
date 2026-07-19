# Notebooks

Exploratory and prototype space. **Iterate here first, then port stable code into `project1/ml/`.**

## Conventions

- Number prefix by order: `01_`, `02_`, ... so the sort order is the reading order.
- One notebook = one concern. Split rather than grow.
- Clear outputs before committing any notebook larger than ~1 MB. Large plots bloat diffs.
- When a function stabilises, move it to `project1/ml/<module>.py` and import it back into the notebook. The notebook stays as narrative; the module becomes the source of truth.

## Planned notebooks

| # | Notebook | Purpose | Feeds into |
|---|----------|---------|------------|
| 01 | `01_iris_exploration.ipynb` | Load iris, inspect, pair plots, decide what a **user** would want to see | `project1/ml/loaders.py`, `project1/ml/visualize.py` |
| 02 | `02_training_pipeline.ipynb` | End-to-end sklearn: split → fit multiple models → sweep → score → confusion matrix → permutation importance | `project1/ml/train.py` |
| 03 | _(add as needed)_ | | |

## Running

```bash
# from repo root, with venv active
jupyter notebook notebooks/
```
