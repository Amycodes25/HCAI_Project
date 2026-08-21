# Project 1 — Supervised Learning Interface: design notes

A record of the decisions behind the app, and why each one went the way it did.
The open questions this file started with have been settled; where a decision
was a judgement call rather than a necessity, the reasoning is given so it can
be argued with.

## Problem statement

A Django app inside the HCAI project that lets a user:

1. Upload a CSV, feature names in row 0 and the label in the last column.
2. Visualise it — a scatter of two chosen features, coloured by class.
3. Train a model across a hyperparameter sweep and compare the results.

## Scope decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Classification, regression, or both? | **Classification only** | The brief permits choosing. Supporting both would double the branching in a single view for a second problem type the brief only mentions in passing. The app states the limit rather than guessing and failing confusingly. |
| Problem-type detection | **Neither: the scope is declared** | Auto-detection would silently mislead on an integer-coded target, and asking the user to declare a type the app cannot honour either way is worse. The interface says classification. |
| Supported algorithms | **Logistic regression, decision tree, k-NN** | Three families that differ in what they can express: a linear boundary, axis-aligned rules, and a local one. That contrast is the point of offering a choice at all. |
| Hyperparameter sweep | **One axis per family** | The sweep exists to show the user how the score moves with complexity, not to find a global optimum. A grid over several axes would produce a table nobody reads. |
| Scoring | **User picks: accuracy, F1-macro, balanced accuracy** | The brief asks explicitly who chooses the score. On imbalanced data these disagree — accuracy can look high while a minority class is never predicted — and only the user knows which mistake costs more. |

## What the user controls, and what the app decides

This is the human-centric axis of Task 4, and it is surfaced in the interface
itself rather than only recorded here.

| Step | Who | Why |
|------|-----|-----|
| Dataset | User | |
| Target column | User | Suggested from the column name, overridable. |
| Train/test split | User | The trade-off between fitting and measuring is a judgement. |
| Model family | User | The families differ in expressiveness and readability. |
| Score | User | See above. |
| Hyperparameter range | App | Choosing a range well needs knowledge of the model that the interface should not demand of its user. |
| Preprocessing | App | Median imputation, scaling, one-hot encoding — mechanical once the column types are known. |
| Identifier columns | App | Detected and dropped, then listed in the results so the decision stays visible and contestable. |
| Model selection from the sweep | App | The best configuration under the user's chosen score. The whole sweep is shown, so the choice can be checked. |

## Structure

One URL and one view, with the action carried in a hidden field. The four
actions — upload, plot, train, reset — share the uploaded dataset held in the
session, and each renders the same page with more of it filled in.

The alternative considered was a URL per step, with `Dataset`, `TrainingRun`
and `ModelResult` persisted so that a run could be linked to and runs compared.
That is the better structure for an app that outlives a session; it was not
adopted because nothing in the brief needs a run to survive one, and the cost
is a larger surface for no user-visible gain. The consequence is accepted
knowingly: a trained run has no address, and a refresh re-submits.

Forms are posted with `fetch` and the page is swapped in place, so an action
does not throw the page away — but they remain ordinary POST forms and work
without JavaScript.

## Presentation

- The results table reports every configuration in the sweep, with the best row
  highlighted.
- A sweep plot shows the shape the table cannot: whether the model is under- or
  over-fitting across the range.
- A confusion matrix for the best configuration shows *what is confused with
  what*, which is the part a user can act on. A single score cannot say that.

## Known gaps

- Regression is unsupported, by the decision above.
- `project1/ml/` is empty. The extraction it was created for did not happen, and
  its docstring now says so rather than claiming to be the source of truth.
- Feature importance is not shown. It was on the original wish list; it is
  genuinely useful and simply was not reached.
