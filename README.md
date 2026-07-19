# HCAI Project 3: Active Learning for Learning-to-Defer

This repository extends the course-provided Django skeleton. The current checkpoint implements the Task 1 application structure and the reproducible AG News baseline.

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train Task 1

```bash
python -m learning_to_defer.services.train_baseline
```

The command downloads the official AG News splits, trains TF-IDF plus Logistic Regression on all 120,000 training articles, evaluates on the 7,600 test articles, and saves the pipeline and measured metrics under `learning_to_defer/artifacts/`.

## Run Django

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in a browser.

## Tests

```bash
python manage.py test
```
