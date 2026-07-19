# HCAI-Project-SoSe26

Coursework for **Human-Centric Artificial Intelligence**, TUHH SoSe 2026.

All projects for the course live in this single Django project, one app per
project, accessible from a shared home launch page.

- **Upstream skeleton:** https://github.com/ppaamm/HCAI-PBL
- **Course brief (Project 1):** see `docs/project1-brief.pdf` *(add after first sync)*

## Team

| Name | Matriculation |
|------|---------------|
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |

Update this table and `home/views.py` together — both are displayed on the home page.

## Setup

Requires Python 3.11+.

```bash
# 1. Clone
git clone https://github.com/darrennoronha75/HCAI-Project-SoSe26.git
cd HCAI-Project-SoSe26

# 2. Virtualenv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install deps
pip install -r requirements.txt

# 4. Apply initial migrations (SQLite)
python manage.py migrate

# 5. Run the dev server
python manage.py runserver
# → http://127.0.0.1:8000/home/
```

## Notebooks

We iterate on concepts and ML logic in Jupyter first, then port stable code
into `project1/ml/`. Run from the repo root:

```bash
jupyter notebook notebooks/
```

Notebooks are committed. Clear outputs before committing large runs so diffs stay readable.

## Repository layout

```
HCAI-Project-SoSe26/
├── manage.py
├── pbl/                    # Django project (settings, root urls)
├── home/                   # Launch page, group roster, project links
├── demos/                  # Reference patterns (file upload, matplotlib → PNG)
├── project1/               # Supervised learning interface (Project 1)
│   └── ml/                 # Pure-Python ML logic (testable, importable in notebooks)
├── notebooks/              # Exploratory + prototype notebooks
├── docs/                   # Design decisions, write-ups for submission
├── static/                 # Global CSS shared across apps
├── templates/              # Global base template
├── media/                  # Uploaded CSVs + generated plots (git-ignored contents)
├── requirements.txt
└── .gitignore
```

## Workflow

- One branch: **`main`**. No feature branches.
- Pull → work → commit → push. Small commits, often.
- Discuss in notebooks and `docs/` before writing UI code.
- `upstream` remote is set to the course skeleton — `git fetch upstream` if the
  course authors publish fixes.

## Project status

- [x] P0: Skeleton imported, team scaffolding in place
- [ ] P1: Supervised learning interface — see `docs/project1-design.md`
