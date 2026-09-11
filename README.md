# HCAI Group Project — SoSe 2026

Coursework for **Human-Centric Artificial Intelligence**, TUHH.

All four course projects live in this single Django project, one app per project,
reachable from a shared launch page at the site root.

Upstream skeleton: <https://github.com/ppaamm/HCAI-PBL>

## Team

| Name | Matriculation |
|------|---------------|
| Darren Noronha | 637482 |
| Chandana Putta | 637789 |
| Divya Chidananda | 672434 |
| Amritha Subramanian | 641269 |

The names shown on the home page come from `home/views.py`, not from this table.
Update both together.

## Projects

| App | URL | Project | Status |
|-----|-----|---------|--------|
| `project1` | `/project1/` | Supervised Learning Interface | Working |
| `project2` | `/project2/` | Explainability | Working |
| `learning_to_defer` | `/project3/` | Active Learning for Learning-to-Defer | Working |
| `project4` | `/project4/` | Preference Elicitation | Working |

## Setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Amycodes25/HCAI_Project.git
cd HCAI_Project

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>.

**`migrate` is not optional.** Project 1 keeps the uploaded dataset in the Django
session, the session backend is the database, and `db.sqlite3` is deliberately not
committed. Skipping `migrate` makes the first CSV upload fail with
`no such table: django_session`.

Run `migrate` again after pulling changes to `project1/models.py` (or any new file
under `project1/migrations/`) — Django only applies a migration once per database,
so an out-of-date `db.sqlite3` will otherwise fail with a `no such column` error the
first time a run tries to record a field that migration adds.

## Development

`requirements-dev.txt` adds the two packages needed only to regenerate
committed build products:

```bash
pip install -r requirements-dev.txt
```

Nothing the site serves depends on them, and it never downloads anything at
runtime. Every dataset and model it needs is committed.

| Command | Rebuilds | Needs |
|---------|----------|-------|
| `manage.py build_project2` | Project 2's 69 fitted models | runtime only |
| `manage.py run_project4_pilot` | Project 4's simulated pilot | runtime only |
| `manage.py build_project4_report` | Project 4's report PDF | `reportlab` |
| `learning_to_defer/services/*.py` | Project 3's artifacts | `datasets` |

## Tests

```bash
python manage.py test
```

67 tests across Projects 1 and 4, covering the model code and the views.

## Conventions

Set by the course brief; keep to them when adding an app.

- Templates live at `templates/<app>/<file>.html`.
- Per-app stylesheets live at `static/<app>/style.css`, in addition to the shared
  `static/style.css`.
- Templates extend `templates/base.html`.
- Each app defines `app_name` in its `urls.py`, so URLs are referenced namespaced,
  as `{% url 'project1:index' %}`.

## Repository layout

```
pbl/                  project settings and root URL configuration
home/                 launch page listing the group and the four projects
demos/                course-provided examples of file upload and plotting
project1/             Project 1
project2/             Project 2
learning_to_defer/    Project 3
project4/             Project 4
static/               shared stylesheet
templates/base.html   shared page shell
media/                generated plots and uploads (not committed)
```
