# HCAI Group Project — SoSe 2026

Coursework for **Human-Centric Artificial Intelligence**, TUHH.

All four course projects live in this single Django project, one app per project,
reachable from a shared launch page at the site root.

Upstream skeleton: <https://github.com/ppaamm/HCAI-PBL>

## Team

| Name | Matriculation |
|------|---------------|
| Darren | _to add_ |
| Chandana | _to add_ |
| Divya | _to add_ |
| Amritha | _to add_ |

The names shown on the home page come from `home/views.py`, not from this table.
Update both together.

## Projects

| App | URL | Project | Status |
|-----|-----|---------|--------|
| `project1` | `/project1/` | Supervised Learning Interface | Working |
| `project2` | `/project2/` | Explainability | Placeholder — port in progress |
| `learning_to_defer` | `/project3/` | Active Learning for Learning-to-Defer | Working |
| `project4` | `/project4/` | Preference Elicitation | Placeholder — in progress |

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

## Development

`requirements-dev.txt` adds `datasets`, which is needed only to regenerate the
Project 3 artifacts from scratch:

```bash
pip install -r requirements-dev.txt
```

The site itself never downloads AG News — Project 3 serves from the committed
artifacts in `learning_to_defer/artifacts/`.

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
