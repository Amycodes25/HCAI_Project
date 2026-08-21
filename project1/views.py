import base64
import csv
from io import BytesIO, StringIO

import matplotlib

# Matplotlib must use a non-interactive backend inside Django.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from django.shortcuts import render

from .models import Dataset, ModelResult, TrainingRun

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------

def load_dataframe_from_session(request):
    """Load the uploaded dataset stored in the Django session."""
    data_json = request.session.get("data")

    if not data_json:
        return None

    return pd.read_json(StringIO(data_json), orient="split")


# -------------------------------------------------------------------
# COLUMN HELPERS
# -------------------------------------------------------------------

def is_identifier_column(column_name):
    """
    Detect columns that mainly identify rows rather than describe them.

    Examples:
        Id
        PassengerId
        customer_id
        row_index
    """
    normalized = str(column_name).strip().lower().replace(" ", "")

    exact_identifier_names = {
        "id",
        "index",
        "rowindex",
        "passengerid",
        "customerid",
        "recordid",
        "sampleid",
    }

    return normalized in exact_identifier_names or normalized.endswith("_id")


def get_available_columns(df):
    """Return columns available for visualization and target selection."""
    return [
        column
        for column in df.columns
        if not is_identifier_column(column)
    ]


def choose_default_target(available_columns):
    """
    Suggest a likely target column.

    Common target names are preferred. If no common name is found,
    the last available column is used, matching the assignment format.
    """
    common_target_names = {
        "target",
        "label",
        "class",
        "outcome",
        "output",
        "species",
        "survived",
        "diagnosis",
        "quality",
        "income",
        "y",
    }

    for column in available_columns:
        normalized = str(column).strip().lower()

        if normalized in common_target_names:
            return column

    return available_columns[-1]


def get_model_feature_columns(df, target):
    """
    Select useful model-input columns.

    Very high-cardinality categorical columns are ignored because columns
    such as passenger names, ticket numbers, and cabin numbers often act
    like identifiers and can create many one-hot-encoded features.
    """
    usable_columns = []
    ignored_columns = []

    number_of_rows = len(df)

    for column in df.columns:
        if column == target:
            continue

        if is_identifier_column(column):
            ignored_columns.append(column)
            continue

        if (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
        ):
            unique_count = df[column].nunique(dropna=True)

            if number_of_rows > 0 and unique_count / number_of_rows > 0.50:
                ignored_columns.append(column)
                continue

        usable_columns.append(column)

    return usable_columns, ignored_columns


# -------------------------------------------------------------------
# VISUALIZATION
# -------------------------------------------------------------------

def create_scatter_plot(df, x_feature, y_feature, target):
    """Generate a scatter plot and return it as a Base64 string."""
    selected_columns = list(dict.fromkeys([x_feature, y_feature, target]))
    plot_data = df[selected_columns].dropna().copy()

    if plot_data.empty:
        raise ValueError(
            "No complete rows are available for the selected plot columns."
        )

    figure, axis = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    style_axis(axis)

    for index, (class_name, class_data) in enumerate(plot_data.groupby(target)):
        axis.scatter(
            class_data[x_feature],
            class_data[y_feature],
            label=str(class_name),
            s=34,
            alpha=0.85,
            color=SERIES[index % len(SERIES)],
            edgecolors="none",
        )

    axis.set_xlabel(str(x_feature))
    axis.set_ylabel(str(y_feature))
    axis.set_title(f"{x_feature} vs {y_feature}, grouped by {target}")

    if plot_data[target].nunique() <= 20:
        legend = axis.legend(title=str(target), frameon=False, fontsize=8)
        legend.get_title().set_color(MUTED)
        legend.get_title().set_fontsize(8)
        for text in legend.get_texts():
            text.set_color(INK)
    else:
        axis.text(
            0.02,
            0.98,
            "Legend hidden because the target has more than 20 values.",
            transform=axis.transAxes,
            verticalalignment="top",
        )

    buffer = BytesIO()
    figure.tight_layout()
    figure.savefig(
        buffer,
        format="png",
        dpi=130,
        bbox_inches="tight",
        facecolor=SURFACE,
    )
    plt.close(figure)

    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# The site palette, so figures do not look pasted in from another application.
SURFACE = "#ffffff"
INK = "#002429"
MUTED = "#58686e"
GRID = "#dfe5ee"
SERIES = ["#ce9ad5", "#62baf9", "#9999d6", "#f0a202", "#08765a", "#a62942"]


def style_axis(axis):
    axis.set_facecolor(SURFACE)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color=GRID, linewidth=0.6, linestyle="--", alpha=0.8)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.title.set_color(INK)
    return axis


def encode_figure(figure):
    buffer = BytesIO()
    figure.tight_layout()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight",
                   facecolor=SURFACE)
    plt.close(figure)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_sweep_plot(results, parameter_name, score_label):
    """The score against the swept hyperparameter.

    The table already carries the numbers; the plot carries the shape, which is
    what tells the user whether the model is under- or over-fitting.
    """
    labels = [str(row["parameter"]) for row in results]
    scores = [row["accuracy"] for row in results]

    figure, axis = plt.subplots(figsize=(7, 3.4), facecolor=SURFACE)
    style_axis(axis)
    axis.plot(labels, scores, "-o", color="#62baf9", linewidth=2, markersize=6)

    best = max(range(len(scores)), key=lambda i: scores[i])
    axis.scatter([labels[best]], [scores[best]], s=140, zorder=3,
                 facecolor="none", edgecolor="#a62942", linewidth=2)
    axis.annotate(f"best: {scores[best]:.1f}%", (labels[best], scores[best]),
                  textcoords="offset points", xytext=(0, 12),
                  ha="center", fontsize=9, color="#a62942")

    axis.set_xlabel(parameter_name)
    axis.set_ylabel(f"{score_label} (%)")
    axis.set_title(f"{score_label} against {parameter_name.lower()}",
                   fontsize=10, pad=10)
    return encode_figure(figure)


def create_confusion_matrix_plot(y_true, y_pred, labels):
    """Where the best model's mistakes actually fall.

    A single score says how often the model is wrong; this says what it confuses
    with what, which is the part a user can act on.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    size = max(3.4, min(7.0, len(labels) * 0.85))
    figure, axis = plt.subplots(figsize=(size + 0.6, size), facecolor=SURFACE)
    image = axis.imshow(matrix, cmap="Purples")
    axis.grid(False)

    axis.set_xticks(range(len(labels)), [str(label) for label in labels],
                    rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), [str(label) for label in labels])
    axis.set_xlabel("Predicted", color=MUTED)
    axis.set_ylabel("Actual", color=MUTED)
    axis.set_title("Confusion matrix, best configuration", fontsize=10,
                   color=INK, pad=10)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(GRID)

    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      fontsize=9,
                      color="white" if matrix[i, j] > threshold else "#233b3f")

    bar = figure.colorbar(image, ax=axis, shrink=0.72)
    bar.ax.tick_params(colors=MUTED, labelsize=7)
    bar.outline.set_edgecolor(GRID)
    return encode_figure(figure)


# -------------------------------------------------------------------
# PREPROCESSING
# -------------------------------------------------------------------

def create_preprocessor(X):
    """
    Build preprocessing for numerical and categorical features.

    Numerical columns:
        - fill missing values with the median
        - standardize values

    Categorical columns:
        - fill missing values with the most frequent category
        - convert categories using one-hot encoding
    """
    numerical_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X.columns
        if column not in numerical_columns
    ]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if numerical_columns:
        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError(
            "No usable numerical or categorical input features were found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor, numerical_columns, categorical_columns


# -------------------------------------------------------------------
# MODEL TRAINING AND HYPERPARAMETER SEARCH
# -------------------------------------------------------------------

# The models on offer, each with the single hyperparameter that is swept.
# One axis per model keeps the comparison readable: the point of the sweep is to
# show the user how a score moves with complexity, not to find the global best.
MODEL_SPECS = {
    "logistic": {
        "label": "Logistic Regression",
        "parameter_name": "C (inverse regularisation)",
        "values": [0.01, 0.1, 1, 10, 100],
        "build": lambda value: LogisticRegression(
            C=value, max_iter=2000, random_state=42
        ),
    },
    "tree": {
        "label": "Decision Tree",
        "parameter_name": "Maximum depth",
        "values": [1, 2, 3, 4, 5, None],
        "build": lambda value: DecisionTreeClassifier(
            max_depth=value, random_state=42
        ),
    },
    "knn": {
        "label": "K-Nearest Neighbours",
        "parameter_name": "Number of neighbours",
        "values": [1, 3, 5, 7, 9],
        "build": lambda value: KNeighborsClassifier(n_neighbors=value),
    },
}

# The brief asks who chooses the score, so the user does. The three differ in
# what they reward, and on an imbalanced dataset they disagree: accuracy can be
# high while a minority class is never predicted, which is exactly when the
# other two are worth having.
SCORERS = {
    "accuracy": {
        "label": "Accuracy",
        "hint": "Share of correct predictions. Misleading when classes are imbalanced.",
        "score": accuracy_score,
    },
    "f1_macro": {
        "label": "F1 (macro)",
        "hint": "Averages F1 over classes, weighting each class equally regardless of size.",
        "score": lambda y_true, y_pred: f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    },
    "balanced_accuracy": {
        "label": "Balanced accuracy",
        "hint": "Mean recall per class. Rewards getting rare classes right.",
        "score": balanced_accuracy_score,
    },
}


def evaluate_model_hyperparameters(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    score_name="accuracy",
):
    """Train one model family across its hyperparameter values.

    Returns the sweep, the best configuration, and the fitted best pipeline so
    the caller can report a confusion matrix for it.
    """
    if model_name not in MODEL_SPECS:
        raise ValueError("Unknown machine-learning model selected.")

    if score_name not in SCORERS:
        raise ValueError("Unknown score selected.")

    spec = MODEL_SPECS[model_name]
    scorer = SCORERS[score_name]["score"]

    preprocessor, numerical_columns, categorical_columns = (
        create_preprocessor(X_train)
    )

    values = spec["values"]
    if model_name == "knn":
        values = [value for value in values if value <= len(X_train)]

    results = []
    fitted = {}

    for value in values:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", spec["build"](value)),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        label = "Unlimited" if value is None else value
        results.append(
            {
                "parameter": label,
                "accuracy": round(scorer(y_test, predictions) * 100, 2),
            }
        )
        fitted[label] = pipeline

    if not results:
        raise ValueError(
            "No valid hyperparameter configurations could be trained."
        )

    best = max(results, key=lambda result: result["accuracy"])
    best_pipeline = fitted[best["parameter"]]

    return {
        "parameter_name": spec["parameter_name"],
        "results": results,
        "best_parameter": best["parameter"],
        "best_accuracy": best["accuracy"],
        "best_pipeline": best_pipeline,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }


# -------------------------------------------------------------------
# DJANGO VIEW
# -------------------------------------------------------------------

def index(request):
    context = {
        "title": "Project 1 — Supervised Learning Interface",
        "scorers": [
            {"name": name, "label": spec["label"], "hint": spec["hint"]}
            for name, spec in SCORERS.items()
        ],
        "selected_score": request.POST.get("score_name", "accuracy"),
    }

    # Clear the saved dataset and start a fresh session state.
    if request.method == "POST" and request.POST.get("action") == "reset":
        request.session.pop("data", None)
        request.session.pop("data_filename", None)
        request.session.pop("dataset_id", None)
        request.session.modified = True
        return render(request, "project1/index.html", context)

    try:
        # =========================================================
        # ACTION 1: UPLOAD CSV
        # =========================================================
        if (
            request.method == "POST"
            and request.POST.get("action") == "upload"
        ):
            csv_file = request.FILES.get("csv_file")

            if not csv_file:
                raise ValueError("Please choose a CSV file.")

            if not csv_file.name.lower().endswith(".csv"):
                raise ValueError("Only CSV files are supported.")

            # Read the header separately before pandas gets a chance to
            # de-duplicate it. read_csv silently renames a repeated column to
            # "name.1", so checking df.columns afterwards can never detect the
            # problem, and the interface would then offer the user a column
            # name their file does not contain.
            first_line = csv_file.readline().decode("utf-8-sig", errors="replace")
            csv_file.seek(0)

            header_line = first_line.splitlines()[0] if first_line else ""
            header = next(csv.reader([header_line]), [])
            header = [name.strip() for name in header]

            duplicate_names = sorted(
                {name for name in header if header.count(name) > 1}
            )

            if duplicate_names:
                raise ValueError(
                    "The CSV contains duplicate column names: "
                    + ", ".join(map(str, duplicate_names))
                )

            df = pd.read_csv(csv_file)

            if df.empty:
                raise ValueError("The uploaded CSV file is empty.")

            request.session["data"] = df.to_json(orient="split")

            # A file input always renders empty after the page re-renders, so
            # without remembering the name the interface silently loses track of
            # which dataset is loaded and looks as though nothing happened.
            request.session["data_filename"] = csv_file.name

            dataset = Dataset.objects.create(
                filename=csv_file.name,
                n_rows=len(df),
                n_columns=len(df.columns),
                column_names=[str(c) for c in df.columns],
            )
            request.session["dataset_id"] = dataset.pk

        else:
            df = load_dataframe_from_session(request)

        # =========================================================
        # PREPARE DATASET INFORMATION
        # =========================================================
        if df is not None:
            # Every sweep run against this dataset, so two model families can be
            # compared with each other and not only within one sweep.
            context["previous_runs"] = list(
                TrainingRun.objects.filter(
                    dataset_id=request.session.get("dataset_id")
                ).select_related("dataset")[:12]
            )

            available_columns = get_available_columns(df)

            if not available_columns:
                raise ValueError(
                    "No usable columns were found in the dataset."
                )

            numeric_columns = [
                column
                for column in available_columns
                if pd.api.types.is_numeric_dtype(df[column])
            ]

            default_target = choose_default_target(available_columns)

            context["data_filename"] = request.session.get("data_filename")
            context["data_preview"] = df.head().to_html(
                classes="data-table",
                index=False,
            )
            context["columns"] = available_columns
            context["numeric_columns"] = numeric_columns
            context["default_target"] = default_target
            context["row_count"] = len(df)
            context["column_count"] = len(df.columns)

        # =========================================================
        # ACTION 2: SCATTER PLOT
        # =========================================================
        if (
            request.method == "POST"
            and request.POST.get("action") == "plot"
        ):
            if df is None:
                raise ValueError("Please upload a dataset first.")

            x_feature = request.POST.get("x_feature")
            y_feature = request.POST.get("y_feature")
            plot_target = request.POST.get("plot_target")

            if not x_feature or not y_feature or not plot_target:
                raise ValueError(
                    "Please select two features and a target."
                )

            if x_feature == y_feature:
                raise ValueError(
                    "Please select two different numerical features."
                )

            if plot_target in {x_feature, y_feature}:
                raise ValueError(
                    "The colour-by target must be different from "
                    "the X-axis and Y-axis features."
                )

            if (
                x_feature not in numeric_columns
                or y_feature not in numeric_columns
            ):
                raise ValueError(
                    "Scatter-plot axes must contain numerical values."
                )

            if plot_target not in available_columns:
                raise ValueError(
                    "The selected plot target is invalid."
                )

            context["scatter_plot"] = create_scatter_plot(
                df=df,
                x_feature=x_feature,
                y_feature=y_feature,
                target=plot_target,
            )
            context["selected_x"] = x_feature
            context["selected_y"] = y_feature
            context["selected_plot_target"] = plot_target

        # =========================================================
        # ACTION 3: TRAIN AND TUNE MODEL
        # =========================================================
        if (
            request.method == "POST"
            and request.POST.get("action") == "train"
        ):
            if df is None:
                raise ValueError("Please upload a dataset first.")

            target = request.POST.get("target")
            model_name = request.POST.get("model_name")

            try:
                test_size = float(
                    request.POST.get("test_size", "0.2")
                )
            except ValueError as error:
                raise ValueError(
                    "The selected test size is invalid."
                ) from error

            if target not in available_columns:
                raise ValueError(
                    "Please select a valid target column."
                )

            if target not in df.columns:
                raise ValueError(
                    "The target column does not exist."
                )

            feature_columns, ignored_columns = get_model_feature_columns(
                df=df,
                target=target,
            )

            if not feature_columns:
                raise ValueError(
                    "No usable input features remain after removing "
                    "identifier and high-cardinality columns."
                )

            model_data = df[
                feature_columns + [target]
            ].dropna(subset=[target])

            if model_data.empty:
                raise ValueError(
                    "No rows remain after removing missing target values."
                )

            X = model_data[feature_columns]
            y = model_data[target]

            if y.nunique() < 2:
                raise ValueError(
                    "The target must contain at least two classes."
                )

            unique_target_ratio = y.nunique() / len(y)

            if unique_target_ratio > 0.80:
                raise ValueError(
                    "The selected target contains mostly unique values. "
                    "It may be an identifier rather than a class label."
                )

            smallest_class_size = y.value_counts().min()
            stratify_value = y if smallest_class_size >= 2 else None

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify_value,
            )

            score_name = request.POST.get("score_name", "accuracy")
            if score_name not in SCORERS:
                score_name = "accuracy"

            evaluation = evaluate_model_hyperparameters(
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                score_name=score_name,
            )

            best_pipeline = evaluation["best_pipeline"]
            class_labels = sorted(y.unique().tolist(), key=str)

            context["score_name"] = score_name
            context["score_label"] = SCORERS[score_name]["label"]
            context["score_hint"] = SCORERS[score_name]["hint"]
            context["sweep_plot"] = create_sweep_plot(
                evaluation["results"],
                evaluation["parameter_name"],
                SCORERS[score_name]["label"],
            )
            context["confusion_plot"] = create_confusion_matrix_plot(
                y_test, best_pipeline.predict(X_test), class_labels
            )

            dataset = Dataset.objects.filter(
                pk=request.session.get("dataset_id")
            ).first()

            if dataset is not None:
                run = TrainingRun.objects.create(
                    dataset=dataset,
                    algorithm=model_name,
                    algorithm_label=MODEL_SPECS[model_name]["label"],
                    parameter_name=evaluation["parameter_name"],
                    target=target,
                    test_size=test_size,
                    score_name=score_name,
                    score_label=SCORERS[score_name]["label"],
                    best_parameter=str(evaluation["best_parameter"]),
                    best_score=evaluation["best_accuracy"],
                    training_rows=len(X_train),
                    testing_rows=len(X_test),
                )
                ModelResult.objects.bulk_create([
                    ModelResult(
                        run=run,
                        parameter_value=str(row["parameter"]),
                        score=row["accuracy"],
                    )
                    for row in evaluation["results"]
                ])

            context["selected_model"] = MODEL_SPECS[model_name]["label"]
            context["selected_model_value"] = model_name
            context["selected_target"] = target
            context["selected_test_size"] = str(test_size)
            context["parameter_name"] = evaluation["parameter_name"]
            context["hyperparameter_results"] = evaluation["results"]
            context["best_parameter"] = evaluation["best_parameter"]
            context["best_accuracy"] = evaluation["best_accuracy"]
            context["used_features"] = feature_columns
            context["ignored_features"] = ignored_columns
            context["numerical_model_features"] = (
                evaluation["numerical_columns"]
            )
            context["categorical_model_features"] = (
                evaluation["categorical_columns"]
            )
            context["training_rows"] = len(X_train)
            context["testing_rows"] = len(X_test)

    except (
        ValueError,
        KeyError,
        TypeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        context["error"] = str(error)

    return render(
        request,
        "project1/index.html",
        context,
    )