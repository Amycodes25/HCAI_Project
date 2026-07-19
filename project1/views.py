import base64
from io import BytesIO, StringIO

import matplotlib

# Matplotlib must use a non-interactive backend inside Django.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from django.shortcuts import render

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

    figure, axis = plt.subplots(figsize=(9, 5.5))

    for class_name, class_data in plot_data.groupby(target):
        axis.scatter(
            class_data[x_feature],
            class_data[y_feature],
            label=str(class_name),
            alpha=0.70,
            edgecolors="none",
        )

    axis.set_xlabel(str(x_feature))
    axis.set_ylabel(str(y_feature))
    axis.set_title(f"{x_feature} vs {y_feature}, grouped by {target}")

    if plot_data[target].nunique() <= 20:
        axis.legend(title=str(target))
    else:
        axis.text(
            0.02,
            0.98,
            "Legend hidden because the target has more than 20 values.",
            transform=axis.transAxes,
            verticalalignment="top",
        )

    axis.grid(alpha=0.25)

    buffer = BytesIO()
    figure.tight_layout()
    figure.savefig(
        buffer,
        format="png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(figure)

    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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

def evaluate_model_hyperparameters(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Train the selected model using multiple hyperparameter values."""
    results = []

    preprocessor, numerical_columns, categorical_columns = (
        create_preprocessor(X_train)
    )

    if model_name == "logistic":
        parameter_name = "C"
        parameter_values = [0.01, 0.1, 1, 10, 100]

        for value in parameter_values:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            C=value,
                            max_iter=2000,
                            random_state=42,
                        ),
                    ),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            results.append(
                {
                    "parameter": value,
                    "accuracy": round(accuracy * 100, 2),
                }
            )

    elif model_name == "tree":
        parameter_name = "Maximum depth"
        parameter_values = [1, 2, 3, 4, 5, None]

        for value in parameter_values:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        DecisionTreeClassifier(
                            max_depth=value,
                            random_state=42,
                        ),
                    ),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            results.append(
                {
                    "parameter": "Unlimited" if value is None else value,
                    "accuracy": round(accuracy * 100, 2),
                }
            )

    elif model_name == "knn":
        parameter_name = "Number of neighbours"
        parameter_values = [1, 3, 5, 7, 9]
        parameter_values = [
            value
            for value in parameter_values
            if value <= len(X_train)
        ]

        for value in parameter_values:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        KNeighborsClassifier(n_neighbors=value),
                    ),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            results.append(
                {
                    "parameter": value,
                    "accuracy": round(accuracy * 100, 2),
                }
            )

    else:
        raise ValueError("Unknown machine-learning model selected.")

    if not results:
        raise ValueError(
            "No valid hyperparameter configurations could be trained."
        )

    best_result = max(results, key=lambda result: result["accuracy"])

    return {
        "parameter_name": parameter_name,
        "results": results,
        "best_parameter": best_result["parameter"],
        "best_accuracy": best_result["accuracy"],
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }


# -------------------------------------------------------------------
# DJANGO VIEW
# -------------------------------------------------------------------

def index(request):
    context = {
        "title": "Project 1 — Supervised Learning Interface",
    }

    # Clear the saved dataset and start a fresh session state.
    if request.method == "POST" and request.POST.get("action") == "reset":
        request.session.pop("data", None)
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

            df = pd.read_csv(csv_file)

            if df.empty:
                raise ValueError("The uploaded CSV file is empty.")

            if df.columns.duplicated().any():
                duplicate_names = df.columns[
                    df.columns.duplicated()
                ].tolist()

                raise ValueError(
                    "The CSV contains duplicate column names: "
                    + ", ".join(map(str, duplicate_names))
                )

            request.session["data"] = df.to_json(orient="split")

        else:
            df = load_dataframe_from_session(request)

        # =========================================================
        # PREPARE DATASET INFORMATION
        # =========================================================
        if df is not None:
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

            evaluation = evaluate_model_hyperparameters(
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )

            model_display_names = {
                "logistic": "Logistic Regression",
                "tree": "Decision Tree",
                "knn": "K-Nearest Neighbours",
            }

            context["selected_model"] = model_display_names[model_name]
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