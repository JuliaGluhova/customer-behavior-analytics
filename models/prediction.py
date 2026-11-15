
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# числовые признаки
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

# категориальные признаки
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]


def create_preprocessor():
    # подготовка числовых и категориальных признаков
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )


def create_model_configs():
    # модели для сравнения
    return {
        "Логистическая регрессия": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }


def train_models(df):

    # проверка необходимых столбцов
    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ["Revenue"]
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "В наборе данных отсутствуют столбцы: "
            + ", ".join(missing_columns)
        )

    # разделение признаков и целевой переменной
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df["Revenue"].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "Для прогнозирования должны присутствовать оба класса Revenue."
        )

    # разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model_configs = create_model_configs()

    fitted_models = {}
    predictions = {}
    probabilities = {}
    results = []

    # обучение и оценка моделей
    for name, classifier in model_configs.items():

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fitted_models[name] = pipeline
        predictions[name] = y_pred
        probabilities[name] = y_prob

        results.append({
            "Модель": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "F1-score": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

    results_df = pd.DataFrame(results)

    return (
        results_df,
        fitted_models,
        y_test,
        predictions,
        probabilities
    )

