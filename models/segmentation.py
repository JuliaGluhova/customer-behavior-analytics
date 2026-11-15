
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


FEATURES = [
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


def prepare_data(df):
    """Подготовка и стандартизация признаков"""
    X = df[FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def calculate_profiles(segmentation_df):
    """Расчет характеристик полученных сегментов"""
    profiles = (
        segmentation_df.groupby("Сегмент")
        .agg(
            Количество_сессий=("Revenue", "size"),
            Количество_покупок=("Revenue", "sum"),
            Конверсия=("Revenue", "mean"),
            Среднее_число_товарных_страниц=("ProductRelated", "mean"),
            Среднее_значение_PageValues=("PageValues", "mean"),
            Средний_BounceRate=("BounceRates", "mean"),
            Средний_ExitRate=("ExitRates", "mean")
        )
        .reset_index()
    )

    profiles["Конверсия"] *= 100

    return profiles


def evaluate_kmeans(X_scaled, min_clusters=2, max_clusters=6):
    """Оценка K-means для разного количества кластеров"""
    results = []

    for k in range(min_clusters, max_clusters + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(3000, len(X_scaled)),
            random_state=42
        )

        results.append({
            "Количество кластеров": k,
            "Инерция": model.inertia_,
            "Silhouette Score": score
        })

    return pd.DataFrame(results)


def perform_segmentation(df, n_clusters=3):
    """Кластеризация методом K-means"""
    X_scaled, scaler = prepare_data(df)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    segmentation_df = df.copy()
    segmentation_df["Сегмент"] = kmeans.fit_predict(X_scaled)

    cluster_profiles = calculate_profiles(segmentation_df)

    return segmentation_df, cluster_profiles, kmeans, scaler


def calculate_dbscan_silhouette(X_scaled, labels):
    """Расчет Silhouette Score без шумовых объектов"""
    mask = labels != -1

    X_clean = X_scaled[mask]
    labels_clean = labels[mask]

    unique_labels = np.unique(labels_clean)

    if len(unique_labels) < 2 or len(unique_labels) >= len(labels_clean):
        return None

    # ограничиваем размер выборки для ускорения расчета
    if len(X_clean) > 2000:
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(X_clean),
            size=2000,
            replace=False
        )

        X_sample = X_clean[indices]
        labels_sample = labels_clean[indices]

        if len(np.unique(labels_sample)) < 2:
            return None

        if len(np.unique(labels_sample)) >= len(labels_sample):
            return None
    else:
        X_sample = X_clean
        labels_sample = labels_clean

    try:
        return silhouette_score(X_sample, labels_sample)
    except ValueError:
        return None


def evaluate_dbscan(X_scaled, min_samples=10):
    """Подбор eps для DBSCAN на основе расстояний до соседей"""
    neighbors = NearestNeighbors(
        n_neighbors=min_samples,
        n_jobs=-1
    )

    neighbors.fit(X_scaled)

    distances, _ = neighbors.kneighbors(X_scaled)

    # расстояния до ближайшего соседа с учетом min_samples
    k_distances = np.sort(distances[:, -1])

    # проверяем несколько значений eps, полученных из распределения расстояний
    quantiles = [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.96, 0.98]

    eps_values = np.unique(
        np.round(np.quantile(k_distances, quantiles), 3)
    )

    results = []

    for eps in eps_values:
        model = DBSCAN(
            eps=float(eps),
            min_samples=min_samples,
            n_jobs=-1
        )

        labels = model.fit_predict(X_scaled)

        clusters = set(labels) - {-1}
        n_clusters = len(clusters)
        noise_share = (labels == -1).mean() * 100

        score = calculate_dbscan_silhouette(
            X_scaled,
            labels
        )

        results.append({
            "eps": float(eps),
            "Количество кластеров": n_clusters,
            "Доля шума, %": noise_share,
            "Silhouette Score": score
        })

    return pd.DataFrame(results)


def find_best_dbscan(X_scaled, min_samples=10):
    """Выбор лучшего eps по Silhouette Score"""
    evaluation_df = evaluate_dbscan(
        X_scaled,
        min_samples=min_samples
    )

    valid_results = evaluation_df[
        (evaluation_df["Количество кластеров"] >= 2)
        & (evaluation_df["Silhouette Score"].notna())
        & (evaluation_df["Доля шума, %"] <= 50)
    ]

    if valid_results.empty:
        return None, evaluation_df

    best_row = valid_results.loc[
        valid_results["Silhouette Score"].idxmax()
    ]

    return float(best_row["eps"]), evaluation_df


def perform_dbscan(df, eps=2.5, min_samples=10):
    """Кластеризация методом DBSCAN"""
    X_scaled, scaler = prepare_data(df)

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1
    )

    labels = dbscan.fit_predict(X_scaled)

    segmentation_df = df.copy()
    segmentation_df["Сегмент"] = labels

    cluster_profiles = calculate_profiles(segmentation_df)

    n_clusters = len(set(labels) - {-1})
    noise_share = (labels == -1).mean() * 100

    score = calculate_dbscan_silhouette(
        X_scaled,
        labels
    )

    dbscan_metrics = {
        "Количество кластеров": n_clusters,
        "Доля шума, %": noise_share,
        "Silhouette Score": score
    }

    return (
        segmentation_df,
        cluster_profiles,
        dbscan,
        scaler,
        dbscan_metrics
    )
