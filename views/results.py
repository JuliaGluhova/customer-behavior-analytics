
import pandas as pd
import streamlit as st

from models.segmentation import perform_segmentation
from models.prediction import train_models


@st.cache_resource
def get_final_analysis(df):

    segmented_df, profiles_df, kmeans_model, scaler = (
        perform_segmentation(df, n_clusters=3)
    )

    results_df, models, y_test, predictions, probabilities = (
        train_models(df)
    )

    return segmented_df, results_df


def render_results(df):

    st.title("Итоги исследования")

    st.write(
        "Основные результаты моделирования поведения "
        "посетителей интернет-магазина."
    )

    if "Revenue" not in df.columns:
        st.error("В данных отсутствует целевая переменная Revenue.")
        return

    with st.spinner("Подготовка итоговых результатов..."):
        segmented_df, results_df = get_final_analysis(df)

    # основные показатели
    st.subheader("Паспорт исследования")

    total_sessions = len(df)
    total_features = len(df.columns) - 1
    total_clusters = segmented_df["Сегмент"].nunique()
    conversion = df["Revenue"].astype(int).mean() * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Сессии", f"{total_sessions:,}".replace(",", " "))
    col2.metric("Признаки", total_features)
    col3.metric("Сегменты", total_clusters)
    col4.metric("Конверсия", f"{conversion:.2f}%")

    st.caption(
        "Источник данных: Online Shoppers Purchasing Intention Dataset"
    )

    st.divider()

    # результаты сегментации
    st.subheader("1. Сегментация потребителей")

    cluster_results = (
        segmented_df.groupby("Сегмент")
        .agg(
            Посетителей=("Revenue", "count"),
            Конверсия=("Revenue", "mean")
        )
        .reset_index()
    )

    cluster_results["Доля посетителей, %"] = (
        cluster_results["Посетителей"] / total_sessions * 100
    )

    cluster_results["Конверсия"] *= 100

    cluster_results = cluster_results[
        ["Сегмент", "Посетителей", "Доля посетителей, %", "Конверсия"]
    ]

    st.dataframe(
        cluster_results.style.format({
            "Доля посетителей, %": "{:.2f}",
            "Конверсия": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    best_cluster = cluster_results.loc[
        cluster_results["Конверсия"].idxmax()
    ]

    st.write(
        f"Наибольшая конверсия обнаружена в сегменте "
        f"**{int(best_cluster['Сегмент'])}** — "
        f"{best_cluster['Конверсия']:.2f}%."
    )

    st.divider()

    # сравнение моделей
    st.subheader("2. Сравнение моделей прогнозирования")

    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    best_model = results_df.loc[
        results_df["F1-score"].idxmax()
    ]

    st.success(
        f"Лучшая модель по F1-score: **{best_model['Модель']}** "
        f"({best_model['F1-score']:.4f})."
    )

    st.divider()

    # итоговые выводы
    st.subheader("3. Итог исследования")

    st.markdown(
        f"""
        - **Сегментация:** алгоритм K-means выделил
          {total_clusters} группы посетителей с различными
          характеристиками поведения.

        - **Прогнозирование:** среди двух исследованных моделей
          наибольшее значение F1-score показала
          **{best_model['Модель']}**.

        - **Практический результат:** сегментация и прогнозирование
          позволяют выявлять группы посетителей с различной
          склонностью к совершению покупки.
        """
    )
