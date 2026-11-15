
import pandas as pd
import plotly.express as px
import streamlit as st

from models.segmentation import perform_segmentation
from models.prediction import (
    train_models,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)
from sklearn.model_selection import train_test_split


def render_scenario_analysis(df):

    st.title("Сценарный анализ поведения потребителей")

    st.write(
        "Сравнение сегментов посетителей интернет-магазина "
        "по характеристикам поведения и прогнозируемой "
        "вероятности совершения покупки."
    )

    st.info(
        "Для анализа используются три сегмента, выделенные "
        "алгоритмом K-means. Для каждого посетителя рассчитывается вероятность покупки. "
        "Полученные значения усредняются внутри соответствующего сегмента."
    )

    # проверка наличия необходимых данных
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
        st.error(
            "Недостаточно данных для сценарного анализа. "
            "Отсутствуют столбцы: "
            + ", ".join(missing_columns)
        )
        return

    # сегментация посетителей
    with st.spinner("Формирование сегментов посетителей..."):
        (
            segmented_df,
            profiles_df,
            kmeans_model,
            scaler
        ) = perform_segmentation(df, n_clusters=3)

    if "Сегмент" not in segmented_df.columns:
        st.error(
            "В результатах сегментации отсутствует столбец «Сегмент»."
        )
        return

    # получение моделей прогнозирования
    models = st.session_state.get("prediction_models")

    if models is None:
        with st.spinner("Подготовка моделей прогнозирования..."):
            (
                results_df,
                models,
                y_test,
                predictions,
                probabilities
            ) = train_models(df)

        st.session_state["prediction_models"] = models

    # формирование описания сегментов
    st.divider()
    st.subheader("1. Характеристики сегментов")

    cluster_stats = []

    for cluster in sorted(segmented_df["Сегмент"].unique()):

        cluster_data = segmented_df[
            segmented_df["Сегмент"] == cluster
        ]

        cluster_stats.append({
            "Сегмент": int(cluster),
            "Количество посетителей": len(cluster_data),
            "Конверсия, %": cluster_data["Revenue"].mean() * 100,
            "Среднее число товарных страниц":
                cluster_data["ProductRelated"].mean(),
            "Среднее время на страницах товаров":
                cluster_data["ProductRelated_Duration"].mean(),
            "Средний показатель отказов":
                cluster_data["BounceRates"].mean()
        })

    stats_df = pd.DataFrame(cluster_stats)

    # определение уровня активности по количеству товарных страниц
    sorted_clusters = (
        stats_df.sort_values(
            "Среднее число товарных страниц"
        )["Сегмент"].tolist()
    )

    activity_names = [
        "Низкая активность",
        "Средняя активность",
        "Высокая активность"
    ]

    activity_mapping = {
        cluster: activity_names[index]
        for index, cluster in enumerate(sorted_clusters)
    }

    stats_df["Характеристика сегмента"] = (
        stats_df["Сегмент"].map(activity_mapping)
    )

    stats_df = stats_df[
        [
            "Характеристика сегмента",
            "Количество посетителей",
            "Конверсия, %",
            "Среднее число товарных страниц",
            "Среднее время на страницах товаров",
            "Средний показатель отказов"
        ]
    ]

    st.dataframe(
        stats_df.style.format({
            "Конверсия, %": "{:.2f}",
            "Среднее число товарных страниц": "{:.2f}",
            "Среднее время на страницах товаров": "{:.2f}",
            "Средний показатель отказов": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Уровень активности определяется по среднему количеству "
        "просмотренных товарных страниц в каждом сегменте. "
        "Фактическая конверсия рассчитывается по переменной Revenue."
    )

    # график фактической конверсии
    st.subheader("2. Фактическая конверсия сегментов")

    fig_conversion = px.bar(
        stats_df,
        x="Характеристика сегмента",
        y="Конверсия, %",
        text_auto=".2f",
        title="Доля посетителей, совершивших покупку"
    )

    fig_conversion.update_layout(
        template="plotly_white",
        xaxis_title="Сегмент посетителей",
        yaxis_title="Конверсия, %",
        yaxis_range=[0, max(20, stats_df["Конверсия, %"].max() * 1.2)]
    )

    st.plotly_chart(
        fig_conversion,
        use_container_width=True
    )

    # прогнозирование на тестовой выборке
    st.divider()
    st.subheader("3. Прогнозирование для сегментов")

    st.write(
        "Для каждого посетителя тестовой выборки рассчитывается "
        "вероятность покупки. Затем значения усредняются "
        "внутри соответствующего сегмента."
    )

    # повторяем разбиение, использованное при обучении моделей
    _, test_indices = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=df["Revenue"]
    )

    # оставляем только тестовые наблюдения
    test_segmented_df = segmented_df.loc[test_indices]

    prediction_rows = []

    for cluster in sorted(segmented_df["Сегмент"].unique()):

        # посетители текущего сегмента из тестовой выборки
        cluster_data = test_segmented_df[
            test_segmented_df["Сегмент"] == cluster
        ].copy()

        if cluster_data.empty:
            continue

        X_cluster = cluster_data[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]

        # фактическая конверсия тестовой части сегмента
        actual_conversion = (
            cluster_data["Revenue"].mean() * 100
        )

        row = {
            "Сегмент": activity_mapping[cluster],
            "Фактическая конверсия, %": actual_conversion
        }

        # средняя вероятность покупки для каждой модели
        for name, model in models.items():

            probabilities = model.predict_proba(
                X_cluster
            )[:, 1]

            row[f"{name}, %"] = probabilities.mean() * 100

        prediction_rows.append(row)

    prediction_df = pd.DataFrame(prediction_rows)



    st.dataframe(
        prediction_df.style.format({
            column: "{:.2f}"
            for column in prediction_df.columns
            if column != "Сегмент"
        }),
        use_container_width=True,
        hide_index=True
    )

    # сравнение прогнозов
    st.subheader("4. Сравнение прогнозируемой вероятности покупки")

    chart_df = prediction_df.melt(
        id_vars="Сегмент",
        value_vars=[
            "Логистическая регрессия, %",
            "Random Forest, %"
        ],
        var_name="Модель",
        value_name="Вероятность покупки, %"
    )

    fig_prediction = px.bar(
        chart_df,
        x="Сегмент",
        y="Вероятность покупки, %",
        color="Модель",
        barmode="group",
        text_auto=".2f",
        title="Прогноз вероятности покупки по сегментам"
    )

    fig_prediction.update_layout(
        template="plotly_white",
        xaxis_title="Сегмент посетителей",
        yaxis_title="Вероятность покупки, %",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig_prediction,
        use_container_width=True
    )

    # итоговые выводы
    st.divider()
    st.subheader("5. Выводы сценарного анализа")

    best_conversion = stats_df.loc[
        stats_df["Конверсия, %"].idxmax()
    ]

    st.write(
        f"Наибольшая фактическая конверсия наблюдается "
        f"в сегменте «{best_conversion['Характеристика сегмента']}» "
        f"и составляет "
        f"**{best_conversion['Конверсия, %']:.2f}%**."
    )

    best_rf = prediction_df.loc[
        prediction_df["Random Forest, %"].idxmax()
    ]

    st.write(
        f"По оценке Random Forest наибольшая вероятность покупки "
        f"получена для сегмента "
        f"«{best_rf['Сегмент']}» и составляет "
        f"**{best_rf['Random Forest, %']:.2f}%**."
    )

    st.caption(
        "Фактическая конверсия показывает долю реальных покупок "
        "в сегменте. Средняя прогнозируемая вероятность представляет "
        "собой среднее значение индивидуальных оценок модели "
        "для всех посетителей соответствующего сегмента."
    )

    logistic_higher = (
            prediction_df["Логистическая регрессия, %"]
            > prediction_df["Random Forest, %"]
    )

    if logistic_higher.all():
        st.write(
            "Во всех трёх сегментах логистическая регрессия "
            "показывает более высокую среднюю прогнозируемую "
            "вероятность покупки."
        )

    elif (~logistic_higher).all():
        st.write(
            "Во всех трёх сегментах Random Forest показывает "
            "более высокую среднюю прогнозируемую вероятность покупки."
        )

    else:
        st.write(
            "Результаты моделей различаются в зависимости "
            "от сегмента посетителей. Единого превосходства "
            "одной модели по средней вероятности не наблюдается."
        )
