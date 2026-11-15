
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import confusion_matrix, roc_curve, auc

from models.prediction import train_models


@st.cache_resource
def get_prediction_results(df):
    return train_models(df)


def render_prediction(df):

    st.title("Прогнозирование покупок")

    st.write(
        "Сравнение двух алгоритмов машинного обучения "
        "для прогнозирования совершения покупки "
        "посетителями интернет-магазина."
    )

    st.info(
        "Для обучения используется 80% данных, "
        "для тестирования — 20%. "
        "Целевая переменная — Revenue."
    )

    # обучение моделей
    with st.spinner("Обучение моделей..."):

        (
            results_df,
            fitted_models,
            y_test,
            predictions,
            probabilities
        ) = get_prediction_results(df)

    # сохраняем обученные модели для индивидуального прогноза
    st.session_state["prediction_models"] = fitted_models

    st.success("Обучение и тестирование моделей завершены!")

    st.divider()

    # сравнение качества моделей
    st.subheader("1. Сравнение качества моделей")

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

    # сравнение метрик на диаграмме
    st.subheader("2. Сравнение показателей")

    metrics_data = results_df.melt(
        id_vars="Модель",
        var_name="Метрика",
        value_name="Значение"
    )

    fig = px.bar(
        metrics_data,
        x="Метрика",
        y="Значение",
        color="Модель",
        barmode="group",
        text_auto=".3f",
        title="Сравнение качества алгоритмов"
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_title="Значение метрики",
        xaxis_title="Метрика",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # определение лучшей модели
    st.subheader("3. Лучшая модель")

    best_model = results_df.loc[
        results_df["F1-score"].idxmax()
    ]

    st.metric(
        "Лучшая модель по F1-score",
        best_model["Модель"],
        f"F1-score: {best_model['F1-score']:.4f}"
    )

    st.write(
        f"Наибольшее значение F1-score показала модель "
        f"**{best_model['Модель']}**. "
        "Данная метрика учитывает одновременно точность "
        "и полноту обнаружения покупок."
    )

    st.divider()

    # матрицы ошибок
    st.subheader("4. Матрицы ошибок")

    st.write(
        "Матрицы ошибок показывают количество правильных "
        "и ошибочных прогнозов для каждого класса."
    )

    columns = st.columns(2)

    for col, (name, y_pred) in zip(
        columns,
        predictions.items()
    ):

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Без покупки", "Покупка"],
            y=["Без покупки", "Покупка"],
            labels={
                "x": "Предсказанный класс",
                "y": "Фактический класс",
                "color": "Количество"
            },
            title=name
        )

        with col:
            st.plotly_chart(
                fig_cm,
                use_container_width=True
            )

    st.caption(
        "Главная диагональ матрицы содержит правильные прогнозы, "
        "остальные ячейки — ошибочные классификации."
    )

    st.divider()

    # ROC-кривые
    st.subheader("5. ROC-кривые")

    fig_roc = go.Figure()

    for name, y_prob in probabilities.items():

        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_prob
        )

        roc_auc = auc(fpr, tpr)

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC = {roc_auc:.3f})"
            )
        )

    # линия случайного классификатора
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Случайный классификатор",
            line=dict(dash="dash")
        )
    )

    fig_roc.update_layout(
        title="Сравнение ROC-кривых",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        fig_roc,
        use_container_width=True
    )

    st.caption(
        "Чем ближе ROC-кривая к левому верхнему углу, "
        "тем лучше модель различает посетителей, "
        "совершивших и не совершивших покупку."
    )

    st.divider()

    # итоговый вывод
    st.subheader("6. Выводы")

    logistic_f1 = results_df.loc[
        results_df["Модель"] == "Логистическая регрессия",
        "F1-score"
    ].iloc[0]

    forest_f1 = results_df.loc[
        results_df["Модель"] == "Random Forest",
        "F1-score"
    ].iloc[0]

    if forest_f1 > logistic_f1:
        st.write(
            "В проведённом эксперименте Random Forest показал "
            "более высокое значение F1-score по сравнению "
            "с логистической регрессией."
        )

    elif logistic_f1 > forest_f1:
        st.write(
            "В проведённом эксперименте логистическая регрессия "
            "показала более высокое значение F1-score "
            "по сравнению с Random Forest."
        )

    else:
        st.write(
            "Обе модели показали одинаковое значение F1-score."
        )

    st.write(
        "Окончательный выбор модели следует делать с учётом "
        "всех рассчитанных метрик и целей прогнозирования."
    )
