
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_data

from models.segmentation import (
    prepare_data,
    evaluate_kmeans,
    perform_segmentation
)

from views.data_analysis import render_data_analysis
from views.home import render_home
from views.prediction import render_prediction
from views.scenario_analysis import render_scenario_analysis
from views.results import render_results


st.set_page_config(
    page_title="Customer Behavior Analytics",
    page_icon=None,
    layout="wide"
)


# оформление приложения
st.markdown("""
<style>

/* основная область */
.stApp {
    background-color: #F4F6FA;
    color: #202B3C;
    font-family: "Segoe UI", sans-serif;
}

/* боковая панель */
[data-testid="stSidebar"] {
    background-color: #17243A;
    border-right: 1px solid #26364F;
}

[data-testid="stSidebar"] * {
    color: #E8EDF5;
}

/* название приложения */
.brand-title {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #FFFFFF;
    padding: 12px 0 4px 0;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #AEBBD0;
}

/* заголовки */
h1 {
    color: #17243A !important;
    font-size: 34px !important;
    font-weight: 750 !important;
}

h2, h3 {
    color: #263B58 !important;
    font-weight: 650 !important;
}

/* разделители */
hr {
    border-color: #DCE3ED;
}

/* навигация */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 9px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: transparent;
    border-radius: 8px;
    padding: 9px 10px;
    transition: background-color 0.2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background-color: #293B57;
}

/* карточки показателей */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E1E7F0;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(28, 45, 70, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #718096;
    font-size: 14px;
}

[data-testid="stMetricValue"] {
    color: #17243A;
    font-weight: 700;
}

/* таблицы */
[data-testid="stDataFrame"] {
    border: 1px solid #E1E7F0;
    border-radius: 10px;
    overflow: hidden;
}

/* кнопки */
.stButton > button {
    background-color: #267D8B;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #1D6571;
    color: white;
    border: none;
}

/* поля ввода */
.stSelectbox > div > div,
.stNumberInput input,
.stTextInput input {
    border-radius: 8px;
}

/* информационные блоки */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* верхняя панель */
header[data-testid="stHeader"] {
    background-color: #F4F6FA;
}

/* скрываем стандартную нижнюю подпись */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# боковое меню
with st.sidebar:
    st.markdown(
        '<div class="brand-title">CBA</div>',
        unsafe_allow_html=True
    )
    st.caption("Customer Behavior Analytics")

    st.divider()

    page = st.radio(
        "НАВИГАЦИЯ",
        [
            "Главная",
            "Загрузка и анализ данных",
            "Сегментация потребителей",
            "Прогнозирование покупок",
            "Сценарный анализ поведения потребителей",
            "Итоговые результаты"
        ]
    )

    st.divider()

    st.markdown("---")
    st.caption("CUSTOMER ANALYTICS")
    st.caption("Research platform · 2026")


# загрузка данных
df = load_data()


# 1. главная
if page == "Главная":
    render_home(df)


# 2. загрузка и анализ данных
elif page == "Загрузка и анализ данных":
    render_data_analysis(df)


# 3. сегментация потребителей
elif page == "Сегментация потребителей":

    st.title("Сегментация потребителей")

    st.write(
        "Исследование структуры посетителей интернет-магазина "
        "с использованием алгоритма K-means."
    )

    st.divider()

    st.subheader("1. Настройки сегментации")

    n_clusters = st.selectbox(
        "Количество кластеров",
        [2, 3, 4, 5, 6],
        index=1
    )

    st.info(
        "Для сегментации используются 10 числовых признаков "
        "поведения посетителей. Переменная Revenue "
        "не участвует в обучении алгоритма."
    )

    # подготовка данных
    X_scaled, scaler = prepare_data(df)

    # оценка количества кластеров
    st.divider()
    st.subheader("2. Выбор количества кластеров")

    with st.spinner("Выполняется оценка кластеров..."):
        evaluation_df = evaluate_kmeans(
            X_scaled,
            min_clusters=2,
            max_clusters=6
        )

    col1, col2 = st.columns(2)

    with col1:
        fig_elbow = px.line(
            evaluation_df,
            x="Количество кластеров",
            y="Инерция",
            markers=True,
            title="Метод локтя"
        )

        fig_elbow.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig_elbow,
            use_container_width=True
        )

    with col2:
        fig_silhouette = px.line(
            evaluation_df,
            x="Количество кластеров",
            y="Silhouette Score",
            markers=True,
            title="Silhouette Score"
        )

        fig_silhouette.update_layout(
            template="plotly_white"
        )

        st.plotly_chart(
            fig_silhouette,
            use_container_width=True
        )

    best_k = int(
        evaluation_df.loc[
            evaluation_df["Silhouette Score"].idxmax(),
            "Количество кластеров"
        ]
    )

    st.write(
        f"Наибольшее значение Silhouette Score получено "
        f"при количестве кластеров K = **{best_k}**."
    )

    st.dataframe(
        evaluation_df.style.format({
            "Инерция": "{:.2f}",
            "Silhouette Score": "{:.4f}"
        }),
        use_container_width=True
    )

    # результаты кластеризации
    st.divider()
    st.subheader("3. Результаты сегментации")

    with st.spinner("Выполняется кластеризация..."):
        (
            kmeans_df,
            kmeans_profiles,
            kmeans_model,
            kmeans_scaler
        ) = perform_segmentation(df, n_clusters)

    st.success(
        f"Сегментация завершена. Получено кластеров: {n_clusters}."
    )

    fig_kmeans = px.bar(
        kmeans_profiles,
        x="Сегмент",
        y="Количество_сессий",
        text="Количество_сессий",
        title="Распределение посетителей по сегментам"
    )

    fig_kmeans.update_layout(
        template="plotly_white",
        xaxis_title="Номер сегмента",
        yaxis_title="Количество сессий"
    )

    st.plotly_chart(
        fig_kmeans,
        use_container_width=True
    )

    st.subheader("Характеристики сегментов")

    st.dataframe(
        kmeans_profiles.style.format({
            "Конверсия": "{:.2f}%",
            "Среднее_число_товарных_страниц": "{:.2f}",
            "Среднее_значение_PageValues": "{:.2f}",
            "Средний_BounceRate": "{:.4f}",
            "Средний_ExitRate": "{:.4f}"
        }),
        use_container_width=True
    )

    st.subheader("Вывод")

    st.write(
        f"В результате применения алгоритма K-means "
        f"посетители интернет-магазина разделены на "
        f"**{n_clusters} сегмента**. "
        "Полученные группы различаются по характеристикам "
        "поведения и активности на сайте."
    )


# 4. прогнозирование покупок
elif page == "Прогнозирование покупок":

    render_prediction(df)


# 5. индивидуальный прогноз для посетителя
elif page == "Сценарный анализ поведения потребителей":
    render_scenario_analysis(df)


# 6. итоговые результаты
elif page == "Итоговые результаты":
    render_results(df)
