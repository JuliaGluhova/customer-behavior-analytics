
import pandas as pd
import plotly.express as px
import streamlit as st


def render_data_analysis(df):

    st.title("Загрузка и обзор данных")

    st.success("Датасет успешно загружен!")

    # общая информация о датасете
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Количество записей", df.shape[0])

    with col2:
        st.metric("Количество столбцов", df.shape[1])

    with col3:
        st.metric(
            "Пропущенные значения",
            int(df.isnull().sum().sum())
        )

    st.subheader("Первые строки датасета")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Информация о столбцах")
    st.write(df.dtypes.astype(str))

    # анализ целевой переменной
    st.divider()
    st.subheader("Распределение целевой переменной Revenue")

    revenue_counts = df["Revenue"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Посетители с покупкой",
            int(revenue_counts.get(True, 0))
        )

    with col2:
        st.metric(
            "Посетители без покупки",
            int(revenue_counts.get(False, 0))
        )

    revenue_data = pd.DataFrame({
        "Результат": ["Покупка", "Без покупки"],
        "Количество": [
            int(revenue_counts.get(True, 0)),
            int(revenue_counts.get(False, 0))
        ]
    })

    st.bar_chart(
        revenue_data.set_index("Результат")
    )

    purchase_rate = df["Revenue"].mean() * 100

    st.metric(
        "Доля посетителей, совершивших покупку",
        f"{purchase_rate:.2f}%"
    )

    # анализ покупательской активности по месяцам
    st.divider()
    st.subheader("Покупательская активность по месяцам")

    month_order = [
        "Feb", "Mar", "May", "June", "Jul",
        "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    monthly_conversion = (
        df.groupby("Month")["Revenue"]
        .mean()
        .mul(100)
        .reindex(month_order)
        .dropna()
    )

    monthly_data = (
        monthly_conversion
        .rename_axis("Месяц")
        .reset_index()
    )

    monthly_data.columns = ["Месяц", "Конверсия"]

    fig = px.bar(
        monthly_data,
        x="Месяц",
        y="Конверсия",
        text="Конверсия",
        category_orders={"Месяц": month_order}
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        title="Конверсия посетителей по месяцам",
        xaxis_title="Месяц",
        yaxis_title="Конверсия, %",
        template="plotly_white",
        yaxis_range=[0, 30]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Конверсия — доля сессий с покупкой от общего количества "
        "сессий в соответствующем месяце."
    )

    # анализ типов посетителей
    st.divider()
    st.subheader("Анализ типов посетителей")

    visitor_analysis = (
        df.groupby("VisitorType")
        .agg(
            Количество_сессий=("Revenue", "count"),
            Покупки=("Revenue", "sum"),
            Конверсия=("Revenue", "mean")
        )
        .reset_index()
    )

    visitor_analysis["Конверсия"] *= 100

    st.dataframe(
        visitor_analysis.style.format({
            "Конверсия": "{:.2f}%"
        }),
        use_container_width=True
    )

    fig_visitor = px.bar(
        visitor_analysis,
        x="VisitorType",
        y="Конверсия",
        text="Конверсия",
        title="Конверсия по типам посетителей",
        category_orders={
            "VisitorType": [
                "New_Visitor",
                "Returning_Visitor",
                "Other"
            ]
        }
    )

    fig_visitor.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_visitor.update_layout(
        xaxis_title="Тип посетителя",
        yaxis_title="Конверсия, %",
        template="plotly_white"
    )

    st.plotly_chart(fig_visitor, use_container_width=True)
