
import streamlit as st


def render_home(df):
    st.title("Аналитическая панель")

    st.write(
        "Исследование поведения потребителей "
        "в цифровой экономике"
    )

    st.divider()

    st.subheader("Общая информация")

    total_sessions = len(df)
    total_segments = 3
    purchase_rate = df["Revenue"].mean() * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Количество сессий",
        f"{total_sessions:,}".replace(",", " ")
    )

    col2.metric(
        "Количество сегментов",
        total_segments
    )

    col3.metric(
        "Конверсия в покупку",
        f"{purchase_rate:.2f}%"
    )

    st.info(
        "Используйте боковое меню для анализа данных, "
        "сегментации потребителей и прогнозирования покупок."
    )
