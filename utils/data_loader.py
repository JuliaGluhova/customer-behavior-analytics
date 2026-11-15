
from pathlib import Path

import pandas as pd
import streamlit as st


# определяем путь к датасету
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "online_shoppers_intention.csv"


@st.cache_data
def load_data():
    """Загружает исходный датасет."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df
