import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Визуализации", page_icon="📈", layout="wide")

st.title("📈 Визуализации зависимостей")
st.markdown("---")

@st.cache_data
def load_data():
    path = "data/cars_cleaned.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()

if df is None:
    st.warning("Файл `data/cars_cleaned.csv` не найден.")
    st.stop()

# Числовые колонки
num_cols = df.select_dtypes(include=np.number).columns.tolist()
# Категориальные колонки
# стало:
cat_cols = [c for c in ['transmission', 'engine_fuel', 'body_type', 
                         'drivetrain', 'engine_type', 'color', 
                         'state', 'manufacturer_name'] 
            if c in df.columns]

if not cat_cols:
    cat_cols = [c for c in df.columns if df[c].nunique() < 20 and c != 'price_usd']

TARGET = "price_usd"

# ================================================================
# График 1: Гистограмма распределения цен
# ================================================================
st.markdown("## 1. Распределение цен на автомобили")
st.markdown(
    "Гистограмма показывает, что большинство автомобилей в датасете стоят до 20 000 USD. "
    "Распределение правосторонне скошено — есть небольшое число очень дорогих автомобилей."
)

col1, col2 = st.columns([3, 1])
with col2:
    bins = st.slider("Количество столбцов", 10, 80, 40, key="hist_bins")
    log_scale = st.checkbox("Логарифмическая ось X", value=False)

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    data_plot = np.log1p(df[TARGET]) if log_scale else df[TARGET]
    xlabel = "log(price_usd + 1)" if log_scale else "Цена (USD)"
    ax.hist(data_plot.dropna(), bins=bins, color="#5DCAA5", edgecolor="white", linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Количество автомобилей", fontsize=12)
    ax.set_title("Распределение цен", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ================================================================
# График 2: Корреляционная матрица
# ================================================================
st.markdown("## 2. Корреляционная матрица числовых признаков")
st.markdown(
    "Тепловая карта отображает попарные корреляции Пирсона. "
    "Видно, что `year_produced` положительно коррелирует с ценой, "
    "а `odometer_value` — отрицательно (чем больше пробег, тем ниже цена)."
)

corr_cols = ['odometer_value', 'year_produced', 'engine_capacity',
             'number_of_photos', 'up_counter', 'duration_listed', 'price_usd']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=ax,
    annot_kws={"size": 8}
)
ax.set_title("Корреляционная матрица", fontsize=13)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ================================================================
# График 3: Scatter plot — признак vs цена
# ================================================================
st.markdown("## 3. Зависимость признака от цены")
st.markdown("Диаграмма рассеяния позволяет визуально оценить, как числовой признак связан с целевой переменной.")

col1, col2 = st.columns([3, 1])
with col2:
    available = [c for c in num_cols if c != TARGET]
    feature_x = st.selectbox("Признак (ось X)", available,
                              index=available.index("odometer_value") if "odometer_value" in available else 0)
    sample_n = st.slider("Размер выборки", 500, min(5000, len(df)), 2000, step=500)
    alpha_val = st.slider("Прозрачность точек", 0.1, 1.0, 0.4, step=0.1)

with col1:
    sample = df[[feature_x, TARGET]].dropna().sample(min(sample_n, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(sample[feature_x], sample[TARGET], alpha=alpha_val,
               color="#7F77DD", s=15, edgecolors="none")
    ax.set_xlabel(feature_x, fontsize=12)
    ax.set_ylabel("Цена (USD)", fontsize=12)
    ax.set_title(f"{feature_x} vs price_usd", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ================================================================
# График 4: Boxplot цены по категориальному признаку
# ================================================================
st.markdown("## 4. Распределение цен по категориям")
st.markdown(
    "Ящик с усами (boxplot) показывает медиану, квартили и выбросы цен "
    "в каждой категории выбранного признака."
)

col1, col2 = st.columns([3, 1])
with col2:
    cat_feature = st.selectbox(
        "Категориальный признак",
        cat_cols,
        index=cat_cols.index("transmission") if "transmission" in cat_cols else 0
    )
    top_n = st.slider("Топ-N категорий", 3, 15, 8)
    price_cap = st.number_input("Макс. цена (фильтр выбросов, USD)", value=50000, step=5000)

with col1:
    top_cats = df[cat_feature].value_counts().head(top_n).index
    df_box = df[df[cat_feature].isin(top_cats) & (df[TARGET] <= price_cap)]

    fig, ax = plt.subplots(figsize=(10, 4))
    order = df_box.groupby(cat_feature)[TARGET].median().sort_values(ascending=False).index
    sns.boxplot(data=df_box, x=cat_feature, y=TARGET, order=order,
                palette="Set2", ax=ax, flierprops={"markersize": 2})
    ax.set_xlabel(cat_feature, fontsize=12)
    ax.set_ylabel("Цена (USD)", fontsize=12)
    ax.set_title(f"Распределение цен по признаку «{cat_feature}»", fontsize=13)
    plt.xticks(rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ================================================================
# График 5: Средняя цена по году выпуска
# ================================================================
st.markdown("## 5. Средняя цена по году выпуска")
st.markdown(
    "Линейный график показывает, как менялась средняя стоимость автомобилей "
    "в зависимости от года выпуска. Более новые автомобили, как правило, стоят дороже."
)

if "year_produced" in df.columns:
    year_price = (
        df.groupby("year_produced")[TARGET]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    year_price = year_price[year_price["count"] >= 10]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(year_price["year_produced"], year_price["mean"],
            color="#D85A30", linewidth=2, label="Средняя цена")
    ax.plot(year_price["year_produced"], year_price["median"],
            color="#185FA5", linewidth=2, linestyle="--", label="Медианная цена")
    ax.fill_between(year_price["year_produced"],
                    year_price["mean"], year_price["median"],
                    alpha=0.1, color="#888780")
    ax.set_xlabel("Год выпуска", fontsize=12)
    ax.set_ylabel("Цена (USD)", fontsize=12)
    ax.set_title("Динамика цен по году выпуска", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("Признак `year_produced` не найден в датасете.")
