import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download
import os

st.set_page_config(page_title="Инференс", page_icon="🤖", layout="wide")

st.title("🤖 Предсказание цены автомобиля")
st.markdown("---")

# ================================================================
# Загрузка моделей
# ================================================================
from huggingface_hub import hf_hub_download
import os

@st.cache_resource
def load_models():
    models = {}
    errors = []
    os.makedirs("models", exist_ok=True)

    # Скачиваем все файлы с HF Hub
    hub_files = [
        "elasticnet.pkl",
        "gradient_boosting.pkl",
        "bagging.pkl",
        "stacking.pkl",
        "catboost.cbm",
        "keras_rmsprop.keras",
        "scaler.pkl",
    ]
    for filename in hub_files:
        local_path = f"models/{filename}"
        if not os.path.exists(local_path):
            try:
                hf_hub_download(
                    repo_id="Temart00f/cars-models",
                    filename=filename,
                    local_dir="models"
                )
            except Exception as e:
                errors.append(f"Ошибка загрузки {filename}: {e}")

    # pkl модели
    pkl_models = {
        "ElasticNet (линейная)": "models/elasticnet.pkl",
        "GradientBoosting (бустинг)": "models/gradient_boosting.pkl",
        "BaggingRegressor (бэггинг)": "models/bagging.pkl",
        "StackingRegressor (стэкинг)": "models/stacking.pkl",
    }
    for name, path in pkl_models.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = ("pkl", pickle.load(f))
        else:
            errors.append(f"Не найден: `{path}`")

    # CatBoost
    cb_path = "models/catboost.cbm"
    if os.path.exists(cb_path):
        try:
            from catboost import CatBoostRegressor
            cb = CatBoostRegressor()
            cb.load_model(cb_path)
            models["CatBoost"] = ("catboost", cb)
        except Exception as e:
            errors.append(f"CatBoost: {e}")

    # Keras
    keras_path = "models/keras_rmsprop.keras"
    if os.path.exists(keras_path):
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            keras_model = tf.keras.models.load_model(keras_path)
            models["Нейросеть Keras (RMSProp)"] = ("keras", keras_model)
        except Exception as e:
            errors.append(f"Keras: {e}")

    return models, errors


@st.cache_resource
def load_scaler():
    path = "models/scaler.pkl"
    if not os.path.exists(path):
        try:
            hf_hub_download(
                repo_id="Temart00f/cars-models",
                filename="scaler.pkl",
                local_dir="models"
            )
        except Exception as e:
            return None
    with open(path, "rb") as f:
        return pickle.load(f)

models, load_errors = load_models()
scaler = load_scaler()

# Предупреждения о ненайденных файлах
if load_errors:
    with st.expander("⚠️ Некоторые модели не загружены", expanded=False):
        for e in load_errors:
            st.warning(e)

if not models:
    st.error("Ни одна модель не загружена. Убедитесь, что папка `models/` содержит файлы моделей.")
    st.stop()

# ================================================================
# Выбор модели
# ================================================================
st.markdown("## Шаг 1 — Выберите модель")
model_name = st.selectbox("Модель ML", list(models.keys()))
model_type, model_obj = models[model_name]

needs_scaler = model_type in ("pkl",) and "ElasticNet" in model_name
needs_scaler_keras = model_type == "keras"

st.markdown("---")

# ================================================================
# Выбор способа ввода данных
# ================================================================
st.markdown("## Шаг 2 — Введите данные")
input_method = st.radio(
    "Способ ввода",
    ["✏️ Ввести вручную", "📂 Загрузить CSV-файл"],
    horizontal=True,
)

# ================================================================
# Вспомогательная функция предсказания
# ================================================================
def predict(model_type, model_obj, X: pd.DataFrame) -> np.ndarray:
    """Возвращает предсказания в USD (исходный масштаб)."""
    if model_type == "pkl":
        # ElasticNet обёрнут в TransformedTargetRegressor — уже возвращает USD
        return model_obj.predict(X)
    elif model_type == "catboost":
        return model_obj.predict(X)
    elif model_type == "keras":
        preds_log = model_obj.predict(X, verbose=0).flatten()
        return np.expm1(preds_log)
    return np.array([])

def apply_scaler(X: pd.DataFrame) -> pd.DataFrame:
    if scaler is not None:
        try:
            return pd.DataFrame(scaler.transform(X), columns=X.columns)
        except Exception:
            return X
    return X

def format_price(usd: float) -> str:
    return f"${usd:,.0f} USD"

# ================================================================
# Способ 1: ввод вручную
# ================================================================
if input_method == "✏️ Ввести вручную":
    st.markdown("### Характеристики автомобиля")

    col1, col2, col3 = st.columns(3)

    with col1:
        year_produced = st.number_input("Год выпуска", min_value=1970, max_value=2024, value=2015)
        odometer_value = st.number_input("Пробег (км)", min_value=0, max_value=1_000_000, value=80_000, step=1000)
        engine_capacity = st.number_input("Объём двигателя (л)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        engine_has_gas = st.checkbox("Газовое оборудование (ГБО)")
        st.markdown("**Дополнительные опции**")
        st.caption("Наличие доп. оборудования")
        feature_0 = int(st.checkbox("Опция 1"))
        feature_1 = int(st.checkbox("Опция 2"))
        feature_2 = int(st.checkbox("Опция 3"))
        feature_3 = int(st.checkbox("Опция 4"))
        feature_4 = int(st.checkbox("Опция 5"))
        feature_5 = int(st.checkbox("Опция 6"))
        feature_6 = int(st.checkbox("Опция 7"))
        feature_7 = int(st.checkbox("Опция 8"))
        feature_8 = int(st.checkbox("Опция 9"))
        feature_9 = int(st.checkbox("Опция 10"))

    with col2:
        transmission = st.selectbox("Трансмиссия", ["mechanical", "automatic"])
        drivetrain = st.selectbox("Привод", ["front", "rear", "all"])
        state = st.selectbox("Состояние", ["emergency", "owned", "new"])
        has_warranty = st.checkbox("Есть гарантия")

    with col3:
        body_type = st.selectbox("Тип кузова", [
            "sedan", "suv", "hatchback", "universal", "minivan",
            "coupe", "cabriolet", "pickup", "minibus", "van", "liftback", "limousine"
        ])
        engine_fuel = st.selectbox("Тип топлива", [
            "gasoline", "diesel", "gas", "electric", "hybrid-petrol", "hybrid-diesel"
        ])
        location = st.selectbox("Регион", [
            "Минская обл.", "Брестская обл.", "Витебская обл.",
            "Гомельская обл.", "Гродненская обл.", "Могилевская обл."
        ])
        manufacturer_segment = st.selectbox("Ценовой сегмент марки", [
            "Бюджетный (Daewoo, ВАЗ…)",
            "Средний (Volkswagen, Toyota…)",
            "Премиум (BMW, Mercedes…)"
        ])
        number_of_photos = st.slider("Количество фото", 0, 30, 5)

    st.markdown("---")
    st.markdown("### Шаг 3 — Получить предсказание")

    if st.button("🔮 Предсказать цену", type="primary", use_container_width=True):
        car_age = 2024 - year_produced
        log_odometer = np.log1p(odometer_value)

        # Формируем вектор признаков.
        # ВАЖНО: порядок и набор столбцов должен совпадать с тем, что видела модель при обучении.
        # Если у вас другие признаки — добавьте/уберите столбцы здесь.
        input_dict = {
            "manufacturer_name": ["Бюджетный (Daewoo, ВАЗ…)", "Средний (Volkswagen, Toyota…)", "Премиум (BMW, Mercedes…)"].index(manufacturer_segment),
            "transmission":      1 if transmission == "automatic" else 0,
            "odometer_value":    odometer_value,
            "year_produced":     year_produced,
            "engine_has_gas":    int(engine_has_gas),
            "engine_capacity":   engine_capacity,
            "has_warranty":      int(has_warranty),
            "state":             ["emergency", "owned", "new"].index(state),
            "drivetrain":        ["front", "rear", "all"].index(drivetrain),
            "is_exchangeable":   0,
            "number_of_photos":  number_of_photos,
            "up_counter":        0,
            "feature_0": feature_0, "feature_1": feature_1, "feature_2": feature_2,
            "feature_3": feature_3, "feature_4": feature_4, "feature_5": feature_5,
            "feature_6": feature_6, "feature_7": feature_7, "feature_8": feature_8,
            "feature_9": feature_9,
            "duration_listed":   30,
            # локация
            "loc_Брестская обл.":  int(location == "Брестская обл."),
            "loc_Витебская обл.":  int(location == "Витебская обл."),
            "loc_Гомельская обл.": int(location == "Гомельская обл."),
            "loc_Гродненская обл.":int(location == "Гродненская обл."),
            "loc_Минская обл.":    int(location == "Минская обл."),
            "loc_Могилевская обл.":int(location == "Могилевская обл."),
            # кузов
            "body_cabriolet": int(body_type == "cabriolet"),
            "body_coupe":     int(body_type == "coupe"),
            "body_hatchback": int(body_type == "hatchback"),
            "body_liftback":  int(body_type == "liftback"),
            "body_limousine": int(body_type == "limousine"),
            "body_minibus":   int(body_type == "minibus"),
            "body_minivan":   int(body_type == "minivan"),
            "body_pickup":    int(body_type == "pickup"),
            "body_sedan":     int(body_type == "sedan"),
            "body_suv":       int(body_type == "suv"),
            "body_universal": int(body_type == "universal"),
            "body_van":       int(body_type == "van"),
            # топливо
            "fuel_diesel":         int(engine_fuel == "diesel"),
            "fuel_electric":       int(engine_fuel == "electric"),
            "fuel_gas":            int(engine_fuel == "gas"),
            "fuel_gasoline":       int(engine_fuel == "gasoline"),
            "fuel_hybrid-diesel":  int(engine_fuel == "hybrid-diesel"),
            "fuel_hybrid-petrol":  int(engine_fuel == "hybrid-petrol"),
            # инженерия признаков
            "car_age":       2024 - year_produced,
            "log_odometer":  np.log1p(odometer_value),
        }

        X_input = pd.DataFrame([input_dict])

        # Масштабирование для ElasticNet и Keras
        X_to_predict = X_input.copy()
        if needs_scaler or needs_scaler_keras:
            X_to_predict = apply_scaler(X_to_predict)

        try:
            pred = predict(model_type, model_obj, X_to_predict)[0]
            pred = max(0, pred)  # цена не может быть отрицательной

            st.success(f"### 💰 Прогнозируемая стоимость: **{format_price(pred)}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Модель", model_name)
            col2.metric("Возраст авто", f"{car_age} лет")
            col3.metric("Пробег", f"{odometer_value:,} км")

        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
            st.info(
                "Возможно, признаки входного вектора не совпадают с признаками обучения. "
                "Проверьте набор столбцов в `X_input`."
            )

# ================================================================
# Способ 2: загрузка CSV
# ================================================================
else:
    st.markdown("### Загрузка CSV-файла")
    st.info(
        "Загрузите CSV-файл, содержащий те же признаки, что использовались при обучении модели "
        "(без столбца `price_usd`). Модель выдаст предсказание для каждой строки."
    )

    uploaded = st.file_uploader("Выберите CSV-файл", type=["csv"])

    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            st.markdown(f"**Загружено строк:** {len(df_upload)} · **Признаков:** {df_upload.shape[1]}")
            st.dataframe(df_upload.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            st.stop()

        st.markdown("---")
        if st.button("🔮 Предсказать для всех строк", type="primary", use_container_width=True):
            X_to_predict = df_upload.copy()
            if needs_scaler or needs_scaler_keras:
                X_to_predict = apply_scaler(X_to_predict)

            try:
                preds = predict(model_type, model_obj, X_to_predict)
                preds = np.maximum(preds, 0)

                df_result = df_upload.copy()
                df_result["predicted_price_usd"] = preds.round(0).astype(int)

                st.success(f"Предсказания получены для {len(preds)} строк")
                st.dataframe(df_result, use_container_width=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Средняя цена", format_price(preds.mean()))
                col2.metric("Мин. цена", format_price(preds.min()))
                col3.metric("Макс. цена", format_price(preds.max()))

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Скачать результаты CSV",
                    csv_out,
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
