# ML Dashboard — Предсказание цен на автомобили

Расчётно-графическая работа по дисциплине **«Машинное обучение и большие данные»**

> **Студент:** Темербалинов Артур Кайратович · Группа ФИТ-242  
> **Тема:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных

---

## О проекте

Интерактивный веб-дашборд для предсказания цены подержанного автомобиля.  
Реализован на **Streamlit**, развёрнут на **Hugging Face Spaces**.

**Ссылка на приложение:** [открыть на Hugging Face](https://huggingface.co/spaces/...)

---

## Структура проекта

```
my_rgr/
├── app.py                  # главная страница
├── requirements.txt        # зависимости
├── pages/
│   ├── 1_Разработчик.py   # информация об авторе
│   ├── 2_Датасет.py       # описание данных и EDA
│   ├── 3_Визуализации.py  # графики (5 видов)
│   └── 4_Инференс.py      # предсказание цены
├── models/
│   ├── elasticnet.pkl      # ElasticNet
│   ├── gradient_boosting.pkl # GradientBoosting
│   ├── catboost.cbm        # CatBoost
│   ├── bagging.pkl         # BaggingRegressor
│   ├── stacking.pkl        # StackingRegressor
│   ├── keras_rmsprop.keras # Нейросеть Keras
│   └── scaler.pkl          # StandardScaler
├── data/
│   └── cars_cleaned.csv    # предобработанный датасет
└── assets/
    └── photo.jpg           # фото разработчика
```

---

## Модели ML

| # | Модель | Тип | Метрика (R²) |
|---|--------|-----|-------------|
| 1 | ElasticNet + TransformedTarget | Классическая линейная | ~0.83 |
| 2 | GradientBoostingRegressor | Бустинг | ~0.90 |
| 3 | CatBoostRegressor | Продвинутый бустинг | ~0.90 |
| 4 | BaggingRegressor (DecisionTree) | Бэггинг | ~0.89 |
| 5 | StackingRegressor (RF+SVR+KNN+GBR → Ridge) | Стэкинг | ~0.90 |
| 6 | Keras Dense NN (RMSProp) | Нейросеть | ~0.87 |

---

## Датасет

- **Источник:** [Used Cars Dataset](https://www.kaggle.com/datasets/lepchenkov/usedcarscatalog)  
- **Целевая переменная:** `price_usd` — цена автомобиля в долларах США  
- **Признаки:** год выпуска, пробег, марка, тип двигателя, трансмиссия и др.

---

## Запуск локально

```bash
git clone https://github.com/Temart00f/ML_RGR
pip install -r requirements.txt
streamlit run app.py
```

---

## Технологии

`Python` · `Streamlit` · `Scikit-learn` · `CatBoost` · `TensorFlow/Keras` · `Pandas` · `Matplotlib` · `Seaborn` · `Optuna`
