import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import to_hex as rgb2hex
from sqlalchemy import create_engine
# import psycopg2
import sqlite3
import os
import urllib.request
import gdown
from datetime import datetime


# ---- ЗАГРУЗКА ДАННЫХ ----

@st.cache_data
def load_data():
    # df = pd.read_excel("defects.xlsx") - использовалось ранее
    
    # # !!!!!!параметры подключения  для PostgreSQL
    # engine = create_engine("postgresql+psycopg2://postgres:@localhost:5432/defects")
    # query = "SELECT * FROM defects" 
    # df = pd.read_sql(query, engine)

    # DB_URL = "https://drive.google.com/uc?export=download&id=18rFP7h9Dwv6jh-juwTGVfF_PXuI63rdr"
    # DB_FILE = "defects.sqlite"

    DB_ID = "18rFP7h9Dwv6jh-juwTGVfF_PXuI63rdr"
    DB_URL = f"https://drive.google.com/uc?id={DB_ID}"
    DB_FILE = "defects.sqlite"


    gdown.download(DB_URL, DB_FILE, quiet=False)

    # Скачиваем БД
    if not os.path.exists(DB_FILE):
        with st.spinner("Скачиваем базу данных..."):
            urllib.request.urlretrieve(DB_URL, DB_FILE)



    # Подключение SQLite
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM defects", conn)
    conn.close()
      
    df["molding_date"] = pd.to_datetime(df["molding_date"])
    df["month_year"] = df["molding_date"].dt.strftime("%b %y")
    df["Week"] = df["molding_date"].dt.to_period("W").astype(str)
    df["month_start"] = df["molding_date"].dt.to_period("M").dt.to_timestamp()
    # Добавляем год и месяц в датафрейм
    df["year"] = df["molding_date"].dt.year
    df["month"] = df["molding_date"].dt.month
    return df

df = load_data()

st.markdown(
    """
    <style>
        .main {
            max-width: 95vw;  /* делаем шире основное поле,  75% от ширины окна */
            padding-left: 3rem;
            padding-right: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True)

# ---- SIDEBAR ----

# Ширина сайдбара
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            width: 200px !important;  /* регулировать ширину */
        }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.sidebar.header("📊 Фiльтри")

# Фильтр по деталям 

part_name_mapping = {
    'frame': 'Рама',
    'beam': 'Балка',
    'draft_yoke': 'Тягове ярмо',
    'coupler_1008': 'Автозчеп 1008',
    'coupler_1028': 'Автозчеп 1028',
    'plate_stopper': 'Пластина-упор',
    'front_stopper': 'Передній упор',
    'rear_stopper': 'Задній упор'
}
part_options_raw = sorted(df["part_name"].dropna().unique().tolist())
part_options_ukr = [part_name_mapping.get(part, part) for part in part_options_raw]
reverse_mapping = {part_name_mapping.get(k, k): k for k in part_options_raw}
selected_parts_ukr = st.sidebar.multiselect("Виберіть деталь:", ["Усі"] + part_options_ukr, default=["Усі"])
if "Усі" in selected_parts_ukr:
    selected_parts = part_options_raw
else:
    selected_parts = [reverse_mapping[name] for name in selected_parts_ukr]

# Фильтр по году
years = sorted(df["year"].dropna().unique().tolist())
selected_years = st.sidebar.multiselect("Рiк:", ["Усі"] + years, default=["Усі"])
if "Усі" in selected_years:
    selected_years = years

# Фильтр по месяцу
months = sorted(df["month"].dropna().unique())
month_names = {1: "Ciчень", 2: "Лютий", 3: "Березень", 4: "Квiтень", 5: "Травень", 6: "Червень",
               7: "Липень", 8: "Серпень", 9: "Вересень", 10: "Жовтень", 11: "Листопад", 12: "Грудень"}
month_labels = ["Усі"] + [month_names[m] for m in months]
selected_month_labels = st.sidebar.multiselect("Мiсяць:", month_labels, default=["Усі"])

# Преобразуем выбранные месяцы обратно в числа
if "Усі" in selected_month_labels:
    selected_months = months
else:
    selected_months = [k for k, v in month_names.items() if v in selected_month_labels]

# ---- ФИЛЬТРАЦИЯ ----

filtered_df = df[
    (df["part_name"].isin(selected_parts)) &
    (df["year"].isin(selected_years)) &
    (df["month"].isin(selected_months))
]

# ---- ТАБЫ (ВКЛАДКИ) ----

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Динаміка по місяцях",
    "Обрубники",
    "Брак за тижнями",
    "t°C заливки vs % браку",
    "Закінчені плавки"
])

# ---- ОСНОВНОЙ КОНТЕНТ ----

with tab1:

    # 1. Линия динамики по времени
    st.markdown("<h3 style='font-size:20px;'>Динаміка браку по місяцях</h3>", unsafe_allow_html=True)

    # Генерируем полный список месяцев от мин до макс
    all_months = pd.date_range(
        start=filtered_df["molding_date"].min().to_period("M").to_timestamp(),
        end=filtered_df["molding_date"].max().to_period("M").to_timestamp(),
        freq="MS"
    )

    # Группировка по месяцам
    filtered_df["month_start"] = filtered_df["molding_date"].dt.to_period("M").dt.to_timestamp()
    monthly_data = (
        filtered_df.groupby("month_start")
        .agg(total=("cast_id", "count"), defects=("rejected", "sum"))
        .reindex(all_months, fill_value=0) 
        .reset_index()
        .rename(columns={"index": "month_start"})
    )

    # Вычисляем процент брака

    monthly_data["defect_percent"] = monthly_data.apply(
        lambda row: 0 if row["total"] == 0 else 100 * row["defects"] / row["total"],
        axis=1
    )
    monthly_data["defect_label"] = monthly_data["defect_percent"].round(1).astype(str) + "%"

    # Показываем подписи только каждые 6 месяцев
    tickvals = pd.date_range(
        start=monthly_data["month_start"].min(),
        end=monthly_data["month_start"].max(),
        freq="6MS"
    )

    # Создаём фигуру с двумя типами графиков
    fig1 = go.Figure()

    # 1. Столбики — количество отливок (на вторую ось)
    fig1.add_trace(go.Bar(
        x=monthly_data["month_start"],
        y=monthly_data["total"],
        name="К-сть відливок",
        marker_color="lightgray",
        opacity=0.4,
        yaxis="y2"
    ))

    # 2. Линия — % брака
    fig1.add_trace(go.Scatter(
        x=monthly_data["month_start"],
        y=monthly_data["defect_percent"],
        mode="lines+markers",
        name="% браку",
        marker=dict(color="orange"),
        line=dict(color="orange"),
        hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>"
    ))

    # Настройки осей
    fig1.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            tickformat="%b %Y",
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title="% браку",
            rangemode="tozero",
            side="left"
        ),
        yaxis2=dict(
            title="К-сть відливок",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(t=30),
        height=500
    )

    # Выводим график
    st.plotly_chart(fig1, use_container_width=False)


    # --- Нижняя часть: два графика рядом ---
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.markdown("<h3 style='font-size:18px;'>🔝 Топ-10 дефектiв</h3>", unsafe_allow_html=True)

        total_casts = filtered_df["cast_id"].nunique()

        top_defects = (
            filtered_df[filtered_df["rejected"] == 1]
            .groupby("defects")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )

        top_defects["percent"] = 100 * top_defects["count"] / total_casts
        top_defects = top_defects.sort_values("percent", ascending=True)

        fig2 = px.bar(
            top_defects,
            x="percent",
            y="defects",
            orientation="h",
            labels={"percent": "% браку", "defects": ""},
            text=top_defects["percent"].apply(lambda x: f"{x:.1f}%"),
            color_discrete_sequence=["#6BA5A4"]
        )

        # Настройки графика
        fig2.update_traces(
            textposition="outside",
            textfont_size=10  #  размер текста 
        )

        # Максимальное значение + запас (30–50%) для текста
        max_percent = top_defects["percent"].max()
        x_range_max = max_percent * 1.5  # Можешь поиграться с этим коэффициентом

        fig2.update_layout(
            xaxis=dict(
                tickformat=".1f",
                range=[0, x_range_max]
            ),
            width=900,
            height=400,
            margin=dict(r=10, l=150)  # убираем лишний отступ справа
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("<h3 style='font-size:18px;'>Виннi</h3>", unsafe_allow_html=True)

        # Подготовка данных
        filtered_df["vinovn"] = (
            filtered_df["vinovn"]
            .fillna("—")
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.upper()
        )

        defective_df = filtered_df[filtered_df["rejected"] == 1].copy()
        defective_df = defective_df.drop_duplicates(subset=["cast_id"])
        total_defects = defective_df["cast_id"].nunique()

        vinovn_stats = (
            filtered_df.groupby(["vinovn"], dropna=False)
            .agg(defects=("rejected", "sum"))
            .reset_index()
        )
        vinovn_stats["defect_percent"] = 100 * vinovn_stats["defects"] / total_defects
        vinovn_stats = vinovn_stats.sort_values("defect_percent", ascending=False)

        # Оставим топ 8 виновных, остальных объединим в "Інші"
        top_n = 8
        top_vinovn = vinovn_stats.head(top_n).copy()
        other_sum = vinovn_stats["defects"].iloc[top_n:].sum()
        other_percent = vinovn_stats["defect_percent"].iloc[top_n:].sum()

        if other_sum > 0:
            other_row = pd.DataFrame([{
                "vinovn": "ІНШІ",
                "defects": other_sum,
                "defect_percent": other_percent
            }])
            top_vinovn = pd.concat([top_vinovn, other_row], ignore_index=True)

        colors = [
        "#6BA5A4",  # Серо-бирюзовый (основной фирменный)
        "#F4A261",  # Светло-оранжевый (акцент)
        "#E76F51",  # Терракотово-красный (ошибки)
        "#2A9D8F",  # Бирюзовый
        "#264653",  # Тёмно-синий (нейтральный)
        "#E9C46A",  # Жёлтый-песочный (внимание)
        "#A8DADC",  # Голубой холодный
        "#457B9D",  # Синий стальной
        "#B5838D"   # Пыльно-розовый (на фоне нейтральных хорошо смотрится)
        ]

        # Пончиковая диаграмма
        fig3 = go.Figure()
        fig3.add_trace(go.Pie(
            labels=top_vinovn["vinovn"],
            values=top_vinovn["defects"],
            hole=0.4,
            textinfo="label+percent",
            insidetextorientation="radial",
            marker=dict(colors=colors[:len(top_vinovn)])
        ))
        fig3.update_layout(
            height=400,
            margin=dict(t=30),
            showlegend=True
        )
        st.plotly_chart(fig3, use_container_width=True)

#------------------------------------------------------------------------------------------

with tab2:
    st.write("Test deploy")
   

#---------------------------------------------------------------------------------------
with tab3:

    # 3. Дефекты в плавках за последние 7-8 недель

    #Заголовок
    st.markdown("<h3 style='font-size:20px;'>Середня кiлькiсть браку на плавку за останнi 8 тижнiв</h3>", unsafe_allow_html=True)


    filtered_df['pour_date'] = pd.to_datetime(df['pour_date'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['pour_date', 'melt_num', 'rejected'])

    # Группируем по плавкам
    melt_defects = filtered_df.groupby('melt_num').agg({
        'rejected': 'sum',
        'pour_date': 'first'
    }).reset_index()

    # Находим дату начала недели
    melt_defects['week_start'] = melt_defects['pour_date'] - pd.to_timedelta(melt_defects['pour_date'].dt.weekday, unit='d')

    # Группируем по началу недели
    weekly_defects = melt_defects.groupby('week_start')['rejected'].mean().reset_index()
    weekly_defects = weekly_defects.sort_values('week_start')

    # Оставляем только последние 8 недель
    last_8_weeks = weekly_defects.tail(8)

    # Вычисляем общее среднее за эти 8 недель
    overall_mean = last_8_weeks['rejected'].mean()

    fig = go.Figure()

    # Столбцы
    fig.add_trace(go.Bar(
        x=last_8_weeks['week_start'],
        y=last_8_weeks['rejected'],
        name='Середня кiлькiсть браку на плавку',
        marker_color='royalblue',
    ))

    # Линия среднего
    fig.add_trace(go.Scatter(
        x=last_8_weeks['week_start'],
        y=[overall_mean]*len(last_8_weeks),
        mode='lines',
        name=f'Середнє ({overall_mean:.2f})',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title="",
        xaxis_title="Початок тижня",
        yaxis_title="Середня кiлькiсть браку на плавку",
        legend_title="",
        bargap=0.3,
        template="simple_white",
        xaxis_tickformat="%d-%m-%Y" 
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------------
with tab4:

    st.markdown("<h3 style='font-size:20px;'>Температура заливки vs % браку (помiсячно)</h3>", unsafe_allow_html=True)

    # Преобразуем даты к месяцу
    filtered_df['month'] = filtered_df['pour_date'].dt.to_period('M').dt.to_timestamp()

    # Агрегация по месяцам
    monthly_stats = (
        filtered_df.groupby('month')
        .agg(
            avg_temp=('tempk', 'mean'),
            total_casts=('cast_id', 'count'),
            defects=('rejected', 'sum')
        )
        .reset_index()
    )

    # Создание полного диапазона месяцев
    full_range = pd.date_range(
        start=filtered_df['month'].min(),
        end=filtered_df['month'].max(),
        freq='MS'  # "Month Start"
    )

    # Приводим к полному диапазону месяцев
    monthly_stats = monthly_stats.set_index('month').reindex(full_range).reset_index().rename(columns={'index': 'month'})

    # Заполняем отсутствующие значения
    monthly_stats['defects'] = monthly_stats['defects'].fillna(0)
    monthly_stats['total_casts'] = monthly_stats['total_casts'].fillna(0)
    monthly_stats['avg_temp'] = monthly_stats['avg_temp']  # можно оставить NaN

    # Расчёт % брака
    monthly_stats['defect_percent'] = monthly_stats.apply(
        lambda row: 100 * row['defects'] / row['total_casts'] if row['total_casts'] > 0 else 0,
        axis=1
    )

    # График
    fig_temp = go.Figure()

    # Температура
    fig_temp.add_trace(go.Scatter(
        x=monthly_stats['month'],
        y=monthly_stats['avg_temp'],
        mode='lines+markers',
        name='Середня температура заливки',
        yaxis='y2',
        line=dict(color='royalblue'),
        connectgaps=False,
        hovertemplate='%{x|%b %Y}<br>Темп: %{y:.0f} °C<extra></extra>'
    ))

    # % брака
    fig_temp.add_trace(go.Scatter(
        x=monthly_stats['month'],
        y=monthly_stats['defect_percent'],
        mode='lines+markers',
        name='% браку',
        line=dict(color='orange'),
        hovertemplate='%{x|%b %Y}<br>Брак: %{y:.1f}%<extra></extra>'
    ))

    # Настройка осей
    fig_temp.update_layout(
        xaxis=dict(
            title='Місяць',
            tickformat='%b %Y',
            tickangle=-45,
            dtick="M3"
        ),
        yaxis=dict(title='% браку', rangemode='tozero'),
        yaxis2=dict(
            title='Середня температура заливки',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(orientation='h', y=1.15),
        height=500,
        margin=dict(t=30),
        template="simple_white"
    )

    st.plotly_chart(fig_temp, use_container_width=True)

# -------------------------------------------------------------------------------------

with tab5:
    # Матрица с браком по плавкам

    st.markdown("<h3 style='font-size:20px;'>📈 Статистика по закінченим плавкам (поточний рiк)</h3>", unsafe_allow_html=True)

    # --- 1. Фільтруємо поточний рік ---
    filtered_df['pour_date'] = pd.to_datetime(filtered_df['pour_date'], errors='coerce')
    current_year = pd.Timestamp.now().year
    filtered_df_current = filtered_df[filtered_df['pour_date'].dt.year == current_year].copy()

    # --- 2. Визначаємо, чи деталь оброблена (випущена або забракована) ---
    filtered_df_current['is_finished'] = (
        (filtered_df_current['rejected'] == 1) |
        (filtered_df_current[['datas1', 'datas2', 'datas3']].notnull().any(axis=1))
    )

    # --- 3. Групуємо по плавках ---
    melt_group = filtered_df_current.groupby('melt_num').agg(
        total_parts=('cast_id', 'count'),
        finished_parts=('is_finished', 'sum')
    ).reset_index()

    # --- 4. Залишаємо тільки ті плавки, де всі деталі оброблені ---
    melt_group['all_finished'] = melt_group['total_parts'] == melt_group['finished_parts']
    finished_melts = melt_group[melt_group['all_finished']]['melt_num']

    # --- 5. Повертаємось назад до деталей, але тільки для "закінчених" плавок ---
    final_details = filtered_df_current[filtered_df_current['melt_num'].isin(finished_melts)].copy()

    # --- 6. Для кожної плавки рахуємо відсоток забракованих ---
    melt_defect_stats = final_details.groupby('melt_num').agg(
        total_parts=('cast_id', 'count'),
        finished_parts=('is_finished', 'sum'),   # добавляем сюда количество обработанных деталей
        defects=('rejected', 'sum')
    ).reset_index()

    # Рахуємо відсоток браку
    melt_defect_stats['defect_percent'] = 100 * melt_defect_stats['defects'] / melt_defect_stats['total_parts']

    # --- 7. Сортуємо за спаданням відсотка браку ---
    melt_defect_stats = melt_defect_stats.sort_values(by='defect_percent', ascending=False)

    # --- 8. Створюємо фінальну таблицю ---
    matrix = melt_defect_stats[['melt_num', 'finished_parts', 'defect_percent']]


    # --- 9. Стилізуємо таблицю ---
    styled_matrix = matrix.style.background_gradient(
        subset=["defect_percent"],
        cmap="Reds",
        low=0,
        high=1
    ).format({"defect_percent": "{:.1f}%"})

    # --- 10. Відображаємо з прокруткою ---
    st.dataframe(styled_matrix, use_container_width=True)


# -------------------------------------------------------------------------------------

