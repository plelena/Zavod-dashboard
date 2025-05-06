import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import to_hex as rgb2hex
from sqlalchemy import create_engine
import psycopg2
import sqlite3


# ---- ЗАГРУЗКА ДАННЫХ ----

@st.cache_data
def load_data():
    # df = pd.read_excel("defects.xlsx") - использовалось ранее
    
    # # !!!!!!параметры подключения  для PostgreSQL
    # engine = create_engine("postgresql+psycopg2://postgres:@localhost:5432/defects")
    # query = "SELECT * FROM defects" 
    # df = pd.read_sql(query, engine)

    # Подключение SQLite
    conn = sqlite3.connect("defects.sqlite")
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
part_options = sorted(df["part_name"].dropna().unique().tolist())
selected_parts = st.sidebar.multiselect("Виберіть деталь:", ["All"] + part_options, default=["All"])
if "All" in selected_parts:
    selected_parts = part_options

# Фильтр по году
years = sorted(df["year"].dropna().unique().tolist())
selected_years = st.sidebar.multiselect("Рiк:", ["All"] + years, default=["All"])
if "All" in selected_years:
    selected_years = years

# Фильтр по месяцу
months = sorted(df["month"].dropna().unique())
month_names = {1: "Ciчень", 2: "Лютий", 3: "Березень", 4: "Квiтень", 5: "Травень", 6: "Червень",
               7: "Липень", 8: "Серпень", 9: "Вересень", 10: "Жовтень", 11: "Листопад", 12: "Грудень"}
month_labels = ["All"] + [month_names[m] for m in months]
selected_month_labels = st.sidebar.multiselect("Мiсяць:", month_labels, default=["All"])

# Преобразуем выбранные месяцы обратно в числа
if "All" in selected_month_labels:
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Динаміка по місяцях",
    "Топ-10 дефектів",
    "Брак за тижнями",
    "t°C заливки vs % браку",
    "Закінчені плавки",
    "Виннi"
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

#------------------------------------------------------------------------------------------

with tab2:

    # 2. Топ-10 дефектов
    st.markdown("<h3 style='font-size:20px;'>🔝 Топ-10 дефектiв</h3>", unsafe_allow_html=True)

    # Всего отливок 
    total_casts = filtered_df["cast_id"].nunique() 

    # Считаем дефекты
    top_defects = (
        filtered_df[filtered_df["rejected"] == 1]
        .groupby("defects")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )

    # Добавляем столбец с процентами от общего числа отливок
    top_defects["percent"] = 100 * top_defects["count"] / total_casts
    top_defects = top_defects.sort_values("percent", ascending=True)

    # Строим график
    fig2 = px.bar(
        top_defects,
        x="percent",
        y="defects",
        orientation="h",
        labels={"percent": "% браку", "defects": "Тип дефекта"},
        text=top_defects["percent"].apply(lambda x: f"{x:.1f}%"),
        color_discrete_sequence=["#6BA5A4"] 
    )

    fig2.update_traces(textposition="outside")
    fig2.update_layout(xaxis_tickformat=".1f")

    st.plotly_chart(fig2, use_container_width=True)

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

    # # 4. По участкам и сменам (было ранее))

    # # Заголовок
    # st.markdown("<h3 style='font-size:20px;'>Дефекти по ділянках і змінах</h3>", unsafe_allow_html=True)

    # # Группировка по бригадам
    # brig_stats = (
    #     filtered_df.groupby(["brig"], dropna=False)
    #     .agg(total=("cast_id", "count"), defects=("rejected", "sum"))
    #     .reset_index()
    # )
    # brig_stats["defect_percent"] = 100 * brig_stats["defects"] / brig_stats["total"]
    # brig_stats = brig_stats.sort_values("defect_percent", ascending=False)

    # # Генерация градиентных цветов
    # norm = plt.Normalize(vmin=brig_stats["defect_percent"].min(), vmax=brig_stats["defect_percent"].max())
    # cmap = plt.cm.get_cmap("RdYlGn_r")

    # def get_color(val):
    #     rgba = cmap(norm(val))
    #     r, g, b, a = rgba  # раскладываем RGBA
    #     a = 0.5  # задаем свою прозрачность (например 0.5 = 50% прозрачности)
    #     return f"background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a});"

    # # Применяем стили
    # styled_table = brig_stats[["brig","defect_percent"]].style.applymap(get_color, subset=["defect_percent"]).format({"defect_percent": "{:.1f}%"})

    # # Показываем без прокрутки
    # st.dataframe(styled_table, use_container_width=True, height=len(brig_stats)*35 + 35)

    # 4. Изменения температуры заливки и брак

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

with tab6:

    # 4. По виновным

    # Заголовок
    st.markdown("<h3 style='font-size:20px;'>Брак за винними </h3>", unsafe_allow_html=True)

    filtered_df["vinovn"] = (
    filtered_df["vinovn"]
    .fillna("—")  # на случай пропусков
    .astype(str)  # если вдруг были не строки
    .str.strip()  # убрать пробелы в начале и конце
    .str.replace(r"\s+", " ", regex=True)  # нормализовать множественные пробелы
    .str.upper()  # если нужно всё в верхнем регистре
    )

    defective_df = filtered_df[filtered_df["rejected"] == 1].copy()
    defective_df = defective_df.drop_duplicates(subset=["cast_id"])
    total_defects = defective_df["cast_id"].nunique()

    # Группировка по виновным
    vinovn_stats = (
        filtered_df.groupby(["vinovn"], dropna=False)
        .agg(defects=("rejected", "sum"))
        .reset_index()
    )
    vinovn_stats["defect_percent"] = 100 * vinovn_stats["defects"] / total_defects
    vinovn_stats = vinovn_stats.sort_values("defect_percent", ascending=False)

    # Генерация градиентных цветов
    norm = plt.Normalize(vmin=vinovn_stats["defect_percent"].min(), vmax=vinovn_stats["defect_percent"].max())
    cmap = plt.cm.get_cmap("RdYlGn_r")


    def get_color(val):
        rgba = cmap(norm(val))
        r, g, b, a = rgba  # раскладываем RGBA
        a = 0.5  # задаем свою прозрачность (например 0.5 = 50% прозрачности)
        return f"background-color: rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a});"

    # Применяем стили
    styled_table = vinovn_stats[["vinovn","defect_percent"]].style.applymap(get_color, subset=["defect_percent"]).format({"defect_percent": "{:.1f}%"})

    # Показываем без прокрутки
    table_height = len(vinovn_stats) * 35 + 35
    st.dataframe(styled_table, use_container_width=True, height=table_height)
