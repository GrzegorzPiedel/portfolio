import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Ceny nieruchomości w USA", page_icon="🏠")


# Dzięki @st.cache_data dane z pliku Housing.csv zostaną załadowane tylko raz, a przy kolejnych uruchomieniach
# tej funkcji, Streamlit użyje danych z pamięci podręcznej
@st.cache_data
def load_data():
    df = pd.read_csv('Housing.csv')
    df = df.drop(columns=['hotwaterheating', 'prefarea'])
    df.columns = ["Cena", 'Powierzchnia', 'Pokoje', 'Łazienki', 'Piętra', 'Połączenie z drogą główną', 'Pokój gościnny',
                  'Piwnica', 'Klimatyzacja', 'Miejsca parkingowe', 'Umeblowanie']

    df['Umeblowanie'] = df['Umeblowanie'].replace({
        'furnished': 'Umeblowane',
        'semi-furnished': 'Częściowo umeblowane',
        'unfurnished': 'Nieumeblowane'
    })

    df[['Połączenie z drogą główną', 'Pokój gościnny', 'Piwnica', 'Klimatyzacja']] = df[
        ['Połączenie z drogą główną', 'Pokój gościnny', 'Piwnica', 'Klimatyzacja']].replace({
        'yes': 'Tak',
        'no': 'Nie'
    })
    return df


df = load_data()

st.sidebar.title("Nawigacja")

# umożliwia wybór jednej opcji z listy w panelu bocznym
page = st.sidebar.selectbox("Wybierz stronę:", ["Wprowadzenie", "Eksploracja danych", "Model"])

min_value = df["Powierzchnia"].min()
max_value = df["Powierzchnia"].max()

# =============
# WPROWADZENIE
# =============

if page == "Wprowadzenie":
    st.title("Wprowadzenie")
    st.markdown(""" 
        Dashboard prezentuje interaktywną analizę cen domów na podstawie danych z Kaggle.
        
        Dane zawierają informacje o nieruchomościach: metraż, liczba pokoi, łazienek, liczba pięter i inne.
        
        W ramach raportu:
        - Zbadamy czynniki wpływające na cenę domu
        - Ocenimy zależności między powierzchnią mieszkania a ceną
        - Przeanalizujemy wpływ liczby pokoi, łazienek i innych cech na wycenę nieruchomości
        --- 

        Dane pochodzą ze strony Kaggle [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/code)
    """)

elif page == "Eksploracja danych":
    st.title("Eksploracja danych")
    st.sidebar.header("Filtry danych")

    # =============
    # FILTRY
    # =============

    # Ustalamy min i max dla suwaka
    min_value = df["Powierzchnia"].min()
    max_value = df["Powierzchnia"].max()

    # Ręczne wpisanie wartości minimalnej i maksymalnej powierzchni
    min_input = st.sidebar.number_input(
        "Minimalna powierzchnia (sq ft)",
        min_value=min_value,
        max_value=max_value,
        value=min_value
    )

    max_input = st.sidebar.number_input(
        "Maksymalna powierzchnia (sq ft)",
        min_value=min_input,
        max_value=max_value,
        value=max_value
    )

    # Umożliwiamy użytkownikowi wybór zakresu przez suwak
    min_powierzchnia, max_powierzchnia = st.sidebar.slider(
        "Wybierz zakres powierzchni (sq ft)",
        min_value=min_value,
        max_value=max_value,
        value=(min_input, max_input)
    )

    st.sidebar.markdown("---")

    # pozostałe filtry
    pokoje = st.sidebar.multiselect(
        label="Liczba pokoi",
        options=sorted(df["Pokoje"].unique()),
        default=sorted(df["Pokoje"].unique())  # Domyślnie wszystkie opcje są zaznaczone
    )

    lazienki = st.sidebar.multiselect(
        label="Liczba łazienek",
        options=sorted(df["Łazienki"].unique()),
        default=sorted(df["Łazienki"].unique())  # Domyślnie wszystkie opcje są zaznaczone
    )

    pietra = st.sidebar.multiselect(
        label="Liczba pięter",
        options=sorted(df["Piętra"].unique()),
        default=sorted(df["Piętra"].unique())
    )

    umeblowanie = st.sidebar.multiselect(
        label="Umeblowanie",
        options=sorted(df["Umeblowanie"].unique()),
        default=sorted(df["Umeblowanie"].unique())
    )

    filtered_data = df[
        (df["Powierzchnia"].between(min_powierzchnia, max_powierzchnia)) &
        (df["Pokoje"].isin(pokoje) if pokoje else df["Pokoje"]) &
        (df["Łazienki"].isin(lazienki) if lazienki else df["Łazienki"]) &
        (df["Piętra"].isin(pietra) if pietra else df["Piętra"]) &
        (df["Umeblowanie"].isin(umeblowanie) if umeblowanie else df["Umeblowanie"])
        ]

    st.sidebar.markdown(f"**Liczba domów spełniających kryteria: {len(filtered_data)}**")

    # ------ HISTOGRAM
    hist = px.histogram(
        filtered_data,
        x="Cena",
        nbins=40,
        title="Rozkład cen po filtrze",
        text_auto=True,
    )
    hist.update_traces(
        marker_line_color='black',
        marker_line_width=1,
        textposition='outside'
    )
    hist.update_layout(
        yaxis_title='Liczba mieszkań',
        xaxis_title='Cena (USD)'
    )
    st.plotly_chart(hist, use_container_width=True)

    # ------ SCATTER
    st.plotly_chart(
        px.scatter(
            filtered_data,
            x="Powierzchnia",
            y="Cena",
            trendline="ols",
            title="Zależność pomiędzy ceną a powierzchnią",
            color="Cena",
            color_continuous_scale="Plasma",
            trendline_color_override="Red"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.scatter(
            filtered_data,
            x="Pokoje",
            y="Cena",
            title="Zależność ceny od liczby pokoi",
            color="Cena",
            color_continuous_scale="Plasma"
        ),
        use_container_width=True
    )

    st.plotly_chart(
        px.box(
            filtered_data,
            x="Piętra",
            y="Cena",
            title="Cena a liczba pięter"
        ),
        use_container_width=True
    )

# =============
# MODEL
# =============

elif page == "Model":
    st.title("Predykcja cen domu (regresja)")
    st.markdown("""
    Model regresji liniowej na podstawie wybranych zmiennych:
    - Powierzchnia
    - Pokoje
    - Łazienki
    - Piętra
    - Połączenie z drogą główną
    - Pokój gościnny
    - Piwnica
    - Klimatyzacja
    - Miejsca parkingowe
    - Umeblowanie
    """)

    # Przygotowanie danych i modelu
    X = df.drop(columns=["Cena"])
    y = df["Cena"]

    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()

    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.markdown(f"""
    **Wyniki modelu:**
    - R² score: {r2_score(y_test, y_pred):.3f}
    - MAE: {mean_absolute_error(y_test, y_pred):.0f} USD
    """)

    st.subheader("Predykcja vs rzeczywiste ceny")

    pred_df = pd.DataFrame({
        "Rzeczywista cena": y_test,
        "Predykcja": y_pred
    })

    # Scatterplot
    fig = px.scatter(
        pred_df,
        x="Rzeczywista cena",
        y="Predykcja",
        trendline="ols",
        labels={
            "Rzeczywista cena": "Rzeczywista cena (USD)",
            "Predykcja": "Predykcja (USD)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # ====================================
    # Interaktywna część dla użytkownika
    # ====================================
    st.subheader("Wprowadź dane i zobacz prognozę")

    area_input = st.number_input("Powierzchnia mieszkalna", df["Powierzchnia"].min(), df["Powierzchnia"].max())
    bath_input = st.selectbox("Liczba łazienek", sorted(df["Łazienki"].unique()))
    room_input = st.selectbox("Liczba pokoi", sorted(df["Pokoje"].unique()))
    furnishing_input = st.selectbox("Umeblowanie", df["Umeblowanie"].unique())
    climate_input = st.selectbox("Klimatyzacja", df["Klimatyzacja"].unique())
    basement_input = st.selectbox("Piwnica", df["Piwnica"].unique())
    guest_room_input = st.selectbox("Pokój gościnny", df["Pokój gościnny"].unique())
    main_road_input = st.selectbox("Połączenie z drogą główną", df["Połączenie z drogą główną"].unique())

    # Przygotowanie danych wejściowych
    user_input = pd.DataFrame({
        'Powierzchnia': [area_input],
        'Pokoje': [room_input],
        'Łazienki': [bath_input],
        'Umeblowanie_Umeblowane': [1 if furnishing_input == 'Umeblowane' else 0],
        'Umeblowanie_Częściowo umeblowane': [1 if furnishing_input == 'Częściowo umeblowane' else 0],
        'Umeblowanie_Nieumeblowane': [1 if furnishing_input == 'Nieumeblowane' else 0],
        'Klimatyzacja_Tak': [1 if climate_input == 'Tak' else 0],
        'Piwnica_Tak': [1 if basement_input == 'Tak' else 0],
        'Pokój gościnny_Tak': [1 if guest_room_input == 'Tak' else 0],
        'Połączenie z drogą główną_Tak': [1 if main_road_input == 'Tak' else 0]
    })

    # Dopasowanie dummies do danych wejściowych
    user_input = pd.get_dummies(user_input, drop_first=True)

    #Dopasowanie kolumn - nie brakujących kolumn, które występują w X
    missing_cols = set(X.columns) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0  # Dodaj brakujące kolumny z wartością 0

    #Przekształcenie danych wejściowych w celu dopasowania do modelu
    #Przypisanie kolumn w takiej samej kolejności jak w X
    user_input = user_input[X.columns]

    # Skalowanie danych wejściowych (skalowanie bo było też skalowane w X)
    user_input[numeric_columns] = scaler.transform(user_input[numeric_columns])

    pred_price = model.predict(user_input)[0]
    st.success(f"Prognozowana cena domu: {pred_price:.2f} USD")
