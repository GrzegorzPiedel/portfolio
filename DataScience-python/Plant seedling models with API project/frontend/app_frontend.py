import streamlit as st
import requests

st.title("Plant Seedlings Classifier")

# pozwala użytkownikowi przesłać plik z komputera
uploaded_file = st.file_uploader(
    'Wybierz zdjęcie sadzonki',
    type = ['jpg', 'jpeg', 'png'],
)

# wyświetla wgrane zdjęcie po jego zuploadowaniu
if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption = "WYbrane zdjęcie",
        use_container_width = True
    )

    # kod wewnątrz tego bloku wykona się tylko po kliknięciu przycisku
    if st.button("Klasyfikuj"):

        # - Tworzysz słownik files, który jest przygotowany do wysłania do API (przez requests.post w
        # późniejszej części kodu).
        # - Struktura słownika jest wymagana przez bibliotekę requests przy przesyłaniu plików
        files = {
            "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
        }

        # Składnia with ...: → wszystko, co znajduje się w tym bloku, będzie wykonywane podczas wyświetlania spinnera.
        with st.spinner("Przetwarzanie..."):

            # requests.post zwraca obiekt response, który zawiera:
            # - esponse.status_code – kod HTTP (200 = OK)
            # - response.json() – odpowiedź w formacie JSON (jeśli serwer zwraca JSON)
            # - response.text – surowa treść odpowiedzi
            response = requests.post("http://localhost:8000/predict", files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(f"Klasa: {data['class']}")
            st.write(f"Pewność: {data['confidence']:.2f}")
            st.write(response.json())
            st.write(response.text)
        else:
            st.error("Coś poszło nie tak, spróbuj ponownie")