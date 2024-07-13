import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Fungsi untuk memprediksi kategori berdasarkan input float
def predict_category(model, input_data, scaler):
    # Lakukan normalisasi input menggunakan MinMaxScaler
    # input_scaled = scaler.transform([input_data])

    # Lakukan prediksi menggunakan model
    prediction = model.predict(input_data)
    return prediction[0]  # Kembalikan nilai prediksi kategori

# Memuat model klasifikasi dan MinMaxScaler dari file pickle
@st.cache_data()  # Menyimpan model di cache untuk performa
def load_classification_model():
    with open('rf_classifier.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as scale:
        scaler = pickle.load(scale)
    return model, scaler

# Tampilan halaman untuk model klasifikasi
def classification_page():
    st.title('Classification Model')

    # Memuat model klasifikasi dan MinMaxScaler
    model, scaler = load_classification_model()

    # Input 7 nilai float
    st.header('Input Features (7 Float Values)')
    suhu = st.number_input('Suhu', value=0.0)
    arah = st.number_input('Arah', value=0.0)
    kecepatan = st.number_input('Kecepatan', value=0.0)
    rh = st.number_input('RH', value=0.0)
    tekanan = st.number_input('Tekanan', value=0.0)
    ch = st.number_input('CH', value=0.0)
    wl = st.number_input('WL', value=0.0)

    # Predict button to trigger classification
    if st.button('Predict'):
        # Gather input features into an array
        input_data = [suhu, arah, kecepatan, rh, tekanan, ch, wl]

        # Normalisasi input menggunakan MinMaxScaler yang telah dimuat
        input_array = scaler.transform([input_data])

        # Perform prediction
        category_index = predict_category(model, input_array, scaler)

        # Map category index to corresponding label
        category_labels = {0: 'Aman', 1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}
        predicted_category = category_labels[category_index]

        # Display predicted category
        st.success(f'Predicted Category: {predicted_category}')

# Fungsi untuk memprediksi menggunakan model regresi untuk satu titik waktu
def predict_regression(model, date_time):
    # Bentuk input untuk prediksi berdasarkan titik waktu
    year = date_time.year
    month = date_time.month
    day = date_time.day
    hour = date_time.hour

    # Lakukan prediksi menggunakan model regresi
    prediction = model.predict([[year, month, day, hour]])

    return prediction[0]  # Kembalikan nilai prediksi regresi

# Memuat model regresi dari file pickle
@st.cache(allow_output_mutation=True)  # Menyimpan model di cache untuk performa
def load_regression_model():
    with open('model_regression(1).pkl', 'rb') as file:
        model = pickle.load(file)
    return model


# Tampilan apstlikasi Streamlit dengan tabbed layout
def main():
    # Menampilkan gambar dari data yang telah di-cache
    st.title('Sistem Prediksi Banjir Rob Kota Pontianak')

    # Add a selectbox to the sidebar to select the page to display
    page = st.sidebar.selectbox("Select a page", ["Sistem Prediksi", "Sistem Klasifikasi"])

    if page == "Sistem Prediksi":
        st.sidebar.header('Input Date Range')
        start_date = st.sidebar.date_input('Start Date')
        end_date = st.sidebar.date_input('End Date')

        if st.sidebar.button('Predict'):
            st.write(f'Predicting for date range: {start_date} to {end_date}')

            # Memuat model regresi
            regression_model = load_regression_model()

            # Generate list semua titik waktu dalam rentang
            time_range = pd.date_range(start=start_date, end=end_date, freq='H')

            # Lakukan prediksi untuk setiap titik waktu
            predictions = []
            for timestamp in time_range:
                prediction = predict_regression(regression_model, timestamp)
                predictions.append(prediction)

            # Tampilkan hasil prediksi
            df_predictions = pd.DataFrame({
                'Timestamp': time_range,
                'Prediction': predictions
            })
            df_predictions['Convert'] = 240-df_predictions['Prediction']
            st.write(df_predictions)

            # Plot hasil prediksi regresi
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_range, predictions, marker='o', linestyle='-', color='b')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Prediction')
            ax.set_title('Time Series Regression Prediction')
            plt.xticks(rotation=45)  # Rotasi label sumbu x untuk memperbaiki tampilan
            # Menambahkan garis horizontal pada nilai 240 di sumbu Y
            ax.axhline(y=240, color='r', linestyle='--', linewidth=1)
            st.pyplot(fig)  # Menampilkan plot di aplikasi Streamlit

    elif page == "Sistem Klasifikasi":
        classification_page()

if __name__ == '__main__':
    main()