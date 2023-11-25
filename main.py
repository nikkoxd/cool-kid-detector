import streamlit as st
import pandas as pd
import joblib

# Загрузка готовой модели
model = joblib.load('knn_test.pkl')

# Создание заголовка и описания приложения
st.title('Ваше Streamlit приложение')
st.write('Введите данные для получения предсказания')

# Создание формы для ввода данных
user_input = []
for i in range(10):
    user_input.append(st.number_input(f'Введите значение {i+1}', value=0.0))

# Преобразование пользовательского ввода в DataFrame
input_data = pd.DataFrame([user_input])

# Отображение пользовательского ввода
st.write('Введенные данные:')
st.write(input_data)

# Предсказание с помощью загруженной модели
if st.button('Получить предсказание'):
    prediction = model.predict(input_data)
    st.write('Предсказанное значение:')
    st.write(prediction)
