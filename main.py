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
st.caption('Первый год')
user_input.append(st.number_input('Математика', value=0.0, key='math1'));
user_input.append(st.number_input('Русский язык', value=0.0, key='rus1'));
user_input.append(st.number_input('Биология', value=0.0, key='bio1'));
user_input.append(st.number_input('Английский язык', value=0.0, key='eng1'));
user_input.append(st.number_input('География', value=0.0, key='geo1'));
user_input.append(st.number_input('Физика', value=0.0, key='fiz1'));
user_input.append(st.number_input('Химия', value=0.0, key='chem1'));
user_input.append(st.number_input('Обществознание', value=0.0, key='soc1'));
user_input.append(st.number_input('Информатика', value=0.0, key='info1'));
st.caption('Второй год')
user_input.append(st.number_input('Математика', value=0.0, key='math2'));
user_input.append(st.number_input('Русский язык', value=0.0, key='rus2'));
user_input.append(st.number_input('Биология', value=0.0, key='bio2'));
user_input.append(st.number_input('Английский язык', value=0.0, key='eng2'));
user_input.append(st.number_input('География', value=0.0, key='geo2'));
user_input.append(st.number_input('Физика', value=0.0, key='fiz2'));
user_input.append(st.number_input('Химия', value=0.0, key='chem2'));
user_input.append(st.number_input('Обществознание', value=0.0, key='soc2'));
user_input.append(st.number_input('Информатика', value=0.0, key='info2'));


# Преобразование пользовательского ввода в DataFrame
df1 = pd.DataFrame([user_input])
# Добавление столбца 'omathinfo'
# Добавление столбца 'omathinfo' с условием
df1['sum1'] = df1[0] + df1[1] + df1[2] + df1[3]+df1[4]+df1[5] + df1[6] + df1[7] + df1[8]
df1['sum2'] = df1[9] + df1[10] + df1[11] + df1[12]+df1[13]+df1[14] + df1[15] + df1[16] + df1[17]

df1['mathinfo1'] = ((df1[0] + df1[8]) / df1['sum1']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['mathinfo2'] = ((df1[9] + df1[17]) / df1['sum2']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['omathinfo'] = (df1['mathinfo1']+ df1['mathinfo2'])
df1 = df1.drop(['mathinfo1','mathinfo2'], axis=1)

df1['russoc1'] = ((df1[1] + df1[7]) / df1['sum1']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['russoc2'] = ((df1[10] + df1[16]) / df1['sum2']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['orussoc'] = (df1['russoc1']+ df1['russoc2'])
df1 = df1.drop(['russoc1','russoc2'], axis=1)

df1['fizmath1'] = ((df1[0] + df1[5]) / df1['sum1']).apply(lambda x: 1 if x > 0.243 else 0)
df1['fizmath2'] = ((df1[9] + df1[14]) / df1['sum2']).apply(lambda x: 1 if x > 0.243 else 0)
df1['ofizmath'] = (df1['fizmath1']+ df1['fizmath2'])
df1 = df1.drop(['fizmath1','fizmath2'], axis=1)

df1['biochem1'] = ((df1[2] + df1[6]) / df1['sum1']).apply(lambda x: 1 if x > 0.243 else 0)
df1['biochem2'] = ((df1[11] + df1[15]) / df1['sum2']).apply(lambda x: 1 if x > 0.243 else 0)
df1['obiochem'] = (df1['biochem1']+ df1['biochem2'])
df1 = df1.drop(['biochem1','biochem2'], axis=1)

df1['geo11'] = (df1[4] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['geo21'] = (df1[13] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['ogeo'] = (df1['geo11']+ df1['geo21'])
df1 = df1.drop(['geo11','geo21'], axis=1)

df1['chem11'] = (df1[6] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['chem21'] = (df1[15] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['ochem'] = (df1['chem11']+ df1['chem21'])
df1 = df1.drop(['chem11','chem21'], axis=1)

df1['eng11'] = (df1[3] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['eng21'] = (df1[12] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['oeng'] = (df1['eng11']+ df1['eng21'])
df1 = df1.drop(['eng11','eng21'], axis=1)

df1 = df1.drop(['sum1','sum2'],axis=1)
# df1['opotential'] = (df1['omathinfo']+ df1['orussoc']+df1['ofizmath']+df1['ogeo']+df1['ochem']+df1['oeng']).apply(lambda x: 1 if x >= 1 else 0)

df1.columns = df1.columns.astype(str)

# Отображение пользовательского ввода
st.write('Введенные данные:')
st.write(df1)

# Предсказание с помощью загруженной модели
if st.button('Получить предсказание'):

    prediction = model.predict(df1)
    st.write('Предсказанное значение:')
    st.write(prediction)
