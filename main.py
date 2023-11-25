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
df1 = pd.DataFrame([user_input])
# Добавление столбца 'omathinfo'
# Добавление столбца 'omathinfo' с условием
df1['mathinfo1'] = ((df1['math1'] + df1['info1']) / df1['sum1']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['mathinfo2'] = ((df1['math2'] + df1['info2']) / df1['sum2']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['omathinfo'] = (df1['mathinfo1']+ df1['mathinfo2'])
df1 = df1.drop(['mathinfo1','mathinfo2'], axis=1)

df1['russoc1'] = ((df1['rus1'] + df1['soc1']) / df1['sum1']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['russoc2'] = ((df1['rus2'] + df1['soc2']) / df1['sum2']).apply(lambda x: 1 if x >= 0.243 else 0)
df1['orussoc'] = (df1['russoc1']+ df1['russoc2'])
df1 = df1.drop(['russoc1','russoc2'], axis=1)

df1['fizmath1'] = ((df1['math1'] + df1['fiz1']) / df1['sum1']).apply(lambda x: 1 if x > 0.243 else 0)
df1['fizmath2'] = ((df1['math2'] + df1['fiz2']) / df1['sum2']).apply(lambda x: 1 if x > 0.243 else 0)
df1['ofizmath'] = (df1['fizmath1']+ df1['fizmath2'])
df1 = df1.drop(['fizmath1','fizmath2'], axis=1)

df1['biochem1'] = ((df1['bio1'] + df1['chem1']) / df1['sum1']).apply(lambda x: 1 if x > 0.243 else 0)
df1['biochem2'] = ((df1['bio2'] + df1['chem2']) / df1['sum2']).apply(lambda x: 1 if x > 0.243 else 0)
df1['obiochem'] = (df1['biochem1']+ df1['biochem2'])
df1 = df1.drop(['biochem1','biochem2'], axis=1)

df1['geo11'] = (df1['geo1'] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['geo21'] = (df1['geo2'] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['ogeo'] = (df1['geo11']+ df1['geo21'])
df1 = df1.drop(['geo11','geo21'], axis=1)

df1['chem11'] = (df1['chem1'] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['chem21'] = (df1['chem2'] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['ochem'] = (df1['chem11']+ df1['chem21'])
df1 = df1.drop(['chem11','chem21'], axis=1)

df1['eng11'] = (df1['eng1'] / df1['sum1']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['eng21'] = (df1['eng2'] / df1['sum2']).apply(lambda x: 1 if x >= 0.125 else 0)
df1['oeng'] = (df1['eng11']+ df1['eng21'])
df1 = df1.drop(['eng11','eng21'], axis=1)

df1 = df1.drop(['sum1','sum2','average1','average2'],axis=1)
df1['opotential'] = (df1['omathinfo']+ df1['orussoc']+df1['ofizmath']+df1['ogeo']+df1['ochem']+df1['oeng']).apply(lambda x: 1 if x >= 1 else 0)


# Отображение пользовательского ввода
st.write('Введенные данные:')
st.write(input_data)

# Предсказание с помощью загруженной модели
if st.button('Получить предсказание'):

    prediction = model.predict(input_data)
    st.write('Предсказанное значение:')
    st.write(prediction)
