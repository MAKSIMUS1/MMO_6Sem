import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

array = np.random.randint(0, 10, size=(4, 5))

print("Исходный массив:")
print(array)

value_to_find = 6

indices = np.where(array == value_to_find)

count = indices[0].size

print("\nКоличество элементов, равных", value_to_find, ":", count)

split_arrays = np.split(array, 2, axis=0)

print("\nПервый массив:")
print(split_arrays[0])

print("\nВторой массив:")
print(split_arrays[1])

# 2. Pandas. Изучите структуры данных Series и Dataframe:

# создайте объект Series из массива NumPy
np_array = np.array([1, 2, 3, 4, 5])
series = pd.Series(np_array)

# произведите с ним различные математические операции
print("Series:\n", series)
print("Сумма элементов:", series.sum())
print("Максимальное значение:", series.max())
print("Минимальное значение:", series.min())

# создайте объект Dataframe из массива NumPy
np_array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(np_array_2d)
print("\nDataFrame:")
print(df)

# напишите строку заголовков в созданном Dataframe
headers = ['A', 'B', 'C']
df.columns = headers
print("\nDataFrame с заголовками:")
print(df)

# удалите любую строку
df = df.drop(0)
print("\nDataFrame после удаления строки:")
print(df)

# удалите любой столбец
df = df.drop('B', axis=1)
print("\nDataFrame после удаления столбца:")
print(df)

# выведите размер получившегося Dataframe
print("\nРазмер DataFrame:", df.shape)

# найдите все элементы равные какому-либо числу
equal_to_4 = df[df == 4]
print("\nЭлементы равные 4:")
print(equal_to_4)

# Часть 2. Статистическая обработка данных и библиотека Matplotlib
# Импортирование CSV-файла в DataFrame
df = pd.read_csv('stats/results.csv')

# Укажите столбец (параметр), по которому хотите построить гистограмму
column_to_plot = 'home_score'

# Построение гистограммы частот
plt.hist(df[column_to_plot], bins=10, color='skyblue', edgecolor='black')

# Добавление заголовка и меток осей
plt.title('Histogram of home_score')
plt.xlabel('home_score')
plt.ylabel('Frequency')

# Рассчет медианы
median_value = df[column_to_plot].median()

# Рассчет среднего значения
mean_value = df[column_to_plot].mean()

print(f"Медиана {column_to_plot}: {median_value}")
print(f"Среднее значение {column_to_plot}: {mean_value}")

# Построение box plot
plt.figure(figsize=(8, 6))
plt.boxplot(df[column_to_plot], vert=False)
plt.title('Box plot для параметра home_score')
plt.xlabel('home_score')
plt.show()

# Отображение гистограммы
plt.show()

# Применение метода .describe() к выбранному параметру
description = df[column_to_plot].describe()

print(description)

grouped_data = df.groupby('tournament').agg({
    'home_score': ['mean', 'sum'],
    'away_score': ['mean', 'sum']
})

print(grouped_data)
