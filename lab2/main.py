import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Чтение данных из CSV файла
germany_cars = pd.read_csv("data/autoscout24-germany-dataset.csv")

print(germany_cars.dtypes)

# 1. Выявление пропусков данных
# Визуальный метод
missing_values = germany_cars.isnull().sum()
plt.figure(figsize=(10, 6))
missing_values.plot(kind='bar', color='skyblue')
plt.title('Количество пропусков в каждом столбце')
plt.xlabel('Столбец')
plt.ylabel('Количество пропусков')
plt.xticks(rotation=45)
plt.show()

# Расчетный метод
missing_values = germany_cars.isnull().sum()
print("Количество пропусков данных в каждом столбце:")
print(missing_values)

# 2. Исключение строк и столбцов с наибольшим количеством пропусков
# Исключение столбцов с пропусками более 30%
threshold = len(germany_cars) * 0.3
germany_cars.dropna(thresh=threshold, axis=1, inplace=True)  # Добавлено inplace=True

# 3. Замена оставшихся пропусков на логически обоснованные значения
germany_cars['gear'] = germany_cars['gear'].replace({'Manual': 0, 'Automatic': 1})
germany_cars.drop('model', axis=1, inplace=True)
mean_hp = germany_cars['hp'].mean()
germany_cars['hp'].fillna(mean_hp, inplace=True)
most_common_gear = germany_cars['gear'].mode()[0]
germany_cars['gear'].fillna(most_common_gear, inplace=True)

# 4. Гистограмма распределения датасета после обработки пропусков
missing_values_after = germany_cars.isnull().sum()  # Количество пропусков после обработки
plt.figure(figsize=(10, 6))
missing_values_after.plot(kind='bar', color='lightgreen')
plt.title('Распределение пропусков данных после обработки')
plt.xlabel('Столбец')
plt.ylabel('Количество пропусков')
plt.xticks(rotation=45)
plt.show()

missing_values = germany_cars.isnull().sum()
print("Количество пропусков данных в каждом столбце:")
print(missing_values)

# 5. Проверьте датасет на наличие выбросов, удалите найденные аномальные записи.
columns_to_exclude = ['make', 'fuel', 'offerType']  # Исключаем столбцы с категориальными данными
numeric_columns = germany_cars.select_dtypes(include=np.number).columns.tolist()
numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude]

plt.figure(figsize=(10, 6))
sns.boxplot(data=germany_cars[numeric_columns])
plt.title('Boxplot для выявления выбросов')
plt.xticks(rotation=45)
plt.show()

q1 = germany_cars[numeric_columns].quantile(0.25)
q3 = germany_cars[numeric_columns].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

germany_cars_no_outliers = germany_cars[~((germany_cars[numeric_columns] < lower_bound) | (germany_cars[numeric_columns] > upper_bound)).any(axis=1)]

print("Количество удаленных аномальных записей:", len(germany_cars) - len(germany_cars_no_outliers))

# 6. Применяем One-Hot Encoding к столбцам с категориальными данными
germany_cars_encoded = pd.get_dummies(germany_cars, columns=['make', 'fuel', 'offerType'])

# 7. Сохраняем обработанный датасет
germany_cars_encoded.to_csv("data/autoscout24-germany-dataset-encoded.csv", index=False)



print(germany_cars)
