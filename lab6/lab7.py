import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели случайного леса на исходных данных
rf = RandomForestClassifier(random_state=42)
start_time = time.time()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred)
time_original = time.time() - start_time
print(f"Accuracy on original data: {accuracy_original:.4f}")
print(f"Time on original data: {time_original:.4f} seconds")

# Сокращение количества параметров методом отбора признаков с низкой дисперсией
selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_reduced = selector.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки для сокращенного набора данных
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

# Обучение модели случайного леса на сокращенном датасете
rf_red = RandomForestClassifier(random_state=42)
start_time = time.time()
rf_red.fit(X_train_red, y_train_red)
y_pred_red = rf_red.predict(X_test_red)
accuracy_reduced = accuracy_score(y_test_red, y_pred_red)
time_reduced = time.time() - start_time
print(f"Accuracy on reduced data: {accuracy_reduced:.4f}")
print(f"Time on reduced data: {time_reduced:.4f} seconds")

# Применение метода PCA к исходному датасету и нахождение 2 главных компонент
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Визуализация данных по двум главным компонентам
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis')
plt.title('PCA with 2 components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Обучение модели случайного леса на данных с двумя главными компонентами
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)
rf_pca = RandomForestClassifier(random_state=42)
start_time = time.time()
rf_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = rf_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
time_pca = time.time() - start_time
print(f"Accuracy with PCA (2 components): {accuracy_pca:.4f}")
print(f"Time with PCA (2 components): {time_pca:.4f} seconds")

# Объясненная дисперсия в зависимости от количества главных компонент
pca_full = PCA().fit(X)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained variance vs. number of components')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.show()

# Нахождение количества компонент для объяснения 90% дисперсии
n_components_90 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.9) + 1
print(f"Number of components for 90% variance: {n_components_90}")

# Обучение модели случайного леса на данных с количеством компонент для 90% дисперсии
pca_90 = PCA(n_components=n_components_90)
X_pca_90 = pca_90.fit_transform(X)
X_train_pca_90, X_test_pca_90, y_train_pca_90, y_test_pca_90 = train_test_split(X_pca_90, y, test_size=0.3, random_state=42)
rf_pca_90 = RandomForestClassifier(random_state=42)
start_time = time.time()
rf_pca_90.fit(X_train_pca_90, y_train_pca_90)
y_pred_pca_90 = rf_pca_90.predict(X_test_pca_90)
accuracy_pca_90 = accuracy_score(y_test_pca_90, y_pred_pca_90)
time_pca_90 = time.time() - start_time
print(f"Accuracy with PCA (90% variance): {accuracy_pca_90:.4f}")
print(f"Time with PCA (90% variance): {time_pca_90:.4f} seconds")
