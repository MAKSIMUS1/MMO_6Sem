import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import graphviz

# 1. Загрузка данных
data = pd.read_csv("data/Credit_card.csv")

# 2. Анализ пропущенных значений
missing_values = data.isnull().sum()

# Заполнение пропущенных значений
imputer = SimpleImputer(strategy="most_frequent")
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Работа с категориальными признаками
label_encoder = LabelEncoder()
categorical_columns = data_filled.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    data_filled[col] = label_encoder.fit_transform(data_filled[col])

# Удаление ненужной информации
data_processed = data_filled.drop(columns=["Ind_ID", "EMAIL_ID"])

# 3. Выделите из данных вектор меток У и матрицу признаков Х
# Выделение вектора меток y (целевой переменной)
y = data_processed["Car_Owner"]

# Выделение матрицы признаков X
X = data_processed.drop(columns=["Car_Owner"])
print(X.head())
print(y.head())

# 4. Разделите набор данных на обучающую и тестовую выборки.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

print("Inputs X (train/test):", X_train.shape, X_test.shape)

print("Outputs Y (train/test):", y_train.shape, y_test.shape)

# 5. На обучающей выборке получите модели дерева решений и k-ближайших
# соседей, рассчитайте точность моделей.

tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(X_train, y_train)
print("Правильность на обучающем наборе (Decision Tree): {:.3f}".format(tree.score(X_train, y_train)))
print("Правильность на тестовом наборе (Decision Tree): {:.3f}".format(tree.score(X_test, y_test)))

# Обучение модели k-ближайших соседей
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Оценка точности модели k-ближайших соседей
print("Правильность на обучающем наборе (KNN): {:.3f}".format(knn.score(X_train, y_train)))
print("Правильность на тестовом наборе (KNN): {:.3f}".format(knn.score(X_test, y_test)))

# 6. Подберите наилучшие параметры моделей (например, глубину для дерева
# решений, количество соседей для алгоритма knn)

# Определение сетки параметров для поиска
param_grid_tree = {'max_depth': [3, 5, 7, 10]}

# Создание экземпляра модели
tree = DecisionTreeClassifier(random_state=0)

# Объект поиска наилучших параметров
grid_search_tree = GridSearchCV(tree, param_grid_tree, cv=5)

# Поиск наилучших параметров на обучающем наборе
grid_search_tree.fit(X_train, y_train)

# Вывод наилучших параметров
print("Наилучшие параметры для дерева решений:", grid_search_tree.best_params_)

# Оценка точности наилучшей модели на тестовом наборе
best_tree = grid_search_tree.best_estimator_
print("Правильность на тестовом наборе для дерева решений:", best_tree.score(X_test, y_test))

# Определение сетки параметров для поиска
param_grid_knn = {'n_neighbors': [3, 5, 7, 10]}

# Создание экземпляра модели
knn = KNeighborsClassifier()

# Объект поиска наилучших параметров
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=5)

# Поиск наилучших параметров на обучающем наборе
grid_search_knn.fit(X_train, y_train)

# Вывод наилучших параметров
print("Наилучшие параметры для KNN:", grid_search_knn.best_params_)

# Оценка точности наилучшей модели на тестовом наборе
best_knn = grid_search_knn.best_estimator_
print("Правильность на тестовом наборе для KNN:", best_knn.score(X_test, y_test))

# 7. Рассчитайте матрицу ошибок (confusion matrix) для каждой модели.

# Рассчитываем матрицу ошибок для дерева решений
tree_confusion_mat = confusion_matrix(y_test, best_tree.predict(X_test))
print("Confusion matrix for Decision Tree:")
print(tree_confusion_mat)

# Рассчитываем матрицу ошибок для KNN
knn_confusion_mat = confusion_matrix(y_test, best_knn.predict(X_test))
print("Confusion matrix for KNN:")
print(knn_confusion_mat)

# 9*. Визуализируйте полученную модель дерева решений (при визуализации
# желательно уменьшить глубину дерева, что бы рисунок был читаемым, или
# сохранить в отдельный файл)
# Экспорт структуры дерева решений в файл
export_graphviz(best_tree, out_file="tree.dot", feature_names=X.columns, class_names=['Not Car Owner', 'Car Owner'], filled=True, rounded=True)

# Визуализация дерева решений
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
#dot -Tpng tree.dot -o tree.png

