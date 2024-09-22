import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("data/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("data/house-prices-advanced-regression-techniques/test.csv")

# 2. Визуализация матрицы корреляций в виде тепловой карты
numeric_columns = train_data.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train_data[numeric_columns].corr()
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Overall Qual: Rates the overall material and finish of the house
# Gr Liv Area: Above grade (ground) living area square feet
# Garage Cars: Size of garage in car capacity
# Total Bsmt SF: Total square feet of basement area
# Sale Price - the property's sale price in dollars. This is the target variable that you're trying to predict.

# Построение матрицы диаграмм рассеяния
sns.pairplot(train_data, x_vars=["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"], y_vars=["SalePrice"])
plt.suptitle("Scatterplot Matrix", y=1.02)
plt.show()

# Проанализируйте коэффициенты корреляции и диаграммы рассеяния и
# выберите параметры для построения модели простой линейной регрессии
X = train_data[["OverallQual"]]
y = train_data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Оценка качества модели простой линейной регрессии
r_squared_single = model.score(X_test, y_test)
print("R-squared (Single Feature):", r_squared_single)

# Визуализация графика модели на диаграмме рассеяния
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, model.predict(X_test), color='red', linewidth=2)
plt.title("Linear Regression Model (Single Feature)")
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")
plt.show()

# Рассчитайте модель с несколькими параметрами и оцените качество модели.
X_multi = train_data[["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"]]

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

r_squared_multi = model_multi.score(X_test_multi, y_test_multi)
print("R-squared (Multiple Features):", r_squared_multi)
