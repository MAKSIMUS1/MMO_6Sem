import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

data = pd.read_csv("data/Country-data.csv")

X = data[['child_mort', 'gdpp']]
print(X.isnull().sum())

# Нормализация данных
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Метод локтя для определения оптимального количества кластеров
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Кластеризация методом K-means
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# Визуализация результатов кластеризации K-means
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters_kmeans, palette='viridis')
plt.title('K-means Clustering')
plt.xlabel('Child Mortality')
plt.ylabel('GDP per capita')
plt.legend(title='Cluster')
plt.show()

# Иерархическая кластеризация
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(40, 24))
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# Выбор оптимального количества кластеров по дендрограмме и иерархическая кластеризация
hierarchical_clusters = AgglomerativeClustering(n_clusters=optimal_clusters)
clusters_hierarchical = hierarchical_clusters.fit_predict(X_scaled)

# Визуализация результатов иерархической кластеризации
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters_hierarchical, palette='viridis')
plt.title('Hierarchical Clustering')
plt.xlabel('Child Mortality')
plt.ylabel('GDP per capita')
plt.legend(title='Cluster')
plt.show()

# Оценка качества кластеризации
silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans)
silhouette_hierarchical = silhouette_score(X_scaled, clusters_hierarchical)
print("Silhouette Score for K-means:", silhouette_kmeans)
print("Silhouette Score for Hierarchical Clustering:", silhouette_hierarchical)

# Визуализация конкретного объекта
country_index = 0
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters_kmeans, palette='viridis')
plt.scatter(X_scaled[country_index, 0], X_scaled[country_index, 1], color='purple', s=100, label=data['country'][country_index])
plt.title('K-means Clustering with Selected Country Highlighted')
plt.xlabel('Child Mortality')
plt.ylabel('GDP per capita')
plt.legend()
plt.show()
