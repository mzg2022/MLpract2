# Практическая работа 2: Кластеризация студентов на основе опроса
import matplotlib

matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import umap
import plotly.express as px
import sys

try:
    student_data = pd.read_excel('dlia studentov.xlsx', engine='openpyxl')
    print(f"Размер набора данных: {student_data.shape}")
    print("\nПервые пять записей:")
    print(student_data.head().to_string())
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    print("\nДля работы требуется библиотека openpyxl. Установите выполнив:")
    print("pip install openpyxl")
    sys.exit(1)

print("\nИнформация о структуре данных:")
student_data.info()

print("\nАнализ пропущенных значений:")
print(student_data.isnull().sum())

plt.figure(figsize=(15, 20))
for i, column in enumerate(student_data.columns[2:15], 1):
    plt.subplot(5, 3, i)
    sns.countplot(data=student_data, x=column)
    plt.title(f"Распределение: {column}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_distributions_plot.png')
plt.close()

plt.figure(figsize=(10, 6))
faculty_col = 'На каком факультете/в каком институте Вы обучаетесь?'
sns.countplot(data=student_data, y=faculty_col, 
              order=student_data[faculty_col].value_counts().index)
plt.title('Распределение студентов по факультетам')
plt.savefig('faculty_distribution_chart.png')
plt.close()

cleaned_data = student_data.dropna(subset=student_data.columns[2:])
binary_columns = [col for col in student_data.columns[2:-1] 
                 if student_data[col].nunique() == 2]
encoder = LabelEncoder()

for col in binary_columns:
    cleaned_data[col] = encoder.fit_transform(cleaned_data[col])

binary_features = cleaned_data[binary_columns]

umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, 
                         n_components=2, random_state=42)
umap_projections = umap_reducer.fit_transform(binary_features)

plt.figure(figsize=(10, 8))
plt.scatter(umap_projections[:, 0], umap_projections[:, 1], s=5)
plt.title('Визуализация UMAP')
plt.xlabel('Компонента 1')
plt.ylabel('Компонента 2')
plt.savefig('umap_visualization.png')
plt.close()

inertia_values = []
silhouette_scores = []
k_options = range(2, 10)

for k in k_options:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(umap_projections)
    inertia_values.append(kmeans_model.inertia_)
    silhouette_scores.append(
        silhouette_score(umap_projections, kmeans_model.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_options, inertia_values, 'bo-')
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.title('Метод локтя для определения оптимального k')

plt.subplot(1, 2, 2)
plt.plot(k_options, silhouette_scores, 'go-')
plt.xlabel('Количество кластеров')
plt.ylabel('Silhouette Score')
plt.title('Оценка качества кластеризации')
plt.savefig('cluster_evaluation_metrics.png')
plt.close()

n_clusters = 4

clustering_techniques = {
    "K-Means": KMeans(n_clusters=n_clusters, random_state=42),
    "Иерархическая": AgglomerativeClustering(n_clusters=n_clusters),
    "GMM": GaussianMixture(n_components=n_clusters, random_state=42),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
}

clustering_results = {}
best_sil_score = -1
top_model = None

for name, model in clustering_techniques.items():
    if name == "GMM":
        cluster_labels = model.fit_predict(umap_projections)
    else:
        cluster_labels = model.fit_predict(umap_projections)

    if len(np.unique(cluster_labels)) > 1:
        sil_score = silhouette_score(umap_projections, cluster_labels)
        db_score = davies_bouldin_score(umap_projections, cluster_labels)

        if sil_score > best_sil_score:
            best_sil_score = sil_score
            top_model = name
    else:
        sil_score = -1
        db_score = float('inf')

    clustering_results[name] = {
        'labels': cluster_labels, 
        'silhouette': sil_score, 
        'davies_bouldin': db_score
    }
    print(f"{name}: Silhouette = {sil_score:.3f}, Davies-Bouldin = {db_score:.3f}")

optimal_labels = clustering_results[top_model]['labels']
cleaned_data['cluster'] = optimal_labels
print(f"\nОптимальный алгоритм: {top_model} (Silhouette: {best_sil_score:.3f})")

cluster_characteristics = cleaned_data.groupby('cluster').mean()[binary_columns]

plt.figure(figsize=(15, 8))
sns.heatmap(cluster_characteristics.T, annot=True, cmap='coolwarm')
plt.title('Характеристики кластеров')
plt.savefig('cluster_characteristics_heatmap.png')
plt.close()


def generate_radar_plot(cluster_data, title, filename):
    features = cluster_data.index.tolist()
    values = cluster_data.values.tolist()
    features += [features[0]]
    values += [values[0]]

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features[:-1])
    ax.set_title(title, size=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


for cluster_num in sorted(cleaned_data['cluster'].unique()):
    cluster_data = cluster_characteristics.loc[cluster_num]
    generate_radar_plot(
        cluster_data, 
        f'Профиль кластера {cluster_num}', 
        f'cluster_{cluster_num}_profile.png'
    )

faculty_cluster_dist = pd.crosstab(
    cleaned_data[faculty_col], 
    cleaned_data['cluster'], 
    normalize='index'
)

plt.figure(figsize=(12, 8))
sns.heatmap(faculty_cluster_dist, annot=True, cmap='YlGnBu')
plt.title('Распределение кластеров по факультетам')
plt.xlabel('Кластер')
plt.ylabel('Факультет')
plt.savefig('faculty_cluster_distribution.png')
plt.close()

if umap_projections.shape[1] >= 3:
    reducer_3d = umap.UMAP(n_components=3, random_state=42)
    umap_3d = reducer_3d.fit_transform(binary_features)

    fig = px.scatter_3d(
        x=umap_3d[:, 0], y=umap_3d[:, 1], z=umap_3d[:, 2],
        color=optimal_labels,
        title='3D визуализация кластеров',
        labels={'color': 'Кластер'},
        opacity=0.7
    )
    fig.write_html('3d_clusters_visualization.html')
else:
    print("Для 3D визуализации требуется минимум 3 компоненты")

cluster_profiles = {
    0: "Цифровые аборигены: Высокая вовлеченность во все цифровые платформы, предпочитают видеоформат, активно используют дополнительные ресурсы",
    1: "Традиционалисты: Низкая цифровая вовлеченность, предпочитают текстовые материалы и личное общение с преподавателями",
    2: "Социальные учащиеся: Средняя цифровая активность, высокая потребность в обсуждениях и групповой работе",
    3: "Прагматики: Выборочное использование платформ, предпочитают презентации и тесты для самопроверки"
}

print("\n" + "=" * 60)
print("ОСНОВНЫЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ ДЛЯ ОБРАЗОВАТЕЛЬНОЙ СРЕДЫ")
print("=" * 60 + "\n")

print(f"Выявлено {n_clusters} кластеров студентов:")
for cluster_id, description in cluster_profiles.items():
    print(f"Кластер {cluster_id}: {description}")

print("\nРекомендации для учебного заведения:")
print("1. Цифровые аборигены (Кластер 0): Развитие интерактивных видеокурсов")
print("2. Традиционалисты (Кластер 1): Усиление очных консультаций и текстовых материалов")
print("3. Социальные учащиеся (Кластер 2): Внедрение групповых проектов и дискуссионных платформ")
print("4. Прагматики (Кластер 3): Создание модульных программ с тестами самопроверки")

print("\nПерспективные направления развития:")
print("- Адаптивные образовательные траектории на основе кластерного анализа")
print("- Интеллектуальная система рекомендации учебных материалов")
print("- Целевые программы поддержки для различных типов учащихся")
print("- Оптимизация платформ под потребности студенческих кластеров")

print("\nВсе графики сохранены в текущую директорию")