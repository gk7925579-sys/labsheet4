import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------------------------------------
# Setup & Basic Exploratory Data Analysis
# ----------------------------------------------------

# Program 1: Load a dataset suitable for clustering using the Pandas library
df = pd.DataFrame({
    'Annual_Income': [15, 16, 17, 18, 19, 60, 62, 63, 65, 67, 100, 105, 110, 112, 115],
    'Spending_Score': [39, 81, 6, 77, 40, 42, 50, 52, 48, 55, 18, 83, 12, 90, 10],
    'Age': [19, 21, 20, 23, 31, 22, 35, 40, 64, 30, 49, 36, 52, 28, 45]
})

# Program 2: Display the first and last five records of the dataset
first_five = df.head(5)
last_five = df.tail(5)

# Program 3: Explore dataset information, summary statistics, and data types
df_info = df.info()
df_stats = df.describe()
df_dtypes = df.dtypes

# Program 4: Select relevant numerical features for clustering
X_unscaled = df[['Annual_Income', 'Spending_Score']].values

# Program 5: Standardize the dataset before clustering using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X_unscaled)


# ----------------------------------------------------
# K-Means Clustering
# ----------------------------------------------------

# Program 6: Implement the K-Means clustering algorithm
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Program 7: Apply K-Means clustering with K=2 clusters
kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_k2 = kmeans_k2.fit_predict(X)

# Program 8: Apply K-Means clustering with K=3 clusters
kmeans_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_k3 = kmeans_k3.fit_predict(X)

# Program 9: Apply K-Means clustering with different values of K and compare the results
inertias_list = []
for k in range(1, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias_list.append(km.inertia_)

# Program 10: Determine the optimal number of clusters using the Elbow Method
plt.figure()
plt.plot(range(1, 6), inertias_list, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.close()

# Program 11: Visualize clusters using a scatter plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels_k3, cmap='viridis')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.close()

# Program 12: Display the cluster centroids obtained from the K-Means algorithm
centroids_k3 = kmeans_k3.cluster_centers_

# Program 13: Assign cluster labels to the original dataset
df_labeled = df.copy()
df_labeled['KMeans_Cluster'] = labels_k3

# Program 14: Compare clustering results before and after feature scaling
labels_unscaled = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_unscaled)
scaling_comparison = pd.DataFrame({'Unscaled_Labels': labels_unscaled, 'Scaled_Labels': labels_k3})


# ----------------------------------------------------
# Hierarchical Clustering
# ----------------------------------------------------

# Program 15: Implement Agglomerative Hierarchical Clustering
agg_model = AgglomerativeClustering(n_clusters=3)

# Program 16: Generate a dendrogram using the SciPy library
plt.figure()
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.close()

# Program 17: Apply Hierarchical Clustering using Ward linkage
agg_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_ward = agg_ward.fit_predict(X)

# Program 18: Apply Hierarchical Clustering using Complete linkage
agg_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
labels_complete = agg_complete.fit_predict(X)

# Program 19: Compare different linkage methods
linkage_comparison = pd.DataFrame({'Ward_Labels': labels_ward, 'Complete_Labels': labels_complete})

# Program 20: Visualize Hierarchical Clustering results
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels_ward, cmap='rainbow')
plt.close()

# Program 21: Compare Hierarchical Clustering with K-Means clustering
kmeans_vs_hierarchical = pd.DataFrame({'KMeans': labels_k3, 'Hierarchical': labels_ward})


# ----------------------------------------------------
# Dimensionality Reduction (PCA)
# ----------------------------------------------------

# Program 22: Apply Principal Component Analysis (PCA) to reduce dataset dimensions
pca = PCA(n_components=2)

# Program 23: Reduce a dataset from multiple features to two principal components
X_all = scaler.fit_transform(df[['Annual_Income', 'Spending_Score', 'Age']])
X_pca = pca.fit_transform(X_all)

# Program 24: Visualize the transformed data using a two-dimensional scatter plot
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.close()

# Program 25: Display the explained variance ratio of principal components
var_ratio = pca.explained_variance_ratio_

# Program 26: Determine the cumulative explained variance
cum_var_ratio = np.cumsum(var_ratio)

# Program 27: Compare the dataset before and after dimensionality reduction
shape_before = X_all.shape
shape_after = X_pca.shape

# Program 28: Perform K-Means clustering on the PCA-transformed dataset
labels_pca = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_pca)

# Program 29: Compare clustering performance before and after applying PCA
score_before_pca = silhouette_score(X_all, labels_k3)
score_after_pca = silhouette_score(X_pca, labels_pca)


# ----------------------------------------------------
# Cluster Evaluation & Analysis
# ----------------------------------------------------

# Program 30: Calculate the Silhouette Score for cluster evaluation
sil_score_k3 = silhouette_score(X, labels_k3)

# Program 31: Compare Silhouette Scores for different values of K
sil_scores = {}
for k in range(2, 6):
    lbls = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    sil_scores[k] = silhouette_score(X, lbls)

# Program 32: Analyze the characteristics of each cluster using descriptive statistics
cluster_summary = df_labeled.groupby('KMeans_Cluster').mean()

# Program 33: Visualize cluster distributions using pair plots or scatter plots
plt.figure()
sns.pairplot(df_labeled, hue='KMeans_Cluster')
plt.close()

# Program 34: Save the clustered dataset as a CSV file
df_labeled.to_csv('clustered_dataset.csv', index=False)

# Program 35: Compare the performance of K-Means and Hierarchical Clustering and summarize the findings
comparison_summary = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical (Ward)'],
    'Silhouette_Score': [silhouette_score(X, labels_k3), silhouette_score(X, labels_ward)]
})
