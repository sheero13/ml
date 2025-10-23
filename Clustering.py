#wine/iris.py

# from sklearn.datasets import load_iris, load_wine
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.mixture import GaussianMixture
# import matplotlib.pyplot as plt

# # ---------- Choose Dataset ----------
# dataset_name = "wine"  # "iris" or "wine"

# if dataset_name == "iris":
#     data = load_iris()
# elif dataset_name == "wine":
#     data = load_wine()
# else:
#     raise ValueError("Dataset must be 'iris' or 'wine'")

# X = data.data
# y = data.target

# # Standardize
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # ---------- Choose Algorithm ----------
# algorithm = "hierarchical"  # "kmeans", "hierarchical", "gmm"

# if algorithm == "kmeans":
#     model = KMeans(n_clusters=len(set(y)), random_state=42)
#     clusters = model.fit_predict(X_scaled)
# elif algorithm == "hierarchical":
#     model = AgglomerativeClustering(n_clusters=len(set(y)), linkage='ward')
#     clusters = model.fit_predict(X_scaled)
# elif algorithm == "gmm":
#     model = GaussianMixture(n_components=len(set(y)), random_state=42)
#     clusters = model.fit_predict(X_scaled)
# else:
#     raise ValueError("Algorithm must be 'kmeans', 'hierarchical', or 'gmm'")

# # ---------- Visualization ----------
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', edgecolor='k')
# plt.title(f"{algorithm.upper()} Clustering on {dataset_name.upper()} Dataset")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

#--------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.mixture import GaussianMixture
# import matplotlib.pyplot as plt

# # ---------- 1. Load Dataset ----------
# file_path = "your_dataset.csv"  # <-- replace with your CSV file
# df = pd.read_csv(file_path)

# # ---------- 2. Preliminary Cleaning ----------
# # Drop columns with all nulls
# df = df.dropna(axis=1, how='all')

# # Fill remaining nulls
# for col in df.columns:
#     if df[col].dtype == object:
#         df[col] = df[col].fillna("Missing")
#     else:
#         df[col] = df[col].fillna(df[col].median())

# # ---------- 3. One-Hot Encoding ----------
# categorical_cols = df.select_dtypes(include=['object']).columns
# df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# # Convert to numpy array for clustering
# X = df_encoded.values

# # ---------- 4. Standardize ----------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # ---------- 5. Dimensionality Reduction for Visualization ----------
# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X_scaled)

# # ---------- 6. Choose Algorithm ----------
# algorithm = "kmeans"  # Options: "kmeans", "hierarchical", "gmm"
# n_clusters = 3         # Set number of clusters

# if algorithm == "kmeans":
#     model = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = model.fit_predict(X_scaled)
# elif algorithm == "hierarchical":
#     model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
#     clusters = model.fit_predict(X_scaled)
# elif algorithm == "gmm":
#     model = GaussianMixture(n_components=n_clusters, random_state=42)
#     clusters = model.fit_predict(X_scaled)
# else:
#     raise ValueError("Algorithm must be 'kmeans', 'hierarchical', or 'gmm'")

# # ---------- 7. Visualization ----------
# plt.figure(figsize=(8,6))
# plt.scatter(X_2d[:,0], X_2d[:,1], c=clusters, cmap='viridis', edgecolor='k', s=50)
# plt.title(f"{algorithm.upper()} Clustering")
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.show()
