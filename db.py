import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# -----------------------------
# Step 0: Load dataset
# -----------------------------
file_path = "your_dataset.csv"  # replace with your dataset
df = pd.read_csv(file_path)

# -----------------------------
# Step 1: Preprocessing
# -----------------------------
df = df.dropna()  # remove missing values

# One-hot encode categorical features
categorical_features = df.select_dtypes(include=['object']).columns
X = pd.get_dummies(df, columns=categorical_features)
X = X.values

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 2: Choose clustering algorithm
# -----------------------------
# Options: "kmeans", "hierarchical", "gmm", "dbscan"
clustering_algo = "kmeans"  

if clustering_algo == "kmeans":
    model = KMeans(n_clusters=3, random_state=42)  # change n_clusters as needed
    labels = model.fit_predict(X_scaled)
elif clustering_algo == "hierarchical":
    model = AgglomerativeClustering(n_clusters=3)  # change n_clusters as needed
    labels = model.fit_predict(X_scaled)
elif clustering_algo == "gmm":
    model = GaussianMixture(n_components=3, random_state=42)
    labels = model.fit_predict(X_scaled)
elif clustering_algo == "dbscan":
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X_scaled)
else:
    raise ValueError("Choose kmeans, hierarchical, gmm, or dbscan")

# -----------------------------
# Step 3: Visualize clusters (PCA 2D)
# -----------------------------
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='rainbow', edgecolors='k')
plt.title(f"{clustering_algo.upper()} Clustering (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -----------------------------
# Step 4: Optional info
# -----------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Clusters found: {n_clusters}")
print(f"Unique labels: {set(labels)}")
