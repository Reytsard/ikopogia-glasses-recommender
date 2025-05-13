import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
import json

# Load dataset
df = pd.read_csv("./res/frame_preferences_dataset.csv")

# Encode categorical features
encoded_df = df.copy()
label_encoders = {}
categorical_cols = ['Material', 'Color', 'RimType']

for col in categorical_cols:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    label_encoders[col] = le

# Select features for clustering
features = ['FrameWidth(mm)', 'FrameHeight(mm)', 'Material', 'Color', 'RimType']
X = encoded_df[features]

# Cluster the data
kmeans = KMeans(n_clusters=5, random_state=42)
encoded_df['Cluster'] = kmeans.fit_predict(X)

# Map clusters to the most common FaceShape
cluster_to_shape = (
    encoded_df.groupby('Cluster')['FaceShape']
    .agg(lambda x: Counter(x).most_common(1)[0][0])
    .to_dict()
)

# Add the mapped face shape as "ClusterFaceShape"
encoded_df['ClusterFaceShape'] = encoded_df['Cluster'].map(cluster_to_shape)

# Join back original columns for export
output_df = df.copy()
output_df['Cluster'] = encoded_df['Cluster']
output_df['ClusterFaceShape'] = encoded_df['ClusterFaceShape']

# Export to JSON
output_json = output_df.to_dict(orient='records')
with open("./res/clustered_frames.json", "w") as f:
    json.dump(output_json, f, indent=2)

print("Clustered data saved to clustered_frames.json")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

# Add PCA results to DataFrame for plotting
encoded_df['PCA1'] = pca_result[:, 0]
encoded_df['PCA2'] = pca_result[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    encoded_df['PCA1'],
    encoded_df['PCA2'],
    c=encoded_df['Cluster'],
    cmap='viridis',
    alpha=0.7,
    edgecolors='k'
)

plt.title("Eyeglass Frame Clustering (Visualized with PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True)
plt.tight_layout()
plt.show()
