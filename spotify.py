import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('spotify_tracks.csv')

feature_columns = ['popularity', 'duration_ms']
data = data[feature_columns]

data.fillna(data.mean(), inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)  

plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(reduced_features)

data['Cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=clusters, palette="viridis", legend="full")
plt.title("Clusters of Songs Based on Features")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(scaled_features)

def get_recommendations(song_index, data, model, n_recommendations=5):
    distances, indices = model.kneighbors([scaled_features[song_index]])
    recommendations = []
    for i in range(1, n_recommendations + 1): 
        recommended_song = data.iloc[indices[0][i]]
        recommendations.append(recommended_song)
    return recommendations

song_index = 42 
recommendations = get_recommendations(song_index, data, knn)
print("Recommendations:")
for recommendation in recommendations:
    print(recommendation)
