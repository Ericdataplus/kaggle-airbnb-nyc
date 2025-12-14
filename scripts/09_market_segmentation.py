"""
09 - Market Segmentation (K-Means Clustering)
Segments the Airbnb market using unsupervised learning
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '09_market_segmentation.png')

print("Loading data...")
df = pd.read_csv(data_path)

# Clean data
df = df[df['price'] > 0]
df = df[df['price'] <= 1000]
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Encode room type
le = LabelEncoder()
df['room_encoded'] = le.fit_transform(df['room_type'])

# Features for clustering
cluster_features = ['price', 'minimum_nights', 'number_of_reviews', 
                    'reviews_per_month', 'availability_365', 'room_encoded']

X = df[cluster_features].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Finding optimal clusters...")
# Elbow method
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Use 5 clusters (good balance)
n_clusters = 5
print(f"Training K-Means with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# Analyze clusters
cluster_profiles = df.groupby('cluster').agg({
    'price': 'mean',
    'minimum_nights': 'mean',
    'number_of_reviews': 'mean',
    'availability_365': 'mean',
    'room_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
    'neighbourhood_group': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
    'id': 'count'
}).rename(columns={'id': 'count'})

# Name the segments
segment_names = {
    0: 'Budget Stays',
    1: 'Popular Picks', 
    2: 'Luxury Escapes',
    3: 'Long-Term Rentals',
    4: 'Hidden Gems'
}

# Assign names based on characteristics
sorted_by_price = cluster_profiles.sort_values('price')
cluster_map = {}
cluster_map[sorted_by_price.index[0]] = 'Budget Stays'
cluster_map[sorted_by_price.index[-1]] = 'Luxury Escapes'

remaining = [i for i in range(n_clusters) if i not in cluster_map]
sorted_remaining = cluster_profiles.loc[remaining].sort_values('number_of_reviews', ascending=False)
if len(sorted_remaining) > 0:
    cluster_map[sorted_remaining.index[0]] = 'Popular Picks'
if len(sorted_remaining) > 1:
    cluster_map[sorted_remaining.index[1]] = 'Long-Term Rentals'
if len(sorted_remaining) > 2:
    cluster_map[sorted_remaining.index[2]] = 'Hidden Gems'

df['segment'] = df['cluster'].map(cluster_map)
cluster_profiles['segment'] = cluster_profiles.index.map(cluster_map)

print("\nCluster Profiles:")
print(cluster_profiles[['segment', 'price', 'count']])

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Market Segmentation (K-Means Clustering)', fontsize=22, fontweight='bold', color='white', y=0.98)

cluster_colors = ['#ff6b6b', '#4ecdc4', '#ffd93d', '#96ceb4', '#45b7d1']

# ----- Plot 1: PCA Scatter -----
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')

for i, (cluster_id, segment) in enumerate(cluster_map.items()):
    mask = df['cluster'] == cluster_id
    ax1.scatter(df.loc[mask, 'pca1'], df.loc[mask, 'pca2'], 
                c=cluster_colors[i], s=5, alpha=0.5, label=segment)

ax1.set_xlabel('Principal Component 1', color='white', fontsize=11)
ax1.set_ylabel('Principal Component 2', color='white', fontsize=11)
ax1.set_title('Market Segments (PCA Visualization)', color='white', fontsize=14, fontweight='bold')
ax1.legend(facecolor='#161b22', labelcolor='white', markerscale=3)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Segment Sizes -----
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')

segment_sizes = df['segment'].value_counts()
colors = [cluster_colors[list(cluster_map.keys())[list(cluster_map.values()).index(s)]] 
          for s in segment_sizes.index]

wedges, texts, autotexts = ax2.pie(segment_sizes.values, labels=segment_sizes.index,
                                    autopct='%1.1f%%', colors=colors,
                                    textprops={'color': 'white', 'fontsize': 10},
                                    startangle=90)

ax2.set_title('Segment Distribution', color='white', fontsize=14, fontweight='bold')

# ----- Plot 3: Segment Price Comparison -----
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')

segment_order = cluster_profiles.sort_values('price')['segment'].tolist()
prices = [cluster_profiles[cluster_profiles['segment'] == s]['price'].values[0] for s in segment_order]
colors_ordered = [cluster_colors[list(cluster_map.keys())[list(cluster_map.values()).index(s)]] 
                  for s in segment_order]

bars = ax3.barh(range(len(segment_order)), prices, color=colors_ordered)

ax3.set_yticks(range(len(segment_order)))
ax3.set_yticklabels(segment_order, color='white', fontsize=11)
ax3.set_xlabel('Average Price ($)', color='white', fontsize=11)
ax3.set_title('Average Price by Segment', color='white', fontsize=14, fontweight='bold')
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_color('#30363d')

for bar, price in zip(bars, prices):
    ax3.text(price + 5, bar.get_y() + bar.get_height()/2, f'${price:.0f}',
             va='center', color='white', fontsize=11, fontweight='bold')

# ----- Plot 4: Elbow + Summary -----
ax4 = axes[1, 1]
ax4.set_facecolor('#0d1117')

ax4.plot(list(K_range), inertias, 'o-', color='#4ecdc4', linewidth=2, markersize=8)
ax4.axvline(x=n_clusters, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Chosen: {n_clusters}')

ax4.set_xlabel('Number of Clusters (K)', color='white', fontsize=11)
ax4.set_ylabel('Inertia', color='white', fontsize=11)
ax4.set_title('Elbow Method for Optimal K', color='white', fontsize=14, fontweight='bold')
ax4.legend(facecolor='#161b22', labelcolor='white')
ax4.tick_params(colors='white')
for spine in ax4.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"Saved: {output_path}")
