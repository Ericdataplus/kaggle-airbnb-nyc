"""
11 - Geographic Hotspot Analysis
Identifies pricing hotspots using density-based clustering
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '11_geographic_hotspots.png')

print("Loading data...")
df = pd.read_csv(data_path)

# Clean data
df = df[df['price'] > 0]
df = df[df['price'] <= 1000]

# Focus on high-price listings for hotspot analysis
high_price = df[df['price'] > df['price'].quantile(0.75)].copy()

print(f"Analyzing {len(high_price):,} high-price listings...")

# Geographic features
coords = high_price[['latitude', 'longitude']].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# DBSCAN clustering for density-based hotspots
print("Running DBSCAN clustering...")
dbscan = DBSCAN(eps=0.15, min_samples=50)
high_price['hotspot'] = dbscan.fit_predict(coords_scaled)

# Analyze hotspots
n_hotspots = len(set(high_price['hotspot'])) - (1 if -1 in high_price['hotspot'].values else 0)
print(f"Found {n_hotspots} price hotspots")

# Get hotspot statistics
hotspot_stats = high_price[high_price['hotspot'] != -1].groupby('hotspot').agg({
    'price': ['mean', 'count'],
    'latitude': 'mean',
    'longitude': 'mean',
    'neighbourhood': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Mixed'
}).reset_index()
hotspot_stats.columns = ['hotspot', 'avg_price', 'count', 'lat', 'lon', 'neighborhood']
hotspot_stats = hotspot_stats.sort_values('avg_price', ascending=False)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Geographic Hotspot Analysis (DBSCAN)', fontsize=22, fontweight='bold', color='white', y=0.98)

# ----- Plot 1: Full NYC with hotspots -----
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')

# Background - all listings
sample = df.sample(min(15000, len(df)), random_state=42)
ax1.scatter(sample['longitude'], sample['latitude'], c='#333', s=1, alpha=0.3)

# Hotspots colored by price
hotspot_listings = high_price[high_price['hotspot'] != -1]
scatter = ax1.scatter(hotspot_listings['longitude'], hotspot_listings['latitude'],
                      c=hotspot_listings['price'], cmap='hot', s=10, alpha=0.7)

ax1.set_xlabel('Longitude', color='white', fontsize=11)
ax1.set_ylabel('Latitude', color='white', fontsize=11)
ax1.set_title('High-Price Hotspots in NYC', color='white', fontsize=14, fontweight='bold')
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
cbar.set_label('Price ($)', color='white')
cbar.ax.tick_params(colors='white')

# ----- Plot 2: Hotspot Rankings -----
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')

top_hotspots = hotspot_stats.head(10)
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_hotspots)))

bars = ax2.barh(range(len(top_hotspots)), top_hotspots['avg_price'], color=colors)

labels = [f"Hotspot {int(h)} ({n})" for h, n in zip(top_hotspots['hotspot'], top_hotspots['neighborhood'].str[:15])]
ax2.set_yticks(range(len(top_hotspots)))
ax2.set_yticklabels(labels, color='white', fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Average Price ($)', color='white', fontsize=11)
ax2.set_title('Top 10 Price Hotspots', color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

for bar, price in zip(bars, top_hotspots['avg_price']):
    ax2.text(price + 5, bar.get_y() + bar.get_height()/2, f'${price:.0f}',
             va='center', color='white', fontsize=10, fontweight='bold')

# ----- Plot 3: Price Heatmap (Hexbin) -----
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')

hb = ax3.hexbin(df['longitude'], df['latitude'], C=df['price'], 
                gridsize=40, cmap='YlOrRd', reduce_C_function=np.mean, mincnt=5)

ax3.set_xlabel('Longitude', color='white', fontsize=11)
ax3.set_ylabel('Latitude', color='white', fontsize=11)
ax3.set_title('Price Density Heatmap', color='white', fontsize=14, fontweight='bold')
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_color('#30363d')

cbar = plt.colorbar(hb, ax=ax3, shrink=0.8)
cbar.set_label('Avg Price ($)', color='white')
cbar.ax.tick_params(colors='white')

# ----- Plot 4: Summary Stats -----
ax4 = axes[1, 1]
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_color('#30363d')

ax4.text(0.5, 0.95, 'Hotspot Analysis Results', fontsize=16, fontweight='bold',
         ha='center', va='top', color='white', transform=ax4.transAxes)

summary = [
    ('Algorithm:', 'DBSCAN', '#d4a72c'),
    ('Hotspots Found:', f'{n_hotspots}', '#ff6b6b'),
    ('Listings in Hotspots:', f'{len(hotspot_listings):,}', '#4ecdc4'),
    ('Noise Points:', f'{(high_price["hotspot"] == -1).sum():,}', '#8b949e'),
    ('Top Hotspot Avg:', f'${hotspot_stats.iloc[0]["avg_price"]:.0f}/night' if len(hotspot_stats) > 0 else 'N/A', '#ff6b6b'),
    ('Top Neighborhood:', hotspot_stats.iloc[0]['neighborhood'][:20] if len(hotspot_stats) > 0 else 'N/A', '#56d364'),
    ('Analysis Focus:', 'Top 25% by price', '#58a6ff'),
]

for i, (label, value, color) in enumerate(summary):
    y_pos = 0.82 - i * 0.10
    ax4.text(0.1, y_pos, label, fontsize=12, color='#8b949e', transform=ax4.transAxes, va='center')
    ax4.text(0.50, y_pos, str(value), fontsize=12, color=color, fontweight='bold', transform=ax4.transAxes, va='center')

ax4.text(0.5, 0.12, 'DBSCAN identifies dense clusters of premium listings',
         fontsize=10, ha='center', color='#8b949e', transform=ax4.transAxes)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"Saved: {output_path}")
