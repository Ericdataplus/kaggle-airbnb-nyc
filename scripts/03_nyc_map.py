"""
03 - NYC Map Visualization
Creates a scatter plot map of Airbnb listings
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '03_nyc_map.png')

# Load data
df = pd.read_csv(data_path)
df = df[df['price'] <= 1000]
df = df[df['price'] > 0]

# Borough colors
borough_colors = {
    'Manhattan': '#ff6b6b',
    'Brooklyn': '#4ecdc4',
    'Queens': '#45b7d1',
    'Bronx': '#96ceb4',
    'Staten Island': '#ffeaa7'
}

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('NYC Airbnb Listings Map', fontsize=22, fontweight='bold', color='white', y=0.98)

# ----- Plot 1: Map by Borough -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

for borough, color in borough_colors.items():
    subset = df[df['neighbourhood_group'] == borough]
    ax1.scatter(subset['longitude'], subset['latitude'], 
                c=color, s=3, alpha=0.5, label=f'{borough} ({len(subset):,})')

ax1.set_xlabel('Longitude', color='white', fontsize=11)
ax1.set_ylabel('Latitude', color='white', fontsize=11)
ax1.set_title('Listings by Borough', color='white', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', facecolor='#161b22', labelcolor='white', fontsize=9)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Map by Price (heatmap-like) -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

# Sample for performance
sample = df.sample(min(20000, len(df)), random_state=42)

# Create scatter with price coloring
scatter = ax2.scatter(sample['longitude'], sample['latitude'],
                      c=sample['price'], cmap='hot', s=4, alpha=0.6,
                      vmin=0, vmax=500)

ax2.set_xlabel('Longitude', color='white', fontsize=11)
ax2.set_ylabel('Latitude', color='white', fontsize=11)
ax2.set_title('Listings by Price', color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
cbar.set_label('Price ($)', color='white')
cbar.ax.tick_params(colors='white')

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Total listings mapped: {len(df):,}")
