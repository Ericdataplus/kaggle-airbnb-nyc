"""
04 - Top Neighborhoods Analysis
Analyzes the most expensive and most listed neighborhoods
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '04_top_neighborhoods.png')

# Load data
df = pd.read_csv(data_path)
df = df[df['price'] <= 1000]
df = df[df['price'] > 0]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0d1117')

# ----- Plot 1: Most listings by neighborhood -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

top_neighborhoods = df['neighbourhood'].value_counts().head(15)
colors = plt.cm.viridis(np.linspace(0.9, 0.3, len(top_neighborhoods)))

bars = ax1.barh(range(len(top_neighborhoods)), top_neighborhoods.values, color=colors)

for bar, val in zip(bars, top_neighborhoods.values):
    ax1.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val:,}',
             va='center', ha='left', color='white', fontsize=9)

ax1.set_yticks(range(len(top_neighborhoods)))
ax1.set_yticklabels(top_neighborhoods.index, color='white', fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel('Number of Listings', color='white', fontsize=11)
ax1.set_title('Top 15 Neighborhoods by Listings', color='white', fontsize=14, fontweight='bold')
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Most expensive neighborhoods -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

# Filter neighborhoods with at least 50 listings
neighborhood_stats = df.groupby('neighbourhood').agg({
    'price': ['mean', 'median', 'count']
}).reset_index()
neighborhood_stats.columns = ['neighbourhood', 'mean_price', 'median_price', 'count']
neighborhood_stats = neighborhood_stats[neighborhood_stats['count'] >= 50]
top_expensive = neighborhood_stats.nlargest(15, 'mean_price')

colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_expensive)))

bars = ax2.barh(range(len(top_expensive)), top_expensive['mean_price'].values, color=colors)

for bar, val in zip(bars, top_expensive['mean_price'].values):
    ax2.text(val + 5, bar.get_y() + bar.get_height()/2, f'${val:.0f}',
             va='center', ha='left', color='white', fontsize=9)

ax2.set_yticks(range(len(top_expensive)))
ax2.set_yticklabels(top_expensive['neighbourhood'].values, color='white', fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel('Average Price per Night ($)', color='white', fontsize=11)
ax2.set_title('Top 15 Most Expensive Neighborhoods', color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Most expensive: {top_expensive.iloc[0]['neighbourhood']} (${top_expensive.iloc[0]['mean_price']:.0f}/night)")
