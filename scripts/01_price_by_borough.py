"""
01 - Price Distribution by Borough
Visualizes price patterns across NYC boroughs
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '01_price_by_borough.png')

# Load data
df = pd.read_csv(data_path)

# Remove extreme outliers (price > $1000)
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
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')

# ----- Plot 1: Box plot by borough -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
data_by_borough = [df[df['neighbourhood_group'] == b]['price'] for b in boroughs]

bp = ax1.boxplot(data_by_borough, labels=boroughs, patch_artist=True)

for patch, borough in zip(bp['boxes'], boroughs):
    patch.set_facecolor(borough_colors[borough])
    patch.set_alpha(0.7)
for whisker in bp['whiskers']:
    whisker.set_color('white')
for cap in bp['caps']:
    cap.set_color('white')
for median in bp['medians']:
    median.set_color('white')
    median.set_linewidth(2)
for flier in bp['fliers']:
    flier.set(marker='o', markerfacecolor='#666', markersize=2, alpha=0.3)

ax1.set_ylabel('Price per Night ($)', color='white', fontsize=12)
ax1.set_title('Price Distribution by Borough', color='white', fontsize=16, fontweight='bold')
ax1.tick_params(colors='white', labelsize=10)
ax1.set_xticklabels(boroughs, rotation=15)
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Average price bar chart -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

avg_prices = df.groupby('neighbourhood_group')['price'].mean().reindex(boroughs)
colors = [borough_colors[b] for b in boroughs]

bars = ax2.bar(boroughs, avg_prices, color=colors, edgecolor='white', linewidth=0.5)

for bar, price in zip(bars, avg_prices):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f'${price:.0f}',
             ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')

ax2.set_ylabel('Average Price ($)', color='white', fontsize=12)
ax2.set_title('Average Nightly Rate by Borough', color='white', fontsize=16, fontweight='bold')
ax2.tick_params(colors='white', labelsize=10)
ax2.set_xticklabels(boroughs, rotation=15)
for spine in ax2.spines.values():
    spine.set_color('#30363d')

# Add stats
total_listings = len(df)
avg_all = df['price'].mean()
fig.text(0.5, 0.02, f'Total Listings: {total_listings:,} | Overall Avg: ${avg_all:.0f}/night', 
         ha='center', color='#8b949e', fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Most expensive: Manhattan (${avg_prices['Manhattan']:.0f}/night)")
