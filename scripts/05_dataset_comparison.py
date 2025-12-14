"""
05 - Multi-Dataset Comparison (2019 vs Extended Data)
Compares the two Kaggle datasets
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_2019 = os.path.join(project_dir, 'airbnb_nyc.csv')
data_extended = os.path.join(project_dir, 'airbnb_open_data.csv')
output_path = os.path.join(project_dir, 'graphs', '05_dataset_comparison.png')

# Load datasets
df_2019 = pd.read_csv(data_2019)
df_2019 = df_2019[df_2019['price'] <= 1000]
df_2019 = df_2019[df_2019['price'] > 0]
df_2019['source'] = '2019 Dataset'

df_extended = pd.read_csv(data_extended, low_memory=False)
# Clean extended data - handle price column with $ sign
if 'price' in df_extended.columns:
    df_extended['price'] = pd.to_numeric(df_extended['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
df_extended = df_extended[df_extended['price'].notna()]
df_extended = df_extended[df_extended['price'] <= 1000]
df_extended = df_extended[df_extended['price'] > 0]
df_extended['source'] = 'Extended Dataset'

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Multi-Dataset Analysis: Comparing Two Kaggle Sources', 
             fontsize=20, fontweight='bold', color='white', y=0.98)

# ----- Plot 1: Dataset sizes -----
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')

sizes = [len(df_2019), len(df_extended)]
labels = ['NYC Airbnb 2019\n(dgomonov)', 'Airbnb Open Data\n(arianazmoudeh)']
colors = ['#4ecdc4', '#ff6b6b']

bars = ax1.bar(labels, sizes, color=colors, edgecolor='white', linewidth=0.5)
for bar, size in zip(bars, sizes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, f'{size:,}',
             ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')

ax1.set_ylabel('Number of Listings', color='white', fontsize=12)
ax1.set_title('Dataset Sizes', color='white', fontsize=14, fontweight='bold')
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Price distribution comparison -----
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')

ax2.hist(df_2019['price'], bins=50, alpha=0.7, color='#4ecdc4', label='2019 Dataset', density=True)
ax2.hist(df_extended['price'], bins=50, alpha=0.7, color='#ff6b6b', label='Extended Dataset', density=True)

ax2.set_xlabel('Price ($)', color='white', fontsize=12)
ax2.set_ylabel('Density', color='white', fontsize=12)
ax2.set_title('Price Distribution Comparison', color='white', fontsize=14, fontweight='bold')
ax2.legend(facecolor='#161b22', labelcolor='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

# ----- Plot 3: Average price by borough (2019) -----
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')

boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
avg_2019 = df_2019.groupby('neighbourhood_group')['price'].mean().reindex(boroughs)

colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
bars = ax3.bar(boroughs, avg_2019, color=colors)
for bar, val in zip(bars, avg_2019):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'${val:.0f}',
             ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

ax3.set_ylabel('Average Price ($)', color='white', fontsize=12)
ax3.set_title('2019 Dataset: Price by Borough', color='white', fontsize=14, fontweight='bold')
ax3.tick_params(colors='white', labelsize=9)
ax3.set_xticklabels(boroughs, rotation=15)
for spine in ax3.spines.values():
    spine.set_color('#30363d')

# ----- Plot 4: Stats summary -----
ax4 = axes[1, 1]
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_color('#30363d')

ax4.text(0.5, 0.95, 'Combined Dataset Statistics', fontsize=16, fontweight='bold',
         ha='center', va='top', color='white', transform=ax4.transAxes)

total_listings = len(df_2019) + len(df_extended)
combined_avg = (df_2019['price'].mean() + df_extended['price'].mean()) / 2

stats = [
    ('Data Sources:', '2 Kaggle Datasets', '#d4a72c'),
    ('2019 Dataset:', f"{len(df_2019):,} listings", '#4ecdc4'),
    ('Extended Dataset:', f"{len(df_extended):,} listings", '#ff6b6b'),
    ('Total Records:', f"{total_listings:,}", '#56d364'),
    ('2019 Avg Price:', f"${df_2019['price'].mean():.0f}/night", '#4ecdc4'),
    ('Extended Avg Price:', f"${df_extended['price'].mean():.0f}/night", '#ff6b6b'),
    ('Features (2019):', '16 columns', '#58a6ff'),
    ('Features (Extended):', '26 columns', '#58a6ff'),
]

for i, (label, value, color) in enumerate(stats):
    y_pos = 0.82 - i * 0.09
    ax4.text(0.1, y_pos, label, fontsize=12, color='#8b949e', transform=ax4.transAxes, va='center')
    ax4.text(0.55, y_pos, value, fontsize=12, color=color, fontweight='bold', transform=ax4.transAxes, va='center')

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Combined: {total_listings:,} listings from 2 sources")
