"""
07 - Summary Dashboard
Comprehensive overview of NYC Airbnb analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
data_extended = os.path.join(project_dir, 'airbnb_open_data.csv')
output_path = os.path.join(project_dir, 'graphs', '07_summary_dashboard.png')

# Load data
df = pd.read_csv(data_path)
df_ext = pd.read_csv(data_extended, low_memory=False)

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
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0d1117')

fig.suptitle('NYC Airbnb Market Analysis', fontsize=28, fontweight='bold', color='white', y=0.97)
fig.text(0.5, 0.935, 'Multi-Source Analysis from 2 Kaggle Datasets', 
         fontsize=14, ha='center', color='#8b949e')

# Create grid
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, left=0.05, right=0.95, top=0.88, bottom=0.05)

# ===== ROW 1: Key Stats =====
stats = [
    ('Total Listings', f'{len(df):,}', '#58a6ff'),
    ('Avg Price', f'${df["price"].mean():.0f}', '#56d364'),
    ('Unique Hosts', f'{df["host_id"].nunique():,}', '#a371f7'),
    ('Neighborhoods', f'{df["neighbourhood"].nunique()}', '#f0883e'),
]

for i, (label, value, color) in enumerate(stats):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('#161b22')
    ax.axhline(y=0.95, xmin=0.1, xmax=0.9, color=color, linewidth=4)
    ax.text(0.5, 0.55, value, fontsize=32, fontweight='bold', ha='center', va='center', 
            transform=ax.transAxes, color=color)
    ax.text(0.5, 0.2, label, fontsize=12, ha='center', va='center', 
            transform=ax.transAxes, color='#8b949e')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_color('#30363d')

# ===== ROW 2: Charts =====
# Map
ax1 = fig.add_subplot(gs[1, 0:2])
ax1.set_facecolor('#0d1117')

sample = df.sample(15000, random_state=42)
for borough, color in borough_colors.items():
    subset = sample[sample['neighbourhood_group'] == borough]
    ax1.scatter(subset['longitude'], subset['latitude'], c=color, s=2, alpha=0.5, label=borough)

ax1.set_title('NYC Listing Map', color='white', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', facecolor='#161b22', labelcolor='white', fontsize=8, markerscale=3)
ax1.tick_params(colors='white', labelsize=8)
ax1.set_xlabel('Longitude', color='white', fontsize=9)
ax1.set_ylabel('Latitude', color='white', fontsize=9)
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# Borough prices
ax2 = fig.add_subplot(gs[1, 2:4])
ax2.set_facecolor('#0d1117')

boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
avg_prices = df.groupby('neighbourhood_group')['price'].mean().reindex(boroughs)
colors = [borough_colors[b] for b in boroughs]

bars = ax2.bar(boroughs, avg_prices, color=colors)
for bar, price in zip(bars, avg_prices):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'${price:.0f}',
             ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

ax2.set_title('Average Price by Borough', color='white', fontsize=14, fontweight='bold')
ax2.set_ylabel('Price ($)', color='white', fontsize=10)
ax2.tick_params(colors='white', labelsize=9)
ax2.set_xticklabels(boroughs, rotation=15)
for spine in ax2.spines.values():
    spine.set_color('#30363d')

# ===== ROW 3: More analysis =====
# Room type
ax3 = fig.add_subplot(gs[2, 0:2])
ax3.set_facecolor('#0d1117')

room_colors = {'Entire home/apt': '#ff6b6b', 'Private room': '#4ecdc4', 'Shared room': '#ffd93d'}
room_data = df.groupby('room_type').agg({'id': 'count', 'price': 'mean'}).reset_index()
room_data.columns = ['room_type', 'count', 'avg_price']
room_data = room_data.sort_values('count', ascending=False)

x = np.arange(len(room_data))
width = 0.4

bars1 = ax3.bar(x - width/2, room_data['count'], width, label='Listings', 
                color=[room_colors.get(r, '#666') for r in room_data['room_type']])
ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x + width/2, room_data['avg_price'], width, label='Avg Price',
                     color=[room_colors.get(r, '#666') for r in room_data['room_type']], alpha=0.5)

ax3.set_xticks(x)
ax3.set_xticklabels(room_data['room_type'], color='white', fontsize=9)
ax3.set_ylabel('Listing Count', color='white', fontsize=10)
ax3_twin.set_ylabel('Avg Price ($)', color='white', fontsize=10)
ax3.set_title('Room Type Distribution', color='white', fontsize=14, fontweight='bold')
ax3.tick_params(colors='white')
ax3_twin.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_color('#30363d')
for spine in ax3_twin.spines.values():
    spine.set_color('#30363d')

# Data sources panel
ax4 = fig.add_subplot(gs[2, 2:4])
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_color('#30363d')

ax4.text(0.5, 0.92, 'Data Sources', fontsize=14, fontweight='bold',
         ha='center', va='top', color='white', transform=ax4.transAxes)

sources = [
    ('Kaggle Dataset 1:', 'dgomonov/new-york-city-airbnb-open-data', '#4ecdc4'),
    ('', f'{len(df):,} listings | 2019 data', '#8b949e'),
    ('Kaggle Dataset 2:', 'arianazmoudeh/airbnbopendata', '#ff6b6b'),
    ('', f'{len(df_ext):,} listings | Extended features', '#8b949e'),
    ('Combined:', f'{len(df) + len(df_ext):,} total records', '#56d364'),
    ('Analysis:', 'Price, Geography, Hosts, Room Types', '#58a6ff'),
]

for i, (label, value, color) in enumerate(sources):
    y_pos = 0.78 - i * 0.12
    if label:
        ax4.text(0.08, y_pos, label, fontsize=11, color='white', fontweight='bold',
                 transform=ax4.transAxes, va='center')
    ax4.text(0.08 if not label else 0.35, y_pos, value, fontsize=10, color=color,
             transform=ax4.transAxes, va='center')

plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Summary dashboard generated")
