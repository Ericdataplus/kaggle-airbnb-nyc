"""
02 - Room Type Analysis
Analyzes the distribution and pricing of different room types
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '02_room_type_analysis.png')

# Load data
df = pd.read_csv(data_path)
df = df[df['price'] <= 1000]
df = df[df['price'] > 0]

# Room type colors
room_colors = {
    'Entire home/apt': '#ff6b6b',
    'Private room': '#4ecdc4',
    'Shared room': '#ffd93d'
}

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')

# ----- Plot 1: Room type distribution (Pie) -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

room_counts = df['room_type'].value_counts()
colors = [room_colors.get(r, '#666') for r in room_counts.index]

wedges, texts, autotexts = ax1.pie(room_counts.values, labels=room_counts.index,
                                    autopct='%1.1f%%', colors=colors,
                                    textprops={'color': 'white', 'fontsize': 11},
                                    explode=(0.02, 0.02, 0.02), startangle=90)

for autotext in autotexts:
    autotext.set_fontweight('bold')

ax1.set_title('Listing Distribution by Room Type', color='white', fontsize=16, fontweight='bold')

# ----- Plot 2: Average price by room type -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

avg_by_room = df.groupby('room_type')['price'].agg(['mean', 'median', 'count'])
avg_by_room = avg_by_room.sort_values('mean', ascending=False)

room_names = avg_by_room.index.tolist()
means = avg_by_room['mean'].values
medians = avg_by_room['median'].values
colors = [room_colors.get(r, '#666') for r in room_names]

x = np.arange(len(room_names))
width = 0.35

bars1 = ax2.bar(x - width/2, means, width, label='Average', color=colors, alpha=0.9)
bars2 = ax2.bar(x + width/2, medians, width, label='Median', color=colors, alpha=0.5)

for bar, val in zip(bars1, means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'${val:.0f}',
             ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, medians):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'${val:.0f}',
             ha='center', va='bottom', color='white', fontsize=9)

ax2.set_ylabel('Price per Night ($)', color='white', fontsize=12)
ax2.set_title('Price by Room Type', color='white', fontsize=16, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(room_names, color='white', fontsize=10)
ax2.legend(facecolor='#161b22', labelcolor='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Most listings: {room_counts.index[0]} ({room_counts.values[0]:,})")
