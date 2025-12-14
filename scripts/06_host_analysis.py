"""
06 - Host Analysis
Analyzes host patterns and multi-listing hosts
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '06_host_analysis.png')

# Load data
df = pd.read_csv(data_path)
df = df[df['price'] <= 1000]
df = df[df['price'] > 0]

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0d1117')

# ----- Plot 1: Host listing distribution -----
ax1 = axes[0]
ax1.set_facecolor('#0d1117')

# Categorize hosts
host_listings = df.groupby('host_id').size()
host_categories = pd.cut(host_listings, bins=[0, 1, 2, 5, 10, 100, 1000], 
                         labels=['1 listing', '2 listings', '3-5 listings', 
                                '6-10 listings', '11-100 listings', '100+ listings'])
category_counts = host_categories.value_counts().sort_index()

colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(category_counts)))

wedges, texts, autotexts = ax1.pie(category_counts.values, labels=category_counts.index,
                                    autopct='%1.1f%%', colors=colors,
                                    textprops={'color': 'white', 'fontsize': 9},
                                    startangle=90)

ax1.set_title('Host Distribution by Listing Count', color='white', fontsize=14, fontweight='bold')

# ----- Plot 2: Top hosts by listings -----
ax2 = axes[1]
ax2.set_facecolor('#0d1117')

top_hosts = df.groupby(['host_id', 'host_name']).agg({
    'id': 'count',
    'price': 'mean'
}).reset_index()
top_hosts.columns = ['host_id', 'host_name', 'listings', 'avg_price']
top_hosts = top_hosts.nlargest(10, 'listings')

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_hosts)))
y_pos = range(len(top_hosts))

bars = ax2.barh(y_pos, top_hosts['listings'], color=colors)

for i, (bar, row) in enumerate(zip(bars, top_hosts.itertuples())):
    ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
             f'{row.listings} (${row.avg_price:.0f}/night)',
             va='center', ha='left', color='white', fontsize=9)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_hosts['host_name'].str[:20] + '...', color='white', fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('Number of Listings', color='white', fontsize=11)
ax2.set_title('Top 10 Hosts by Listing Count', color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

# Stats
unique_hosts = df['host_id'].nunique()
avg_per_host = len(df) / unique_hosts
fig.text(0.5, 0.02, f'Unique Hosts: {unique_hosts:,} | Avg Listings per Host: {avg_per_host:.1f}', 
         ha='center', color='#8b949e', fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"✅ Saved: {output_path}")
print(f"   Unique hosts: {unique_hosts:,}")
