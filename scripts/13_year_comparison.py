"""
13 - Year-over-Year Comparison (2019 vs 2025)
Analyzes changes in NYC Airbnb market over time
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_2019 = os.path.join(project_dir, 'airbnb_nyc.csv')
data_2025 = os.path.join(project_dir, 'airbnb_nyc_2025.csv')
output_path = os.path.join(project_dir, 'graphs', '13_year_comparison.png')

print("Loading multi-year data...")
df_2019 = pd.read_csv(data_2019)
df_2025 = pd.read_csv(data_2025)

# Clean data
df_2019 = df_2019[df_2019['price'] > 0]
df_2019 = df_2019[df_2019['price'] <= 10000]
df_2025 = df_2025[df_2025['price'] > 0]
df_2025 = df_2025[df_2025['price'] <= 10000]

print(f"2019: {len(df_2019):,} listings")
print(f"2025: {len(df_2025):,} listings")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('NYC Airbnb: 2019 vs 2025 (6 Year Evolution)', fontsize=22, fontweight='bold', color='white', y=0.98)

colors = {'2019': '#ff6b6b', '2025': '#4ecdc4'}

# ----- Plot 1: Market Size Change -----
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')

metrics = ['Listings', 'Avg Price']
values_2019 = [len(df_2019), df_2019['price'].mean()]
values_2025 = [len(df_2025), df_2025['price'].mean()]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, values_2019, width, label='2019', color=colors['2019'])
bars2 = ax1.bar(x + width/2, values_2025, width, label='2025', color=colors['2025'])

ax1.set_xticks(x)
ax1.set_xticklabels(['Listings Count', 'Avg Price ($)'], color='white', fontsize=11)
ax1.set_title('Market Size Comparison', color='white', fontsize=14, fontweight='bold')
ax1.legend(facecolor='#161b22', labelcolor='white')
ax1.tick_params(colors='white')
for spine in ax1.spines.values(): spine.set_color('#30363d')

# Add labels
for bar, val in zip(bars1, values_2019):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,.0f}',
             ha='center', color='white', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, values_2025):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f'{val:,.0f}',
             ha='center', color='white', fontsize=10, fontweight='bold')

# ----- Plot 2: Borough Distribution Change -----
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')

borough_2019 = df_2019['neighbourhood_group'].value_counts()
borough_2025 = df_2025['neighbourhood_group'].value_counts()

boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
vals_2019 = [borough_2019.get(b, 0) for b in boroughs]
vals_2025 = [borough_2025.get(b, 0) for b in boroughs]

x = np.arange(len(boroughs))
width = 0.35

bars1 = ax2.bar(x - width/2, vals_2019, width, label='2019', color=colors['2019'])
bars2 = ax2.bar(x + width/2, vals_2025, width, label='2025', color=colors['2025'])

ax2.set_xticks(x)
ax2.set_xticklabels(boroughs, color='white', fontsize=9, rotation=15)
ax2.set_ylabel('Listings', color='white')
ax2.set_title('Borough Distribution Over Time', color='white', fontsize=14, fontweight='bold')
ax2.legend(facecolor='#161b22', labelcolor='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values(): spine.set_color('#30363d')

# ----- Plot 3: Price Distribution Change -----
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')

ax3.hist(df_2019['price'], bins=50, alpha=0.6, color=colors['2019'], label='2019', density=True)
ax3.hist(df_2025['price'], bins=50, alpha=0.6, color=colors['2025'], label='2025', density=True)

ax3.axvline(df_2019['price'].median(), color=colors['2019'], linestyle='--', linewidth=2)
ax3.axvline(df_2025['price'].median(), color=colors['2025'], linestyle='--', linewidth=2)

ax3.set_xlim(0, 500)
ax3.set_xlabel('Price ($)', color='white')
ax3.set_ylabel('Density', color='white')
ax3.set_title('Price Distribution Shift', color='white', fontsize=14, fontweight='bold')
ax3.legend(facecolor='#161b22', labelcolor='white')
ax3.tick_params(colors='white')
for spine in ax3.spines.values(): spine.set_color('#30363d')

# ----- Plot 4: Key Changes Summary -----
ax4 = axes[1, 1]
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values(): spine.set_color('#30363d')

ax4.text(0.5, 0.95, '6-Year Market Evolution', fontsize=16, fontweight='bold',
         ha='center', va='top', color='white', transform=ax4.transAxes)

listing_change = ((len(df_2025) - len(df_2019)) / len(df_2019)) * 100
price_change = ((df_2025['price'].mean() - df_2019['price'].mean()) / df_2019['price'].mean()) * 100

summary = [
    ('2019 Listings:', f'{len(df_2019):,}', colors['2019']),
    ('2025 Listings:', f'{len(df_2025):,}', colors['2025']),
    ('Listing Change:', f'{listing_change:+.0f}%', '#ff6b6b' if listing_change < 0 else '#56d364'),
    ('2019 Avg Price:', f'${df_2019["price"].mean():.0f}', colors['2019']),
    ('2025 Avg Price:', f'${df_2025["price"].mean():.0f}', colors['2025']),
    ('Price Change:', f'{price_change:+.0f}%', '#ff6b6b' if price_change < 0 else '#56d364'),
    ('Data Source:', 'Inside Airbnb', '#ffd700'),
]

for i, (label, value, color) in enumerate(summary):
    y_pos = 0.80 - i * 0.10
    ax4.text(0.10, y_pos, label, fontsize=12, color='#8b949e', transform=ax4.transAxes, va='center')
    ax4.text(0.55, y_pos, value, fontsize=12, color=color, fontweight='bold', transform=ax4.transAxes, va='center')

ax4.text(0.5, 0.06, 'Oct 2025 data from insideairbnb.com',
         fontsize=9, ha='center', color='#8b949e', transform=ax4.transAxes)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"Saved: {output_path}")
print(f"\nKey Changes:")
print(f"  Listings: {len(df_2019):,} → {len(df_2025):,} ({listing_change:+.0f}%)")
print(f"  Avg Price: ${df_2019['price'].mean():.0f} → ${df_2025['price'].mean():.0f} ({price_change:+.0f}%)")
