"""
12 - Availability & Demand Forecasting
Time-series style analysis of availability patterns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '12_availability_analysis.png')

print("Loading data...")
df = pd.read_csv(data_path)

# Clean data
df = df[df['price'] > 0]
df = df[df['price'] <= 1000]
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Calculate demand metrics
df['booking_rate'] = 1 - (df['availability_365'] / 365)
df['demand_score'] = df['number_of_reviews'] * df['reviews_per_month']

# Borough analysis
borough_stats = df.groupby('neighbourhood_group').agg({
    'availability_365': 'mean',
    'booking_rate': 'mean',
    'price': 'mean',
    'number_of_reviews': 'mean',
    'demand_score': 'mean',
    'id': 'count'
}).rename(columns={'id': 'count'}).reset_index()

# Room type analysis
room_stats = df.groupby('room_type').agg({
    'availability_365': 'mean',
    'booking_rate': 'mean',
    'price': 'mean',
    'id': 'count'
}).rename(columns={'id': 'count'}).reset_index()

# Price vs Availability correlation
price_bins = pd.cut(df['price'], bins=[0, 50, 100, 150, 200, 300, 500, 1000])
availability_by_price = df.groupby(price_bins, observed=True)['availability_365'].mean()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Availability & Demand Analysis', fontsize=22, fontweight='bold', color='white', y=0.98)

borough_colors = {
    'Manhattan': '#ff6b6b',
    'Brooklyn': '#4ecdc4',
    'Queens': '#45b7d1',
    'Bronx': '#96ceb4',
    'Staten Island': '#ffeaa7'
}

# ----- Plot 1: Booking Rate by Borough -----
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')

boroughs = borough_stats['neighbourhood_group'].tolist()
booking_rates = borough_stats['booking_rate'].values * 100
colors = [borough_colors.get(b, '#666') for b in boroughs]

bars = ax1.bar(boroughs, booking_rates, color=colors)

for bar, rate in zip(bars, booking_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{rate:.0f}%',
             ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

ax1.set_ylabel('Implied Booking Rate (%)', color='white', fontsize=11)
ax1.set_title('Booking Rate by Borough', color='white', fontsize=14, fontweight='bold')
ax1.tick_params(colors='white', labelsize=9)
ax1.set_xticklabels(boroughs, rotation=15)
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Price vs Availability Scatter -----
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')

sample = df.sample(min(5000, len(df)), random_state=42)
scatter = ax2.scatter(sample['price'], sample['availability_365'], 
                      c=sample['number_of_reviews'], cmap='viridis', 
                      s=10, alpha=0.5)

# Add trend line
X = sample['price'].values.reshape(-1, 1)
y = sample['availability_365'].values
reg = LinearRegression()
reg.fit(X, y)
ax2.plot([0, 500], [reg.predict([[0]])[0], reg.predict([[500]])[0]], 
         'r--', linewidth=2, label='Trend')

ax2.set_xlabel('Price ($)', color='white', fontsize=11)
ax2.set_ylabel('Availability (days/year)', color='white', fontsize=11)
ax2.set_title('Price vs Availability', color='white', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 500)
ax2.legend(facecolor='#161b22', labelcolor='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
cbar.set_label('Reviews', color='white')
cbar.ax.tick_params(colors='white')

# ----- Plot 3: Room Type Availability -----
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')

room_colors = {'Entire home/apt': '#ff6b6b', 'Private room': '#4ecdc4', 'Shared room': '#ffd93d'}
room_types = room_stats['room_type'].tolist()
availabilities = room_stats['availability_365'].values
colors = [room_colors.get(r, '#666') for r in room_types]

bars = ax3.bar(room_types, availabilities, color=colors)

for bar, avail in zip(bars, availabilities):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{avail:.0f} days',
             ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

ax3.set_ylabel('Avg Availability (days/year)', color='white', fontsize=11)
ax3.set_title('Availability by Room Type', color='white', fontsize=14, fontweight='bold')
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_color('#30363d')

# ----- Plot 4: Demand Summary -----
ax4 = axes[1, 1]
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_color('#30363d')

ax4.text(0.5, 0.95, 'Demand Insights', fontsize=16, fontweight='bold',
         ha='center', va='top', color='white', transform=ax4.transAxes)

# Find highest demand borough
highest_demand = borough_stats.loc[borough_stats['booking_rate'].idxmax()]
lowest_avail_room = room_stats.loc[room_stats['availability_365'].idxmin()]

correlation = df['price'].corr(df['availability_365'])

summary = [
    ('Avg Availability:', f'{df["availability_365"].mean():.0f} days/year', '#58a6ff'),
    ('Implied Booking Rate:', f'{df["booking_rate"].mean()*100:.0f}%', '#4ecdc4'),
    ('Highest Demand:', f'{highest_demand["neighbourhood_group"]} ({highest_demand["booking_rate"]*100:.0f}%)', '#ff6b6b'),
    ('Most Booked Type:', f'{lowest_avail_room["room_type"]}', '#ffd93d'),
    ('Price-Avail Corr:', f'{correlation:.2f}', '#a371f7'),
    ('Avg Reviews/Month:', f'{df["reviews_per_month"].mean():.2f}', '#56d364'),
    ('High-Demand Listings:', f'{(df["booking_rate"] > 0.8).sum():,}', '#d4a72c'),
]

for i, (label, value, color) in enumerate(summary):
    y_pos = 0.82 - i * 0.10
    ax4.text(0.1, y_pos, label, fontsize=12, color='#8b949e', transform=ax4.transAxes, va='center')
    ax4.text(0.55, y_pos, value, fontsize=12, color=color, fontweight='bold', transform=ax4.transAxes, va='center')

ax4.text(0.5, 0.08, 'Lower availability = Higher demand',
         fontsize=10, ha='center', color='#8b949e', transform=ax4.transAxes)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"Saved: {output_path}")
