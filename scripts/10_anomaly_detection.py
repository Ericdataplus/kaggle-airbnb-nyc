"""
10 - Anomaly Detection (Overpriced/Underpriced)
Uses Isolation Forest to detect pricing anomalies
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '10_anomaly_detection.png')

print("Loading data...")
df = pd.read_csv(data_path)

# Clean data
df = df[df['price'] > 0]
df = df[df['price'] <= 1000]
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Encode categoricals
le_borough = LabelEncoder()
le_room = LabelEncoder()
df['borough_encoded'] = le_borough.fit_transform(df['neighbourhood_group'])
df['room_encoded'] = le_room.fit_transform(df['room_type'])

# Features for anomaly detection
features = ['price', 'minimum_nights', 'number_of_reviews', 
            'availability_365', 'borough_encoded', 'room_encoded']

X = df[features].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training Isolation Forest...")
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
df['anomaly'] = iso_forest.fit_predict(X_scaled)
df['anomaly_score'] = iso_forest.decision_function(X_scaled)

# Calculate expected price based on borough and room type FIRST
avg_by_group = df.groupby(['neighbourhood_group', 'room_type'])['price'].mean()
df['expected_price'] = df.apply(lambda x: avg_by_group.get((x['neighbourhood_group'], x['room_type']), x['price']), axis=1)
df['price_diff'] = df['price'] - df['expected_price']
df['price_diff_pct'] = (df['price_diff'] / df['expected_price']) * 100

# Now classify anomalies
anomalies = df[df['anomaly'] == -1].copy()
normal = df[df['anomaly'] == 1]

# Overpriced vs Underpriced
anomalies['type'] = anomalies['price_diff'].apply(lambda x: 'Overpriced' if x > 0 else 'Underpriced')


print(f"\nAnomalies found: {len(anomalies):,} ({len(anomalies)/len(df)*100:.1f}%)")
print(f"  Overpriced: {(anomalies['type'] == 'Overpriced').sum():,}")
print(f"  Underpriced: {(anomalies['type'] == 'Underpriced').sum():,}")

# Top anomalies
top_overpriced = df.nlargest(10, 'price_diff_pct')[['name', 'neighbourhood_group', 'room_type', 'price', 'expected_price', 'price_diff_pct']]
top_underpriced = df.nsmallest(10, 'price_diff_pct')[['name', 'neighbourhood_group', 'room_type', 'price', 'expected_price', 'price_diff_pct']]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Anomaly Detection (Isolation Forest)', fontsize=22, fontweight='bold', color='white', y=0.98)

# ----- Plot 1: Geographic Anomaly Map -----
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')

# Plot normal points
sample_normal = normal.sample(min(10000, len(normal)), random_state=42)
ax1.scatter(sample_normal['longitude'], sample_normal['latitude'], 
            c='#4ecdc4', s=3, alpha=0.3, label='Normal')

# Plot anomalies
ax1.scatter(anomalies['longitude'], anomalies['latitude'],
            c='#ff6b6b', s=15, alpha=0.7, label='Anomaly', marker='x')

ax1.set_xlabel('Longitude', color='white', fontsize=11)
ax1.set_ylabel('Latitude', color='white', fontsize=11)
ax1.set_title('Anomaly Locations in NYC', color='white', fontsize=14, fontweight='bold')
ax1.legend(facecolor='#161b22', labelcolor='white', markerscale=2)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Anomaly Score Distribution -----
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')

ax2.hist(normal['anomaly_score'], bins=50, alpha=0.7, color='#4ecdc4', label='Normal', density=True)
ax2.hist(anomalies['anomaly_score'], bins=30, alpha=0.7, color='#ff6b6b', label='Anomaly', density=True)

ax2.axvline(x=0, color='white', linestyle='--', linewidth=1)

ax2.set_xlabel('Anomaly Score', color='white', fontsize=11)
ax2.set_ylabel('Density', color='white', fontsize=11)
ax2.set_title('Anomaly Score Distribution', color='white', fontsize=14, fontweight='bold')
ax2.legend(facecolor='#161b22', labelcolor='white')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

# ----- Plot 3: Price vs Expected -----
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')

sample_df = df.sample(min(5000, len(df)), random_state=42)

scatter = ax3.scatter(sample_df['expected_price'], sample_df['price'],
                      c=sample_df['anomaly_score'], cmap='RdYlGn', s=10, alpha=0.5)

ax3.plot([0, 300], [0, 300], 'w--', linewidth=1, alpha=0.5, label='Fair Price')

ax3.set_xlabel('Expected Price ($)', color='white', fontsize=11)
ax3.set_ylabel('Actual Price ($)', color='white', fontsize=11)
ax3.set_title('Actual vs Expected Price', color='white', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 300)
ax3.set_ylim(0, 500)
ax3.legend(facecolor='#161b22', labelcolor='white')
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_color('#30363d')

cbar = plt.colorbar(scatter, ax=ax3, shrink=0.8)
cbar.set_label('Anomaly Score', color='white')
cbar.ax.tick_params(colors='white')

# ----- Plot 4: Anomaly Summary -----
ax4 = axes[1, 1]
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_color('#30363d')

ax4.text(0.5, 0.95, 'Anomaly Detection Results', fontsize=16, fontweight='bold',
         ha='center', va='top', color='white', transform=ax4.transAxes)

n_overpriced = (anomalies['type'] == 'Overpriced').sum()
n_underpriced = (anomalies['type'] == 'Underpriced').sum()
avg_overpriced = anomalies[anomalies['type'] == 'Overpriced']['price_diff_pct'].mean()
avg_underpriced = anomalies[anomalies['type'] == 'Underpriced']['price_diff_pct'].mean()

summary = [
    ('Algorithm:', 'Isolation Forest', '#d4a72c'),
    ('Contamination:', '5%', '#58a6ff'),
    ('Total Anomalies:', f'{len(anomalies):,}', '#ff6b6b'),
    ('Overpriced Listings:', f'{n_overpriced:,}', '#ff6b6b'),
    ('Avg Overpricing:', f'+{avg_overpriced:.0f}%' if not np.isnan(avg_overpriced) else 'N/A', '#ff6b6b'),
    ('Underpriced Listings:', f'{n_underpriced:,}', '#4ecdc4'),
    ('Avg Underpricing:', f'{avg_underpriced:.0f}%' if not np.isnan(avg_underpriced) else 'N/A', '#4ecdc4'),
    ('Normal Listings:', f'{len(normal):,}', '#56d364'),
]

for i, (label, value, color) in enumerate(summary):
    y_pos = 0.82 - i * 0.09
    ax4.text(0.1, y_pos, label, fontsize=12, color='#8b949e', transform=ax4.transAxes, va='center')
    ax4.text(0.55, y_pos, value, fontsize=12, color=color, fontweight='bold', transform=ax4.transAxes, va='center')

ax4.text(0.5, 0.08, 'Red = Overpriced | Green = Underpriced (deals!)',
         fontsize=10, ha='center', color='#8b949e', transform=ax4.transAxes)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"Saved: {output_path}")
