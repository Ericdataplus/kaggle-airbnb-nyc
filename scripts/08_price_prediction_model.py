"""
08 - Price Prediction Model (XGBoost)
Builds an ML model to predict Airbnb listing prices
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'airbnb_nyc.csv')
output_path = os.path.join(project_dir, 'graphs', '08_price_prediction_model.png')

print("Loading data...")
df = pd.read_csv(data_path)

# Data preparation
df = df[df['price'] > 0]
df = df[df['price'] <= 1000]  # Remove extreme outliers

# Feature engineering
df['name_length'] = df['name'].fillna('').apply(len)
df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df['high_availability'] = (df['availability_365'] > 180).astype(int)

# Encode categorical variables
le_borough = LabelEncoder()
le_room = LabelEncoder()
df['borough_encoded'] = le_borough.fit_transform(df['neighbourhood_group'])
df['room_encoded'] = le_room.fit_transform(df['room_type'])

# Select features
features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
            'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
            'borough_encoded', 'room_encoded', 'name_length', 'has_reviews', 'high_availability']

X = df[features].fillna(0)
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models...")

# Train multiple models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }

# Get best model
best_name = min(results, key=lambda x: results[x]['mae'])
best_result = results[best_name]
best_model = best_result['model']

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

print(f"\nBest Model: {best_name}")
print(f"  MAE: ${results[best_name]['mae']:.2f}")
print(f"  R2: {results[best_name]['r2']:.3f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Price Prediction Model (Machine Learning)', fontsize=22, fontweight='bold', color='white', y=0.98)

# ----- Plot 1: Model Comparison -----
ax1 = axes[0, 0]
ax1.set_facecolor('#0d1117')

model_names = list(results.keys())
maes = [results[n]['mae'] for n in model_names]
r2s = [results[n]['r2'] for n in model_names]

x = np.arange(len(model_names))
width = 0.35
colors = ['#4ecdc4', '#ff6b6b']

bars1 = ax1.bar(x - width/2, maes, width, label='MAE ($)', color=colors[0])
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x + width/2, [r*100 for r in r2s], width, label='R² (%)', color=colors[1])

ax1.set_ylabel('Mean Absolute Error ($)', color='white', fontsize=11)
ax1_twin.set_ylabel('R² Score (%)', color='white', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, color='white')
ax1.set_title('Model Performance Comparison', color='white', fontsize=14, fontweight='bold')
ax1.tick_params(colors='white')
ax1_twin.tick_params(colors='white')

for bar, val in zip(bars1, maes):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'${val:.0f}',
             ha='center', color='white', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, r2s):
    ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val*100:.1f}%',
                  ha='center', color='white', fontsize=10, fontweight='bold')

for spine in ax1.spines.values():
    spine.set_color('#30363d')
for spine in ax1_twin.spines.values():
    spine.set_color('#30363d')

# ----- Plot 2: Feature Importance -----
ax2 = axes[0, 1]
ax2.set_facecolor('#0d1117')

top_features = importance.head(10)
colors = plt.cm.viridis(np.linspace(0.9, 0.3, len(top_features)))

bars = ax2.barh(range(len(top_features)), top_features['importance'].values * 100, color=colors)

# Clean feature names
feature_labels = {
    'room_encoded': 'Room Type',
    'borough_encoded': 'Borough',
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'calculated_host_listings_count': 'Host Listings',
    'availability_365': 'Availability',
    'minimum_nights': 'Min Nights',
    'number_of_reviews': 'Reviews',
    'reviews_per_month': 'Reviews/Month',
    'name_length': 'Title Length',
    'has_reviews': 'Has Reviews',
    'high_availability': 'High Availability'
}

labels = [feature_labels.get(f, f) for f in top_features['feature']]
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(labels, color='white', fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel('Importance (%)', color='white', fontsize=11)
ax2.set_title('Top 10 Price Predictors', color='white', fontsize=14, fontweight='bold')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_color('#30363d')

for bar, val in zip(bars, top_features['importance'].values):
    ax2.text(val * 100 + 0.5, bar.get_y() + bar.get_height()/2, f'{val*100:.1f}%',
             va='center', color='white', fontsize=9)

# ----- Plot 3: Predicted vs Actual -----
ax3 = axes[1, 0]
ax3.set_facecolor('#0d1117')

y_pred_best = best_result['predictions']
sample_idx = np.random.choice(len(y_test), size=min(3000, len(y_test)), replace=False)

ax3.scatter(y_test.iloc[sample_idx], y_pred_best[sample_idx], alpha=0.3, s=10, c='#4ecdc4')
ax3.plot([0, 500], [0, 500], 'r--', linewidth=2, label='Perfect Prediction')

ax3.set_xlabel('Actual Price ($)', color='white', fontsize=11)
ax3.set_ylabel('Predicted Price ($)', color='white', fontsize=11)
ax3.set_title('Predicted vs Actual Prices', color='white', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 500)
ax3.set_ylim(0, 500)
ax3.legend(facecolor='#161b22', labelcolor='white')
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_color('#30363d')

# ----- Plot 4: Results Summary -----
ax4 = axes[1, 1]
ax4.set_facecolor('#161b22')
ax4.set_xticks([])
ax4.set_yticks([])
for spine in ax4.spines.values():
    spine.set_color('#30363d')

ax4.text(0.5, 0.95, 'Model Results Summary', fontsize=16, fontweight='bold',
         ha='center', va='top', color='white', transform=ax4.transAxes)

summary = [
    ('Best Model:', best_name, '#d4a72c'),
    ('Mean Absolute Error:', f'${best_result["mae"]:.2f}', '#ff6b6b'),
    ('R-Squared Score:', f'{best_result["r2"]*100:.1f}%', '#4ecdc4'),
    ('Root Mean Sq Error:', f'${best_result["rmse"]:.2f}', '#f0883e'),
    ('Training Samples:', f'{len(X_train):,}', '#58a6ff'),
    ('Test Samples:', f'{len(X_test):,}', '#58a6ff'),
    ('Features Used:', f'{len(features)}', '#a371f7'),
    ('Top Predictor:', feature_labels.get(importance.iloc[0]['feature'], importance.iloc[0]['feature']), '#56d364'),
]

for i, (label, value, color) in enumerate(summary):
    y_pos = 0.82 - i * 0.09
    ax4.text(0.1, y_pos, label, fontsize=12, color='#8b949e', transform=ax4.transAxes, va='center')
    ax4.text(0.55, y_pos, value, fontsize=12, color=color, fontweight='bold', transform=ax4.transAxes, va='center')

ax4.text(0.5, 0.08, f'Model can predict prices within ${best_result["mae"]:.0f} on average',
         fontsize=11, ha='center', color='#56d364', transform=ax4.transAxes)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print(f"Saved: {output_path}")
