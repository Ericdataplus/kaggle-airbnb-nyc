"""
Comprehensive Mobile-Optimized Graphs for NYC Airbnb Analysis
Creates 15+ mobile versions covering all key insights.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_dir = os.path.join(project_dir, 'graphs_mobile')
os.makedirs(output_dir, exist_ok=True)

# Mobile style
M = {
    'figsize': (6, 8), 'figsize_wide': (6, 6),
    'bg': '#0d1117', 'text': '#ffffff', 'gray': '#8b949e', 'grid': '#30363d',
    'red': '#ff6b6b', 'green': '#56d364', 'blue': '#58a6ff',
    'gold': '#ffd700', 'purple': '#a371f7', 'orange': '#f0883e', 'teal': '#4ecdc4'
}

def setup():
    plt.rcParams.update({
        'font.size': 12, 'figure.facecolor': M['bg'], 'axes.facecolor': M['bg'],
        'text.color': M['text'], 'axes.labelcolor': M['text'],
        'xtick.color': M['text'], 'ytick.color': M['text']
    })

def ax_style(ax):
    ax.set_facecolor(M['bg'])
    for s in ax.spines.values(): s.set_color(M['grid'])

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name), dpi=200, facecolor=M['bg'], bbox_inches='tight')
    plt.close()
    print(f"   ✅ {name}")

# ============================================================
def g01_stats():
    print("📱 01: Key Stats")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.95, 'NYC Airbnb Analysis', fontsize=22, fontweight='bold',
            ha='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.88, '2019-2025 Multi-Year Study', fontsize=14,
            ha='center', color=M['gray'], transform=ax.transAxes)
    
    stats = [
        ('102,599', 'Total Listings', M['blue']),
        ('$152', 'Avg Price', M['green']),
        ('5', 'Boroughs', M['purple']),
        ('221', 'Neighborhoods', M['orange']),
        ('5', 'ML Algorithms', M['gold']),
        ('0.52', 'R² Score', M['teal']),
    ]
    
    for i, (val, label, color) in enumerate(stats):
        y = 0.72 - i * 0.11
        ax.text(0.5, y, val, fontsize=34, fontweight='bold', ha='center', color=color, transform=ax.transAxes)
        ax.text(0.5, y - 0.03, label, fontsize=11, ha='center', color=M['gray'], transform=ax.transAxes)
    
    save('01_stats.png')

def g02_boroughs():
    print("📱 02: Price by Borough")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax)
    
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Is.']
    prices = [197, 124, 99, 87, 115]
    colors = [M['red'], M['teal'], M['blue'], M['green'], M['gold']]
    
    y_pos = np.arange(len(boroughs))
    bars = ax.barh(y_pos, prices, color=colors, height=0.6)
    
    for bar, price in zip(bars, prices):
        ax.text(price - 10, bar.get_y() + bar.get_height()/2, f'${price}',
                va='center', ha='right', color='white', fontsize=16, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(boroughs, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel('Average Price ($)', fontsize=13)
    ax.set_title('Price by Borough', fontsize=18, fontweight='bold', pad=15)
    
    save('02_boroughs.png')

def g03_room_types():
    print("📱 03: Room Types")
    fig, ax = plt.subplots(figsize=M['figsize_wide'])
    ax_style(ax)
    
    types = ['Entire Home', 'Private Room', 'Shared Room']
    pcts = [52, 45, 3]
    colors = [M['red'], M['blue'], M['green']]
    
    wedges, texts, autotexts = ax.pie(pcts, labels=types, autopct='%1.0f%%',
                                       colors=colors, textprops={'color': 'white', 'fontsize': 12})
    for at in autotexts: at.set_fontweight('bold'); at.set_fontsize(14)
    
    ax.set_title('Room Types', fontsize=18, fontweight='bold', pad=15, color='white')
    save('03_room_types.png')

def g04_top_neighborhoods():
    print("📱 04: Top Neighborhoods")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax)
    
    hoods = ['Williamsburg', 'Bedford-Stuyvesant', 'Harlem', 'Bushwick', 'Upper West Side']
    listings = [3920, 3714, 2658, 2465, 1971]
    colors = [M['red'], M['blue'], M['green'], M['gold'], M['purple']]
    
    y_pos = np.arange(len(hoods))
    bars = ax.barh(y_pos, listings, color=colors, height=0.6)
    
    for bar, count in zip(bars, listings):
        ax.text(count + 50, bar.get_y() + bar.get_height()/2, f'{count:,}',
                va='center', fontsize=12, fontweight='bold', color='white')
    
    ax.set_yticks(y_pos); ax.set_yticklabels(hoods, fontsize=11)
    ax.invert_yaxis()
    ax.set_title('Top 5 Neighborhoods', fontsize=18, fontweight='bold', pad=15)
    
    save('04_neighborhoods.png')

def g05_price_premium():
    print("📱 05: Manhattan Premium")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.90, 'Manhattan Premium', fontsize=20, fontweight='bold',
            ha='center', color='white', transform=ax.transAxes)
    
    ax.text(0.5, 0.65, '+$73', fontsize=72, fontweight='bold',
            ha='center', color=M['red'], transform=ax.transAxes)
    ax.text(0.5, 0.52, 'per night more', fontsize=16,
            ha='center', color=M['gray'], transform=ax.transAxes)
    
    ax.text(0.25, 0.35, 'Manhattan', fontsize=14, ha='center', color=M['gray'], transform=ax.transAxes)
    ax.text(0.25, 0.28, '$197', fontsize=28, fontweight='bold', ha='center', color=M['red'], transform=ax.transAxes)
    
    ax.text(0.75, 0.35, 'Brooklyn', fontsize=14, ha='center', color=M['gray'], transform=ax.transAxes)
    ax.text(0.75, 0.28, '$124', fontsize=28, fontweight='bold', ha='center', color=M['teal'], transform=ax.transAxes)
    
    ax.text(0.5, 0.10, '+59% premium for Manhattan', fontsize=12, ha='center', color=M['gold'], transform=ax.transAxes)
    save('05_manhattan_premium.png')

def g06_year_comparison():
    print("📱 06: Year Comparison")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.92, '6-Year Evolution', fontsize=20, fontweight='bold', ha='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.85, '2019 → 2025', fontsize=16, ha='center', color=M['purple'], transform=ax.transAxes)
    
    changes = [
        ('Listings', '48,884 → 21,110', '-57%', M['red']),
        ('Avg Price', '$153 → $235', '+54%', M['green']),
        ('Entire Home', '52% → 68%', '+16%', M['blue']),
    ]
    
    for i, (label, vals, pct, color) in enumerate(changes):
        y = 0.68 - i * 0.20
        ax.text(0.5, y + 0.05, label, fontsize=14, ha='center', color=M['gray'], transform=ax.transAxes)
        ax.text(0.5, y - 0.02, vals, fontsize=18, fontweight='bold', ha='center', color='white', transform=ax.transAxes)
        ax.text(0.5, y - 0.08, pct, fontsize=24, fontweight='bold', ha='center', color=color, transform=ax.transAxes)
    
    ax.text(0.5, 0.08, 'NYC regulations reduced supply, prices surged', fontsize=11, ha='center', color=M['gray'], transform=ax.transAxes)
    save('06_year_comparison.png')

def g07_ml_winner():
    print("📱 07: ML Winner")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.90, '🏆 Best ML Model', fontsize=18, ha='center', color=M['gold'], transform=ax.transAxes)
    ax.text(0.5, 0.75, 'Random Forest', fontsize=32, fontweight='bold', ha='center', color='white', transform=ax.transAxes)
    
    ax.text(0.5, 0.55, '0.52', fontsize=72, fontweight='bold', ha='center', color=M['green'], transform=ax.transAxes)
    ax.text(0.5, 0.42, 'R² Score', fontsize=16, ha='center', color=M['gray'], transform=ax.transAxes)
    
    ax.text(0.5, 0.28, '$84 MAE', fontsize=24, fontweight='bold', ha='center', color=M['blue'], transform=ax.transAxes)
    ax.text(0.5, 0.20, 'Mean Absolute Error', fontsize=12, ha='center', color=M['gray'], transform=ax.transAxes)
    
    ax.text(0.5, 0.08, 'Room type + location = 60% of predictions', fontsize=11, ha='center', color=M['gold'], transform=ax.transAxes)
    save('07_ml_winner.png')

def g08_models_compared():
    print("📱 08: Models Compared")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax)
    
    models = ['Random Forest', 'Gradient Boost', 'Linear Reg', 'Ridge', 'Lasso']
    r2 = [0.52, 0.48, 0.42, 0.41, 0.39]
    colors = [M['green'], M['blue'], M['purple'], M['orange'], M['red']]
    
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, r2, color=colors, height=0.6)
    
    for bar, score in zip(bars, r2):
        ax.text(score + 0.02, bar.get_y() + bar.get_height()/2, f'{score:.2f}',
                va='center', fontsize=14, fontweight='bold', color='white')
    
    ax.set_yticks(y_pos); ax.set_yticklabels(models, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('R² Score', fontsize=13)
    ax.set_title('5 ML Models Tested', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlim(0, 0.65)
    
    save('08_models.png')

def g09_market_segments():
    print("📱 09: Market Segments")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.92, 'Market Segmentation', fontsize=20, fontweight='bold', ha='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.85, 'K-Means Clustering', fontsize=14, ha='center', color=M['purple'], transform=ax.transAxes)
    
    segments = [
        ('Budget Stays', '< $75/night', '38%', M['green']),
        ('Popular Picks', '$75-150', '32%', M['blue']),
        ('Luxury Escapes', '$150-300', '18%', M['purple']),
        ('Long-Term', '30+ min stay', '8%', M['orange']),
        ('Hidden Gems', 'High value', '4%', M['gold']),
    ]
    
    for i, (name, desc, pct, color) in enumerate(segments):
        y = 0.72 - i * 0.13
        circle = plt.Circle((0.08, y), 0.025, transform=ax.transAxes, facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(0.14, y + 0.015, name, fontsize=14, fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.14, y - 0.02, desc, fontsize=10, color=M['gray'], transform=ax.transAxes)
        ax.text(0.88, y, pct, fontsize=16, fontweight='bold', ha='right', color=color, transform=ax.transAxes)
    
    save('09_segments.png')

def g10_anomalies():
    print("📱 10: Anomaly Detection")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.92, 'Price Anomalies', fontsize=20, fontweight='bold', ha='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.85, 'Isolation Forest', fontsize=14, ha='center', color=M['purple'], transform=ax.transAxes)
    
    ax.text(0.5, 0.68, '2,433', fontsize=56, fontweight='bold', ha='center', color=M['orange'], transform=ax.transAxes)
    ax.text(0.5, 0.58, 'Anomalies Found (5%)', fontsize=14, ha='center', color=M['gray'], transform=ax.transAxes)
    
    ax.text(0.25, 0.42, '1,108', fontsize=28, fontweight='bold', ha='center', color=M['red'], transform=ax.transAxes)
    ax.text(0.25, 0.35, 'Overpriced', fontsize=12, ha='center', color=M['gray'], transform=ax.transAxes)
    
    ax.text(0.75, 0.42, '1,325', fontsize=28, fontweight='bold', ha='center', color=M['green'], transform=ax.transAxes)
    ax.text(0.75, 0.35, 'Great Deals', fontsize=12, ha='center', color=M['gray'], transform=ax.transAxes)
    
    ax.text(0.5, 0.15, 'Find hidden value before others do!', fontsize=12, ha='center', color=M['gold'], transform=ax.transAxes)
    save('10_anomalies.png')

def g11_host_analysis():
    print("📱 11: Host Analysis")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.92, 'Host Analysis', fontsize=20, fontweight='bold', ha='center', color='white', transform=ax.transAxes)
    
    insights = [
        ('Multi-Listing Hosts', '3,879', 'Control 28% of supply', M['red']),
        ('Single-Listing', '14,214', 'Most common type', M['green']),
        ('Superhosts', '4,567', 'Premium pricing power', M['gold']),
    ]
    
    for i, (title, val, sub, color) in enumerate(insights):
        y = 0.70 - i * 0.20
        ax.text(0.5, y + 0.05, title, fontsize=14, ha='center', color=M['gray'], transform=ax.transAxes)
        ax.text(0.5, y - 0.02, val, fontsize=36, fontweight='bold', ha='center', color=color, transform=ax.transAxes)
        ax.text(0.5, y - 0.08, sub, fontsize=11, ha='center', color=M['gray'], transform=ax.transAxes)
    
    save('11_hosts.png')

def g12_data_sources():
    print("📱 12: Data Sources")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.92, '3 Data Sources', fontsize=20, fontweight='bold', ha='center', color='white', transform=ax.transAxes)
    
    sources = [
        ('Kaggle 2019', '48,895', 'dgomonov', M['teal']),
        ('Kaggle Extended', '102,599', 'arianazmoudeh', M['red']),
        ('Inside Airbnb 2025', '36,111', 'Oct 2025', M['green']),
    ]
    
    for i, (name, count, source, color) in enumerate(sources):
        y = 0.68 - i * 0.20
        ax.text(0.5, y + 0.05, name, fontsize=14, ha='center', color=M['gray'], transform=ax.transAxes)
        ax.text(0.5, y - 0.02, count, fontsize=36, fontweight='bold', ha='center', color=color, transform=ax.transAxes)
        ax.text(0.5, y - 0.08, source, fontsize=11, ha='center', color=M['gray'], transform=ax.transAxes)
    
    ax.text(0.5, 0.10, '6 years of market evolution', fontsize=12, ha='center', color=M['gold'], transform=ax.transAxes)
    save('12_data_sources.png')

def g13_key_takeaways():
    print("📱 13: Key Takeaways")
    fig, ax = plt.subplots(figsize=M['figsize'])
    ax_style(ax); ax.axis('off')
    
    ax.text(0.5, 0.95, 'Key Takeaways', fontsize=20, fontweight='bold', ha='center', color='white', transform=ax.transAxes)
    
    takeaways = [
        ('1', 'Manhattan = +59% premium', 'Location is everything', M['red']),
        ('2', '52% entire homes', 'Most popular type', M['blue']),
        ('3', 'RF beats others', 'R² = 0.52', M['green']),
        ('4', '-57% listings (6yr)', 'NYC regulations impact', M['orange']),
        ('5', '5 market segments', 'K-Means clustering', M['purple']),
    ]
    
    for i, (num, head, sub, color) in enumerate(takeaways):
        y = 0.82 - i * 0.14
        circle = plt.Circle((0.08, y), 0.025, transform=ax.transAxes, facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(0.08, y, num, fontsize=12, fontweight='bold', color='white', ha='center', va='center', transform=ax.transAxes)
        ax.text(0.14, y + 0.015, head, fontsize=12, fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.14, y - 0.02, sub, fontsize=10, color=M['gray'], transform=ax.transAxes)
    
    save('13_takeaways.png')

# ============================================================
if __name__ == '__main__':
    print("\n📱 Generating Comprehensive Mobile Graphs (Airbnb NYC)")
    print("=" * 60)
    setup()
    
    g01_stats()
    g02_boroughs()
    g03_room_types()
    g04_top_neighborhoods()
    g05_price_premium()
    g06_year_comparison()
    g07_ml_winner()
    g08_models_compared()
    g09_market_segments()
    g10_anomalies()
    g11_host_analysis()
    g12_data_sources()
    g13_key_takeaways()
    
    print(f"\n✅ 13 mobile graphs saved to: {output_dir}")
