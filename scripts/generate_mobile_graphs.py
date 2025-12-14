"""
Mobile-Optimized Graphs Generator for Airbnb NYC Analysis
Generates 6 key graphs optimized for phone screens with:
- Larger fonts (16pt+ titles, 14pt+ labels)
- Simple layouts (single focus per graph)
- Portrait orientation (600x800)
- High contrast dark theme
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_dir = os.path.join(project_dir, 'graphs_mobile')

os.makedirs(output_dir, exist_ok=True)

# Mobile style settings
MOBILE_CONFIG = {
    'figsize': (6, 8),  # Portrait for phones
    'title_size': 20,
    'label_size': 16,
    'tick_size': 14,
    'value_size': 18,
    'bg_color': '#0d1117',
    'text_color': '#ffffff',
    'grid_color': '#30363d',
}

def setup_mobile_style():
    """Set matplotlib defaults for mobile viewing"""
    plt.rcParams['font.size'] = MOBILE_CONFIG['tick_size']
    plt.rcParams['axes.titlesize'] = MOBILE_CONFIG['title_size']
    plt.rcParams['axes.labelsize'] = MOBILE_CONFIG['label_size']
    plt.rcParams['xtick.labelsize'] = MOBILE_CONFIG['tick_size']
    plt.rcParams['ytick.labelsize'] = MOBILE_CONFIG['tick_size']
    plt.rcParams['figure.facecolor'] = MOBILE_CONFIG['bg_color']
    plt.rcParams['axes.facecolor'] = MOBILE_CONFIG['bg_color']
    plt.rcParams['text.color'] = MOBILE_CONFIG['text_color']
    plt.rcParams['axes.labelcolor'] = MOBILE_CONFIG['text_color']
    plt.rcParams['xtick.color'] = MOBILE_CONFIG['text_color']
    plt.rcParams['ytick.color'] = MOBILE_CONFIG['text_color']

def style_axes(ax):
    """Apply consistent styling to axes"""
    ax.set_facecolor(MOBILE_CONFIG['bg_color'])
    for spine in ax.spines.values():
        spine.set_color(MOBILE_CONFIG['grid_color'])
        spine.set_linewidth(2)

# ============================================================
# GRAPH 1: Big Stats Display
# ============================================================
def generate_stats():
    print("📱 Generating: Key Stats (mobile)")
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.92, 'NYC Airbnb', fontsize=24, fontweight='bold',
            ha='center', va='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.85, 'Dataset Overview', fontsize=16,
            ha='center', va='center', color='#8b949e', transform=ax.transAxes)
    
    stats = [
        ('102,599', 'Total Listings', '#ff6b6b'),
        ('$152', 'Average Price', '#56d364'),
        ('5 Boroughs', 'Coverage', '#58a6ff'),
        ('221', 'Neighborhoods', '#a371f7'),
    ]
    
    for i, (value, label, color) in enumerate(stats):
        y = 0.68 - i * 0.17
        ax.text(0.5, y, value, fontsize=48, fontweight='bold',
                ha='center', va='center', color=color, transform=ax.transAxes)
        ax.text(0.5, y - 0.05, label, fontsize=14,
                ha='center', va='center', color='#8b949e', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_stats.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 01_stats.png")

# ============================================================
# GRAPH 2: Price by Borough
# ============================================================
def generate_borough():
    print("📱 Generating: Price by Borough (mobile)")
    
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Is.']
    prices = [196, 124, 99, 87, 115]
    colors = ['#ff6b6b', '#4facfe', '#56d364', '#feca57', '#a371f7']
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    
    y_pos = np.arange(len(boroughs))
    bars = ax.barh(y_pos, prices, color=colors, height=0.6)
    
    for bar, price in zip(bars, prices):
        ax.text(price - 10, bar.get_y() + bar.get_height()/2, f'${price}',
                va='center', ha='right', color='white', fontsize=18, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(boroughs, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, 220)
    ax.set_xlabel('Average Price ($)', fontsize=14)
    ax.set_title('Price by Borough', fontsize=MOBILE_CONFIG['title_size'], fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_boroughs.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 02_boroughs.png")

# ============================================================
# GRAPH 3: Room Types
# ============================================================
def generate_room_types():
    print("📱 Generating: Room Types (mobile)")
    
    types = ['Entire Home', 'Private Room', 'Shared Room']
    counts = [52, 45, 3]
    colors = ['#ff6b6b', '#4facfe', '#56d364']
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    
    wedges, texts, autotexts = ax.pie(counts, labels=types, autopct='%1.0f%%',
                                       colors=colors, textprops={'color': 'white', 'fontsize': 14},
                                       startangle=90)
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)
    
    ax.set_title('Room Types', fontsize=MOBILE_CONFIG['title_size'], fontweight='bold', pad=20, color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_room_types.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 03_room_types.png")

# ============================================================
# GRAPH 4: ML Model Performance
# ============================================================
def generate_ml():
    print("📱 Generating: ML Model (mobile)")
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.90, 'Price Prediction', fontsize=24, fontweight='bold',
            ha='center', va='center', color='white', transform=ax.transAxes)
    
    # Big R² score
    ax.text(0.5, 0.60, '0.52', fontsize=72, fontweight='bold',
            ha='center', va='center', color='#56d364', transform=ax.transAxes)
    ax.text(0.5, 0.45, 'R² Score', fontsize=20,
            ha='center', va='center', color='#8b949e', transform=ax.transAxes)
    
    # Model name
    ax.text(0.5, 0.28, 'Random Forest', fontsize=24, fontweight='bold',
            ha='center', va='center', color='white', transform=ax.transAxes)
    ax.text(0.5, 0.20, 'Best performing model', fontsize=14,
            ha='center', va='center', color='#58a6ff', transform=ax.transAxes)
    
    # RMSE
    ax.text(0.5, 0.08, 'RMSE: $84', fontsize=16,
            ha='center', va='center', color='#feca57', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_ml.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 04_ml.png")

# ============================================================
# GRAPH 5: Top Neighborhoods
# ============================================================
def generate_neighborhoods():
    print("📱 Generating: Top Neighborhoods (mobile)")
    
    neighborhoods = ['Williamsburg', 'Bedford-Stuyvesant', 'Harlem', 'Bushwick', 'Upper West Side']
    listings = [3920, 3714, 2658, 2465, 1971]
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    
    y_pos = np.arange(len(neighborhoods))
    colors = ['#ff6b6b', '#4facfe', '#56d364', '#feca57', '#a371f7']
    bars = ax.barh(y_pos, listings, color=colors, height=0.6)
    
    for bar, count in zip(bars, listings):
        ax.text(count + 100, bar.get_y() + bar.get_height()/2, f'{count:,}',
                va='center', ha='left', color='white', fontsize=14, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(neighborhoods, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 5000)
    ax.set_xlabel('Number of Listings', fontsize=14)
    ax.set_title('Top 5 Neighborhoods', fontsize=MOBILE_CONFIG['title_size'], fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_neighborhoods.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 05_neighborhoods.png")

# ============================================================
# GRAPH 6: Key Insights
# ============================================================
def generate_insights():
    print("📱 Generating: Key Insights (mobile)")
    
    fig, ax = plt.subplots(figsize=MOBILE_CONFIG['figsize'])
    style_axes(ax)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Key Insights', fontsize=MOBILE_CONFIG['title_size'],
            ha='center', va='center', color='white', fontweight='bold', transform=ax.transAxes)
    
    insights = [
        ('1', 'Manhattan = Premium', '+50% avg price', '#ff6b6b'),
        ('2', '52% Entire Homes', 'Most popular type', '#4facfe'),
        ('3', 'RF Beats Others', 'R² = 0.52', '#56d364'),
        ('4', 'Williamsburg #1', '3,920 listings', '#a371f7'),
    ]
    
    for i, (num, headline, subtext, color) in enumerate(insights):
        y = 0.78 - i * 0.20
        
        circle = plt.Circle((0.12, y), 0.05, transform=ax.transAxes,
                           facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.12, y, num, fontsize=18, fontweight='bold', color='white',
                ha='center', va='center', transform=ax.transAxes)
        
        ax.text(0.22, y + 0.02, headline, fontsize=18, fontweight='bold',
                color='white', va='center', transform=ax.transAxes)
        ax.text(0.22, y - 0.04, subtext, fontsize=14,
                color='#8b949e', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_insights.png'),
                dpi=200, facecolor=MOBILE_CONFIG['bg_color'], bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: 06_insights.png")

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n📱 Generating Mobile-Optimized Graphs (Airbnb NYC)")
    print("=" * 50)
    
    setup_mobile_style()
    
    generate_stats()
    generate_borough()
    generate_room_types()
    generate_ml()
    generate_neighborhoods()
    generate_insights()
    
    print("\n" + "=" * 50)
    print(f"✅ All mobile graphs saved to: {output_dir}")
    print("📱 6 graphs optimized for phone viewing")
