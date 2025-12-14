# 🏙️ NYC Airbnb Market Analysis

> 📊 **Multi-Source Analysis** combining data from:
> - [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (48,895 listings)
> - [Airbnb Open Data](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata) (102,599 listings)
>
> Comprehensive geographic and pricing analysis of NYC's short-term rental market.

🔗 **[View Live Dashboard](https://ericdataplus.github.io/kaggle-airbnb-nyc/)**

![Summary Dashboard](graphs/07_summary_dashboard.png)

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| Total Listings | **151,494** (combined) |
| Boroughs Covered | 5 |
| Neighborhoods | 221 |
| Unique Hosts | 37,457 |
| Avg Price (NYC) | $153/night |
| Most Expensive | Manhattan ($197/night) |

## 🗺️ Visualizations

### Geographic Analysis
- **NYC Listing Map** — All listings color-coded by borough
- **Price Heatmap** — Geographic distribution of prices

### Price Analysis
- **Price by Borough** — Manhattan leads at $197/night avg
- **Top 15 Most Expensive Neighborhoods**
- **Room Type Pricing** — Entire home vs Private room vs Shared

### Market Insights
- **Host Analysis** — Multi-listing hosts dominate supply
- **Neighborhood Rankings** — Most listed vs most expensive
- **Multi-Dataset Comparison** — Side-by-side source analysis

## 📁 Project Structure

```
kaggle-airbnb-nyc/
├── index.html                    # Interactive Dashboard
├── graphs/                       # Visualizations
│   ├── 01_price_by_borough.png
│   ├── 02_room_type_analysis.png
│   ├── 03_nyc_map.png
│   ├── 04_top_neighborhoods.png
│   ├── 05_dataset_comparison.png
│   ├── 06_host_analysis.png
│   └── 07_summary_dashboard.png
├── scripts/                      # Analysis scripts
│   ├── 01_price_by_borough.py
│   ├── 02_room_type_analysis.py
│   ├── 03_nyc_map.py
│   ├── 04_top_neighborhoods.py
│   ├── 05_dataset_comparison.py
│   ├── 06_host_analysis.py
│   ├── 07_summary_dashboard.py
│   └── run_all.py
└── README.md
```

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas** — Data manipulation
- **Matplotlib** — Visualizations
- **NumPy** — Numerical computing

## 📦 Data Sources

| Dataset | Source | Records |
|---------|--------|---------|
| NYC Airbnb 2019 | [dgomonov/new-york-city-airbnb-open-data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) | 48,895 |
| Airbnb Open Data | [arianazmoudeh/airbnbopendata](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata) | 102,599 |
| **Combined** | Multi-source analysis | **151,494** |

## 🔍 Key Findings

1. **Manhattan dominates pricing** — $197/night average, 72% premium over outer boroughs
2. **Entire homes are 2x private rooms** — $212 vs $89 average
3. **Williamsburg most listed** — 3,920 listings in one neighborhood
4. **Professional hosts** — Top 10 hosts control 1,000+ listings
5. **Tribeca most expensive** — $490/night average

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/Ericdataplus/kaggle-airbnb-nyc.git
cd kaggle-airbnb-nyc

# Download data from Kaggle (links above)
# Place CSV files in project root

# Run analysis
python scripts/run_all.py
```

---

Made with 🏙️ by [Ericdataplus](https://github.com/Ericdataplus) | December 2024
