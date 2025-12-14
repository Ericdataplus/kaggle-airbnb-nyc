"""
Run all visualization scripts
"""
import subprocess
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

scripts = [
    '01_price_by_borough.py',
    '02_room_type_analysis.py',
    '03_nyc_map.py',
    '04_top_neighborhoods.py',
    '05_dataset_comparison.py',
    '06_host_analysis.py',
    '07_summary_dashboard.py',
]

print("🏙️ NYC Airbnb Analysis - Generating visualizations...")
print("=" * 50)

for script in scripts:
    script_path = os.path.join(script_dir, script)
    if os.path.exists(script_path):
        print(f"\nRunning: {script}")
        result = subprocess.run([sys.executable, script_path], 
                               capture_output=True, text=True, cwd=script_dir)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"Error: {result.stderr[:300]}")
    else:
        print(f"Not found: {script}")

print("\n" + "=" * 50)
print("All visualizations complete!")
