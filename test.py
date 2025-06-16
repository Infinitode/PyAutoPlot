import os
import shutil
from pyautoplot import AutoPlot
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = "test_output"
DATASET_PATH = "energy_consumption_dataset.csv"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"PyAutoPlot Test. Output: {os.path.abspath(OUTPUT_DIR)}")

if not os.path.exists(DATASET_PATH):
    print(f"Dummy data created for {DATASET_PATH}")
    dummy_data = {
        'Timestamp': pd.to_datetime([f'2023-01-01 {h:02}:00' for h in range(24)] * 3), # 3 days
        'Temperature': [10+i*0.1 for i in range(24*3)],
        'EnergyConsumption': [100+i for i in range(24*3)],
        'DayOfWeek': (['Mon']*24 + ['Tue']*24 + ['Wed']*24),
        'Category': (['A','B','C']*24)
    }
    pd.DataFrame(dummy_data).to_csv(DATASET_PATH, index=False)

try:
    plotter = AutoPlot(DATASET_PATH)
    print("AutoPlot initialized.")
except Exception as e:
    print(f"Init Error: {e}"); exit()

print("Testing auto_plot()...")
try:
    plotter.auto_plot(output_file=os.path.join(OUTPUT_DIR, "ap_default.png"))
except Exception as e: print(f"Error default auto_plot: {e}")

try:
    plotter.auto_plot(output_file=os.path.join(OUTPUT_DIR, "ap_custom.png"), theme="dark", excludes=['pairwise_scatter'])
except Exception as e: print(f"Error custom auto_plot: {e}")

print("Testing manual plot()...")
n1 = plotter.numeric[0] if plotter.numeric else None
n2 = plotter.numeric[1] if len(plotter.numeric) > 1 else n1
c1 = plotter.categorical[0] if plotter.categorical else None

if n1 and n2:
    try:
        plotter.plot(plot_type="scatter", x=n1, y=n2, title="Scatter")
        plt.savefig(os.path.join(OUTPUT_DIR, "m_scatter.png")); plt.show(); plt.close('all')
    except Exception as e: print(f"Error scatter: {e}")
if n1:
    try:
        plotter.plot(plot_type="distribution", x=n1, title="Dist", bins=10)
        plt.savefig(os.path.join(OUTPUT_DIR, "m_dist.png")); plt.show(); plt.close('all')
    except Exception as e: print(f"Error distribution: {e}")
    try:
        plotter.plot(plot_type="boxplot", x=n1, title="Box")
        plt.savefig(os.path.join(OUTPUT_DIR, "m_box.png")); plt.show(); plt.close('all')
    except Exception as e: print(f"Error boxplot: {e}")
if c1:
    try:
        plotter.plot(plot_type="bar", x=c1, title="Bar")
        plt.savefig(os.path.join(OUTPUT_DIR, "m_bar.png")); plt.show(); plt.close('all')
    except Exception as e: print(f"Error bar: {e}")

print("Testing customize()...")
if n1:
    try:
        plotter.customize(**{"font.size": 8, "figure.facecolor": "lightyellow"})
        plotter.plot(plot_type="distribution", x=n1, title="Custom Dist")
        plt.savefig(os.path.join(OUTPUT_DIR, "m_custom_dist.png")); plt.show(); plt.close('all')
        import matplotlib
        plotter.customize(**matplotlib.rcParamsDefault) # Revert
    except Exception as e: print(f"Error customize: {e}")

print("Testing small dataset...")
try:
    small_data = {'A': [1,2,3], 'B': ['x','y','z'], 'C': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])}
    small_df_path = os.path.join(OUTPUT_DIR, "small_ds.csv")
    pd.DataFrame(small_data).to_csv(small_df_path, index=False)
    small_plotter = AutoPlot(csv_path=small_df_path)
    small_plotter.auto_plot(output_file=os.path.join(OUTPUT_DIR, "ap_small_ds.png"))
except Exception as e: print(f"Error small dataset: {e}")

print("Test Script Finished.")
```
