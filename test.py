from pyautoplot import AutoPlot
import os

# Initialize with a CSV file
dataset_path = "energy_consumption_dataset.csv"
plotter = AutoPlot(dataset_path)

# 1. Data Cleanup (Optional)
print("Performing data cleanup...")
plotter.cleanup(strategy='mean')

# 2. Exporting Statistical Analysis to JSON
print("Exporting analysis to JSON...")
plotter.export_json("analysis_output.json")

# 3. Exporting Comprehensive Reports
print("Exporting HTML report...")
plotter.export_report("comprehensive_report.html", format="html")

print("Exporting PDF report...")
plotter.export_report("comprehensive_report.pdf", format="pdf")

# 4. Automatically analyze and plot with new types (Correlation, Violin)
print("Running auto_plot...")
plotter.auto_plot(output_file='test_run', theme="dark", color='orange', excludes=['pairwise_scatter'])

# 5. Manually plot new data types
print("Manual plotting...")
plotter.plot(plot_type="correlation")
plotter.plot(plot_type="violin", x="EnergyConsumption")
plotter.plot(plot_type="scatter", x="Temperature", y="EnergyConsumption")

print("\nTest completed successfully!")
print("Outputs generated: analysis_output.json, comprehensive_report.html, comprehensive_report.pdf, and various test_run_*.png files.")