from pyautoplot import AutoPlot

# Initialize with a CSV file
plotter = AutoPlot("energy_consumption_dataset.csv")

# Automatically analyze and plot
plotter.auto_plot(output_file='test', theme="dark", color='orange', excludes=['detailed_analysis'])

# Manually plot data
plotter.plot(plot_type="scatter", x="Month", y="Hour")