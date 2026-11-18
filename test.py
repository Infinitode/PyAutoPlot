"""Generate example reports and plots for the bundled energy dataset."""

from pathlib import Path

import matplotlib.pyplot as plt

from pyautoplot import AutoPlot


def main():
    output = Path("test_output")
    output.mkdir(exist_ok=True)
    plotter = AutoPlot(Path(__file__).with_name("energy_consumption_dataset.csv"))
    plotter.cleanup(strategy="mean")

    plotter.export_json(output / "analysis.json")
    plotter.export_report(output / "report.html")
    plotter.export_report(output / "report.pdf", format="pdf")
    plotter.auto_plot(output / "energy.png", theme="dark", color="orange",
                      excludes=["pairwise_scatter"], show=False, dpi=100)

    fig, ax = plt.subplots()
    plotter.plot("scatter", x="Temperature", y="EnergyConsumption", ax=ax, show=False)
    fig.savefig(output / "temperature_vs_energy.png")
    plt.close(fig)
    print(f"Example reports and charts saved to {output}/")


if __name__ == "__main__":
    main()
