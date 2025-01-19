# PyAutoPlot
![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
[![Code Size](https://img.shields.io/github/languages/code-size/infinitode/pyautoplot)](https://github.com/infinitode/pyautoplot)
![Downloads](https://pepy.tech/badge/pyautoplot)
![License Compliance](https://img.shields.io/badge/license-compliance-brightgreen.svg)
![PyPI Version](https://img.shields.io/pypi/v/pyautoplot)

PyAutoPlot is an open-source Python library designed to make dataset analysis much easier by generating helpful detailed plots using `matplotlib`. It automatically generates appropriate plots based on the dataset you feed it.

### Changes in version 1.0.1:
- Added package dependencies to PyAutoPlot: `matplotlib>=3.0.0`, `pandas>=1.0.0`, and `numpy>=1.18.0`.

### Changes in version 1.0.0:
- `AutoPlot` class, along with `auto_plot` and `plot` functions. `auto_plot` automatically generates suitable plots based on values in your dataset.

> [!IMPORTANT]
> The `AutoPlot` object needs to be initialized with a **CSV dataset** before any plots can be generated using either `plot` or `auto_plot`.

## Installation

You can install PyAutoPlot using pip:

```bash
pip install pyautoplot
```

## Supported Python Versions

PyAutoPlot supports the following Python versions:

- Python 3.6
- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11/Later (Preferred)

Please ensure that you have one of these Python versions installed before using PyAutoPlot. PyAutoPlot may not work as expected on lower versions of Python than the supported.

## Features

- **Automatic Plotting**: PyAutoPlot automatically generates appropriate plots based on values present in your dataset. Currently, PyAutoPlot supports the following types of data: `numeric`, `categorical`, and `time-series`.
- **Customization**: Due to matplotlib's high customizability, you can create and use custom themes for PyAutoPlot, just pass in your theme as a dictionary of RCParams (See `plot.rcParams.keys()` for a list of valid parameters) or choose from our predefined themes. You can also pass in additional parameters to the function that `matplotlib` can recognize (e.g. `color='#5d17eb'`).

## Usage

### AutoPlot

```python
from pyautoplot import AutoPlot

# Initialize with a CSV file
plotter = AutoPlot("path/to/dataset.csv")

# Automatically analyze and plot
plotter.auto_plot(output_file='test', theme="dark", color='orange', excludes=['detailed_analysis'])
```

> [!NOTE]
> PyAutoPlot may not work well with certain datasets. If you find any issues please open an issue.

### Customization

```python
from pyautoplot import AutoPlot

# Define your custom theme
custom_theme = {
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "figure.facecolor": "#ffffff",
    "grid.color": "#dddddd",
    "text.color": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "legend.frameon": True,
}

# Initialize with a CSV file
plotter = AutoPlot("path/to/dataset.csv")

# Automatically analyze and plot using the custom theme
plotter.auto_plot(output_file='test', theme=custom_theme, color='orange', excludes=['detailed_analysis'])
```

## Contributing

Contributions are welcome! If you encounter any issues, have suggestions, or want to contribute to PyAutoPlot, please open an issue or submit a pull request on [GitHub](https://github.com/infinitode/pyautoplot).

## License

PyAutoPlot is released under the terms of the **MIT License (Modified)**. Please see the [LICENSE](https://github.com/infinitode/pyautoplot/blob/main/LICENSE) file for the full text.

**Modified License Clause**

The modified license clause grants users the permission to make derivative works based on the PyAutoPlot software. However, it requires any substantial changes to the software to be clearly distinguished from the original work and distributed under a different name.
