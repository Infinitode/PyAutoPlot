import base64
import io
import json
import math
import os
import re
import warnings
from contextlib import contextmanager
from html import escape

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_LIGHT_THEME = {
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
_DARK_THEME = {
    "axes.facecolor": "#222222",
    "axes.edgecolor": "#ffffff",
    "axes.labelcolor": "#ffffff",
    "figure.facecolor": "#222222",
    "grid.color": "#444444",
    "text.color": "#ffffff",
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",
    "legend.frameon": False,
}
_DATE_PREFIX = r"^(?:\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?:$|[T ])"
_TIMEZONE_SUFFIX = r"[T ]\d{1,2}:\d{2}.*(?:[Zz]|UTC|GMT|[+-]\d{2}(?::?\d{2})?)$"


@contextmanager
def _figure(*args, **kwargs):
    """Close only the figure this call creates, even if drawing or saving fails."""
    fig, axes = plt.subplots(*args, **kwargs)
    try:
        yield fig, axes
    finally:
        plt.close(fig)


def _filename_labels(columns):
    """Make safe, unique filename components from CSV headers."""
    used = set()
    labels = {}
    for column in columns:
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(column)).strip(" .") or "column"
        label = stem
        index = 2
        while label.casefold() in used:
            label = f"{stem}_{index}"
            index += 1
        labels[column] = label
        used.add(label.casefold())
    return labels


class AutoPlot:
    """
    A class for automatic and customized data visualization from CSV files.

    Attributes:
        data (DataFrame): The dataset loaded from the CSV file.
        categorical (list): List of categorical column names.
        numeric (list): List of numeric column names.
        time_series (list): List of time-series column names.

    Methods:
        auto_plot(output_file=None, theme="light", excludes=None, **kwargs):
            Automatically analyzes the dataset and generates multiple types of plots.

        plot(plot_type, x=None, y=None, **kwargs):
            Manually generates a specific plot based on user input.
    """

    def __init__(self, csv_path):
        """Load and classify data from a CSV path or a readable file-like object."""
        self.data = pd.read_csv(csv_path)
        self._try_parse_dates()
        self._detect_column_types()
        self.theme_settings = _LIGHT_THEME.copy()

    def _try_parse_dates(self):
        """Parse date-like text without overwriting non-missing, unparseable values."""
        # pandas 2+ otherwise infers one format for the whole column.
        options = {"format": "mixed"} if int(pd.__version__.split(".", 1)[0]) >= 2 else {}
        for column, series in self.data.items():
            if not pd.api.types.is_string_dtype(series.dtype):
                continue
            # Object columns can hold non-strings (e.g. booleans with missing values).
            values = series.dropna().astype(str).str.strip()
            if values.empty or not values.str.match(_DATE_PREFIX).all():
                continue

            converted = pd.to_datetime(values, errors="coerce", utc=True, **options)
            if converted.isna().any():
                continue
            if not values.str.contains(_TIMEZONE_SUFFIX).any():
                converted = converted.dt.tz_localize(None)
            self.data[column] = converted.reindex(series.index)

    def _detect_column_types(self):
        """Detect columns as categorical, numeric, or time-series based on their content."""
        self.categorical = []
        self.numeric = []
        self.time_series = []

        for column, series in self.data.items():
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                self.time_series.append(column)
            elif pd.api.types.is_numeric_dtype(series.dtype):
                # Preserve the categorical treatment of low-cardinality codes and flags.
                target = self.categorical if series.nunique() < 10 else self.numeric
                target.append(column)
            elif pd.api.types.is_string_dtype(series.dtype) or series.nunique() < 15:
                self.categorical.append(column)
            else:
                warnings.warn(f"Column '{column}' was not classified and will be ignored.", stacklevel=2)

    def cleanup(self, strategy='mean', fill_value=None, columns=None, drop_na=False):
        """
        Perform basic data cleanup and handle missing values.

        Parameters:
            strategy (str): Strategy to fill missing values ('mean', 'median', 'mode', 'constant'). Default is 'mean'.
            fill_value (any): Value to use if strategy is 'constant'.
            columns (list): List of columns to clean. If None, cleans all columns.
            drop_na (bool): If True, drops rows with missing values instead of filling them.
        """
        if columns is None:
            columns = self.data.columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]
        else:
            columns = list(columns)

        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in dataset.")

        if drop_na:
            self.data = self.data.dropna(subset=columns)
        else:
            if strategy not in ("mean", "median", "mode", "constant"):
                raise ValueError(f"Unknown cleanup strategy: {strategy!r}.")
            if strategy == "constant" and fill_value is None:
                raise ValueError("'fill_value' is required for constant cleanup.")

            for col in columns:
                series = self.data[col]
                if not series.isna().any():
                    continue
                if strategy in ("mean", "median") and pd.api.types.is_numeric_dtype(series):
                    fill = getattr(series, strategy)()
                    if pd.notna(fill):
                        self.data[col] = series.fillna(fill)
                elif strategy == "mode":
                    modes = series.mode()
                    if not modes.empty:
                        self.data[col] = series.fillna(modes.iloc[0])
                elif strategy == "constant":
                    self.data[col] = series.fillna(fill_value)

        self._detect_column_types()

    def _set_theme(self, theme):
        """Select an instance theme without changing matplotlib's global settings."""
        if theme == "light":
            settings = _LIGHT_THEME
        elif theme == "dark":
            settings = _DARK_THEME
        elif isinstance(theme, dict):
            try:
                if "backend" in theme:
                    raise ValueError("select the matplotlib backend separately")
                settings = mpl.RcParams({**_LIGHT_THEME, **theme})
            except (KeyError, TypeError, ValueError) as error:
                warnings.warn(f"Invalid custom theme ({error}); using the light theme.", stacklevel=3)
                settings = _LIGHT_THEME
        else:
            warnings.warn("Invalid theme; using the light theme.", stacklevel=3)
            settings = _LIGHT_THEME
        self.theme_settings = dict(settings)

    def _generate_analysis(self):
        """Generate detailed analysis of the dataset."""
        analysis = {}

        # Numeric Analysis
        for column in self.numeric:
            data_column = self.data[column].replace([np.inf, -np.inf], np.nan).dropna()
            n = len(data_column)  # Get the number of valid data points
            mean = data_column.mean()
            stddev = data_column.std()
            skewness = self._calculate_skewness(data=data_column, mean=mean, stddev=stddev, n=n)
            kurtosis = self._calculate_kurtosis(data=data_column, mean=mean, stddev=stddev, n=n)

            stats = {
                "Type": "Numeric",
                "Count": n,
                "Mean": mean,
                "Median": data_column.median(),
                "StdDev": stddev,
                "Min": data_column.min(),
                "Max": data_column.max(),
                "Skewness": skewness,
                "Kurtosis": kurtosis,
                "IQR": data_column.quantile(0.75) - data_column.quantile(0.25),
                "25th Percentile": data_column.quantile(0.25),
                "50th Percentile (Median)": data_column.quantile(0.50),
                "75th Percentile": data_column.quantile(0.75),
                "Outliers": self._detect_outliers(data_column),
                "Missing Values": len(self.data) - n,
            }
            analysis[column] = stats

        # Categorical Analysis
        for column in self.categorical:
            value_counts = self.data[column].value_counts()
            most_common_count = value_counts.iloc[0] if not value_counts.empty else 0
            analysis[column] = {
                "Type": "Categorical",
                "Count": self.data[column].count(),
                "Unique": len(value_counts),
                "Most Common": value_counts.index[0] if not value_counts.empty else None,
                "Most Common Count": most_common_count,
                "Top 3 Most Common": value_counts.head(3).to_dict(),
                "Balance Ratio": (
                    f"{most_common_count}:{value_counts.iloc[-1]} (Most:Least)"
                    if not value_counts.empty else None
                ),
                "Frequency Distribution": value_counts.to_dict(),
                "Mode Variability": value_counts[value_counts == most_common_count].index.tolist(),
                "Missing Values": self.data[column].isna().sum(),
            }

        # Time Series Analysis
        for column in self.time_series:
            data_column = self.data[column].dropna()
            analysis[column] = {
                "Type": "Time Series",
                "Count": len(data_column),
                "Min": data_column.min(),
                "Max": data_column.max(),
                "Mean": data_column.mean() if not data_column.empty else pd.NaT,
                "Median": data_column.quantile(0.5) if not data_column.empty else pd.NaT,
                "Missing Values": self.data[column].isna().sum(),
                "Autocorrelation": self._calculate_autocorrelation(data_column),
            }

        return analysis

    @staticmethod
    def _calculate_skewness(data, mean, stddev, n):
        """Calculate skewness of a dataset."""
        if n < 2 or stddev == 0 or not np.isfinite(stddev):
            return np.nan
        skewness = np.sum(((data - mean) / stddev) ** 3) / n
        return skewness

    @staticmethod
    def _calculate_kurtosis(data, mean, stddev, n):
        """Calculate kurtosis of a dataset."""
        if n < 2 or stddev == 0 or not np.isfinite(stddev):
            return np.nan
        kurtosis = (np.sum(((data - mean) / stddev) ** 4) / n) - 3
        return kurtosis

    @staticmethod
    def _calculate_autocorrelation(data):
        """Correlate elapsed timestamp values at lags 1–10 in CSV row order."""
        correlations = [np.nan] * 10
        if len(data) < 3:
            return correlations

        # Subtract before converting to floats to preserve short intervals at large epochs.
        ticks = data.astype("int64").tolist()
        elapsed = np.asarray([tick - ticks[0] for tick in ticks], dtype=float)
        for lag in range(1, min(11, len(elapsed) - 1)):
            left, right = elapsed[:-lag], elapsed[lag:]
            if np.ptp(left) and np.ptp(right):
                correlations[lag - 1] = np.corrcoef(left, right)[0, 1]
        return correlations

    @staticmethod
    def _detect_outliers(series, threshold=3):
        """Detect outliers in a numeric series based on standard deviation."""
        mean = series.mean()
        std_dev = series.std()
        outliers = series[(series < mean - threshold * std_dev) | (series > mean + threshold * std_dev)]
        return outliers.tolist()

    def _numeric_plot_data(self):
        return self.data[self.numeric].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    def _categorical_counts(self):
        for column in self.categorical:
            counts = self.data[column].value_counts()
            if not counts.empty:
                yield column, counts

    def _time_plot_data(self):
        # Small CSVs may classify numeric measurements with fewer than 10 distinct
        # values as categorical. They can still be plotted against a date column.
        numeric = self.data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        if numeric.empty:
            return
        for column in self.time_series:
            valid = self.data[column].notna()
            measurements = numeric.loc[valid].dropna(axis=1, how="all")
            if not measurements.empty:
                measurements.index = pd.DatetimeIndex(self.data.loc[valid, column], name=column)
                yield column, measurements.sort_index(kind="mergesort")

    @staticmethod
    def _draw_correlation_heatmap(data, fig, ax):
        corr = data.corr()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        ticks = np.arange(len(data.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(data.columns, rotation=45, ha="right")
        ax.set_yticklabels(data.columns)

        for i in ticks:
            for j in ticks:
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
        ax.set_title("Correlation Heatmap", fontsize=16, weight="bold")

    @staticmethod
    def _draw_violin(data, axes):
        for ax, column in zip(axes, data.columns):
            ax.violinplot(data[column].dropna())
            ax.set_title(f"Violin Plot of {column}", weight="bold")
            ax.set_ylabel("Value")
            ax.set_xticks([])

    @staticmethod
    def _draw_distribution(data, column, axes, bins=10, **kwargs):
        data[column].plot.hist(ax=axes[0], bins=bins, title=f"Distribution of {column}", **kwargs)
        axes[0].set_xlabel(column)
        axes[0].set_ylabel("Frequency")
        data.boxplot(column=column, ax=axes[1], **kwargs)
        axes[1].set_title(f"Boxplot of {column}")

    @staticmethod
    def _draw_bar(counts, column, ax, **kwargs):
        counts.plot.bar(ax=ax, title=f"Bar Plot of {column}", **kwargs)
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    @staticmethod
    def _text_box_style():
        return {"facecolor": plt.rcParams["axes.facecolor"], "edgecolor": plt.rcParams["axes.edgecolor"], "alpha": 0.8}

    @contextmanager
    def _analysis_figure(self, analysis):
        """Render the summary even if there are no analyzable columns."""
        rows = max(1, len(analysis))
        with _figure(rows, 1, figsize=(20, rows * 4), constrained_layout=True, squeeze=False) as (fig, axes):
            fig.suptitle("Detailed Dataset Analysis", fontsize=20, weight="bold")
            for ax, (column, stats) in zip(axes[:, 0], analysis.items()):
                stat_text = "\n".join(f"{key}: {value}" for key, value in stats.items())
                ax.text(0.5, 0.5, stat_text, fontsize=12, va="center", ha="center",
                        transform=ax.transAxes, bbox=self._text_box_style())
                ax.axis("off")
                ax.set_title(f"Analysis of {column}", fontsize=16, pad=20, weight="bold")
            if not analysis:
                ax = axes[0, 0]
                ax.text(0.5, 0.5, "No analyzable columns.", ha="center", va="center")
                ax.axis("off")
            yield fig

    def auto_plot(self, output_file=None, theme="light", excludes=None, *, show=True, dpi=300, bins=10, **kwargs):
        """Analyze the dataset and generate visualizations for the available data.

        ``output_file`` is a base filename. Each section appends its own suffix;
        if no extension is given, matplotlib's default save format is used.
        Set ``show=False`` to generate files without displaying the figures.
        ``dpi`` controls saved figures and ``bins`` controls histograms.

        Sections you can exclude: ``detailed_analysis``, ``numeric``,
        ``correlation``, ``violin``, ``categorical``, ``pairwise_scatter``,
        ``pie_charts``, ``line_plots``, and ``stacked_area``. The pairwise matrix
        may be slow for datasets with many numeric columns.

        Additional keyword arguments (e.g. ``color``) must be accepted by the
        pandas plots in the enabled sections. Use ``plot()`` to style an
        individual chart type.
        """
        self._set_theme(theme)
        with plt.rc_context(self.theme_settings):
            base, extension = os.path.splitext(output_file) if output_file else (None, "")
            extension = extension or f".{plt.rcParams['savefig.format']}"

            def finish(fig, suffix):
                if output_file:
                    fig.savefig(f"{base}_{suffix}{extension}", dpi=dpi)
                if show:
                    plt.show()

            self._auto_plot_sections(finish, set(excludes or ()), bins, kwargs)

    def _auto_plot_sections(self, finish, excludes, bins, kwargs):
        if "detailed_analysis" not in excludes:
            with self._analysis_figure(self._generate_analysis()) as fig:
                finish(fig, "analysis")

        numeric = self._numeric_plot_data()
        num_cols = len(numeric.columns)
        if num_cols and "numeric" not in excludes:
            with _figure(num_cols, 2, figsize=(18, num_cols * 5), constrained_layout=True,
                         squeeze=False) as (fig, axes):
                fig.suptitle("Numeric Variables: Distributions and Boxplots", fontsize=20, weight="bold")
                for column, row in zip(numeric.columns, axes):
                    self._draw_distribution(numeric, column, row, bins=bins, **kwargs)
                finish(fig, "numeric")

        if num_cols > 1 and "correlation" not in excludes:
            with _figure(figsize=(10, 8), constrained_layout=True) as (fig, ax):
                self._draw_correlation_heatmap(numeric, fig, ax)
                finish(fig, "correlation")

        if num_cols and "violin" not in excludes:
            with _figure(num_cols, 1, figsize=(10, num_cols * 4), constrained_layout=True,
                         squeeze=False) as (fig, axes):
                self._draw_violin(numeric, axes[:, 0])
                finish(fig, "violin")

        categories = list(self._categorical_counts()) if not {"categorical", "pie_charts"} <= excludes else []
        if categories and "categorical" not in excludes:
            with _figure(len(categories), 1, figsize=(18, len(categories) * 5), constrained_layout=True,
                         squeeze=False) as (fig, axes):
                fig.suptitle("Categorical Variables: Enhanced Bar Plots", fontsize=20, weight="bold")
                for (column, counts), ax in zip(categories, axes[:, 0]):
                    self._draw_bar(counts, column, ax, **kwargs)
                    ax.text(0.02, 0.95, f"Max: {counts.max()}\nMin: {counts.min()}\nAvg: {counts.mean():.2f}",
                            transform=ax.transAxes, fontsize=10, va="top", bbox=self._text_box_style())
                finish(fig, "categorical")

        if num_cols > 1 and "pairwise_scatter" not in excludes:
            if num_cols > 10:
                warnings.warn(f"Pairwise scatter plots for {num_cols} numeric columns may be slow.", stacklevel=3)
            with _figure(num_cols, num_cols, figsize=(20, 20), constrained_layout=True) as (fig, axes):
                fig.suptitle("Pairwise Scatter Plots of Numeric Variables", fontsize=20, weight="bold")
                for i, y in enumerate(numeric.columns):
                    for j, x in enumerate(numeric.columns):
                        ax = axes[i, j]
                        if i == j:
                            numeric[x].plot.hist(ax=ax, bins=bins, title=f"{x} Distribution", **kwargs)
                            ax.set_xlabel(x)
                        else:
                            numeric.plot.scatter(x=x, y=y, ax=ax, title=f"{x} vs {y}", **kwargs)
                finish(fig, "pairwise")

        if "pie_charts" not in excludes:
            labels = _filename_labels(column for column, _ in categories)
            pie_options = dict(kwargs)
            color = pie_options.pop("color", None)
            if color is not None:
                # Older pandas forwards "color" to Axes.pie, which only takes "colors".
                pie_options.setdefault("colors", [color] if mpl.colors.is_color_like(color) else color)
            for column, counts in categories:
                with _figure(figsize=(8, 8)) as (fig, ax):
                    counts.plot.pie(ax=ax, autopct="%1.1f%%", title=f"Pie Chart of {column}", **pie_options)
                    ax.set_ylabel("")
                    finish(fig, f"pie_{labels[column]}")

        area_pending = "stacked_area" not in excludes
        lines_enabled = "line_plots" not in excludes
        if self.time_series and (lines_enabled or area_pending):
            labels = _filename_labels(self.time_series)
            for column, data in self._time_plot_data():
                if lines_enabled:
                    with _figure(figsize=(10, 6), constrained_layout=True) as (fig, ax):
                        data.plot.line(ax=ax, title=f"Values over {column}", **kwargs)
                        ax.set_xlabel(column)
                        ax.set_ylabel("Value")
                        finish(fig, f"line_{labels[column]}")
                if area_pending:
                    # pandas cannot stack a series that contains both positive and negative values.
                    mixed_signs = ((data.min() < 0) & (data.max() > 0)).any()
                    area_options = {**kwargs, "stacked": not mixed_signs and kwargs.get("stacked", True)}
                    title = "Stacked Area Plot" if area_options["stacked"] else "Area Plot"
                    with _figure(figsize=(12, 8), constrained_layout=True) as (fig, ax):
                        data.plot.area(ax=ax, title=f"{title} over {column}", **area_options)
                        ax.set_xlabel(column)
                        ax.set_ylabel("Value")
                        finish(fig, "stacked_area")
                    area_pending = False
                    if not lines_enabled:
                        break

    def plot(self, plot_type, x=None, y=None, *, show=True, **kwargs):
        """Draw a scatter, distribution, boxplot, bar, violin, or correlation plot.

        All plots except correlation require a column name ``x``. Scatter also
        requires ``y``. Additional keyword arguments are passed to the selected
        plot method. Correlation accepts ``output_file`` and ``dpi``.
        """
        if plot_type not in ("scatter", "distribution", "boxplot", "bar", "violin", "correlation"):
            raise ValueError(f"Plot type '{plot_type}' is not supported.")
        if plot_type != "correlation":
            self._validate_plot_column("x", x)
        if plot_type == "scatter":
            self._validate_plot_column("y", y)

        with plt.rc_context(self.theme_settings):
            if plot_type == "correlation":
                numeric = self._numeric_plot_data()
                if len(numeric.columns) < 2:
                    return
                fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
                self._draw_correlation_heatmap(numeric, fig, ax)
                if kwargs.get("output_file"):
                    fig.savefig(kwargs["output_file"], dpi=kwargs.get("dpi", 300))
            elif plot_type == "scatter":
                self.data.plot.scatter(x=x, y=y, **kwargs)
            elif plot_type == "bar":
                counts = self.data[x].value_counts()
                if counts.empty:
                    raise ValueError(f"Column '{x}' has no values to plot.")
                counts.plot.bar(**kwargs)
            else:
                series = self.data[x]
                if not pd.api.types.is_numeric_dtype(series.dtype):
                    raise ValueError(f"Column '{x}' must be numeric for a {plot_type} plot.")
                series = series.replace([np.inf, -np.inf], np.nan).dropna()
                if series.empty:
                    raise ValueError(f"Column '{x}' has no finite values to plot.")
                if plot_type == "distribution":
                    series.plot.hist(**kwargs)
                elif plot_type == "boxplot":
                    series.to_frame().boxplot(column=x, **kwargs)
                else:
                    ax = kwargs.pop("ax", None)
                    if ax is None:
                        _, ax = plt.subplots()
                    ax.violinplot(series, **kwargs)
                    ax.set_title(f"Violin Plot of {x}")
            if show:
                plt.show()

    def _validate_plot_column(self, argument, column):
        if column is None:
            raise ValueError(f"Argument '{argument}' is required for this plot type.")
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in dataset.")

    def export_report(self, output_file, format='html'):
        """
        Export a comprehensive report of the analysis and visualizations.

        Parameters:
            output_file (str): Path to the output file.
            format (str): Format of the report ('html' or 'pdf'). Default is 'html'.
        """
        format = format.lower()
        with plt.rc_context(self.theme_settings):
            if format == "html":
                self._export_html(output_file)
            elif format == "pdf":
                self._export_pdf(output_file)
            else:
                raise ValueError("Unsupported format. Use 'html' or 'pdf'.")

    def _export_html(self, output_file):
        """Internal method to export HTML report."""
        analysis = self._generate_analysis()

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PyAutoPlot Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f4f4f9; color: #333; }}
                .container {{ max-width: 1200px; margin: auto; padding: 20px; background: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
                .toc {{ background: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 5px solid #3498db; }}
                .toc ul {{ list-style: none; padding: 0; }}
                .toc li {{ margin-bottom: 10px; }}
                .toc a {{ text-decoration: none; color: #3498db; font-weight: bold; }}
                .section {{ margin-bottom: 50px; }}
                .plot-container {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>PyAutoPlot Analysis Report</h1>

                <div class="toc">
                    <h2>Table of Contents</h2>
                    <ul>
                        <li><a href="#summary">1. Dataset Summary</a></li>
                        <li><a href="#detailed-analysis">2. Statistical Analysis</a></li>
                        <li><a href="#visualizations">3. Visualizations</a></li>
                    </ul>
                </div>

                <div id="summary" class="section">
                    <h2>1. Dataset Summary</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Number of Rows</td><td>{len(self.data)}</td></tr>
                        <tr><td>Number of Columns</td><td>{len(self.data.columns)}</td></tr>
                        <tr><td>Numeric Columns</td><td>{escape(", ".join(map(str, self.numeric)) or "None")}</td></tr>
                        <tr><td>Categorical Columns</td><td>{escape(", ".join(map(str, self.categorical)) or "None")}</td></tr>
                        <tr><td>Time Series Columns</td><td>{escape(", ".join(map(str, self.time_series)) or "None")}</td></tr>
                    </table>
                </div>

                <div id="detailed-analysis" class="section">
                    <h2>2. Statistical Analysis</h2>
        """

        html_parts = [html_content]
        for col, stats in analysis.items():
            html_parts.append(f"<h3>Variable: {escape(str(col))} ({escape(stats['Type'])})</h3><table>")
            for key, value in stats.items():
                if key != "Type":
                    html_parts.append(f"<tr><td>{escape(key)}</td><td>{escape(str(value))}</td></tr>")
            html_parts.append("</table>")

        html_parts.append('</div><div id="visualizations" class="section"><h2>3. Visualizations</h2>')

        def add_plot(title, fig):
            with io.BytesIO() as buffer:
                fig.savefig(buffer, format="png", bbox_inches="tight")
                image = base64.b64encode(buffer.getvalue()).decode("ascii")
            title = escape(title)
            html_parts.append(f'<h3>{title}</h3><div class="plot-container">'
                              f'<img alt="{title}" src="data:image/png;base64,{image}"></div>')

        self._report_visualizations(add_plot)
        html_parts.append("</div></div></body></html>")
        with open(output_file, "w", encoding="utf-8") as report:
            report.writelines(html_parts)

    def _export_pdf(self, output_file):
        """Internal method to export PDF report."""
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output_file) as pdf:
            with self._analysis_figure(self._generate_analysis()) as fig:
                pdf.savefig(fig, bbox_inches="tight")
            self._report_visualizations(lambda title, fig: pdf.savefig(fig, bbox_inches="tight"))

    def _report_visualizations(self, render):
        """Send the same set of figures to the HTML or PDF renderer."""
        numeric = self._numeric_plot_data()
        num_cols = len(numeric.columns)
        if num_cols >= 2:
            with _figure(figsize=(10, 8), constrained_layout=True) as (fig, ax):
                self._draw_correlation_heatmap(numeric, fig, ax)
                render("Correlation Heatmap", fig)

        for column in numeric:
            with _figure(1, 2, figsize=(15, 6), constrained_layout=True) as (fig, axes):
                self._draw_distribution(numeric, column, axes)
                render(f"Distribution and Boxplot of {column}", fig)

        if num_cols:
            with _figure(num_cols, 1, figsize=(10, num_cols * 4), constrained_layout=True,
                         squeeze=False) as (fig, axes):
                self._draw_violin(numeric, axes[:, 0])
                render("Violin Plots", fig)

        for column, counts in self._categorical_counts():
            with _figure(figsize=(10, 6), constrained_layout=True) as (fig, ax):
                self._draw_bar(counts, column, ax)
                render(f"Bar Plot of {column}", fig)

    def export_json(self, output_file):
        """Export statistical analysis and metadata to a JSON file."""
        analysis = self._generate_analysis()

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {str(key): convert_to_serializable(value) for key, value in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_to_serializable(value) for value in obj]
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, (float, np.floating)):
                number = float(obj)
                return number if math.isfinite(number) else None
            return None if pd.isna(obj) else obj

        metadata = {
            "num_rows": int(len(self.data)),
            "num_columns": int(len(self.data.columns)),
            "column_types": {
                "numeric": self.numeric,
                "categorical": self.categorical,
                "time_series": self.time_series
            }
        }

        data_to_export = {
            "metadata": metadata,
            "analysis": convert_to_serializable(analysis)
        }

        with open(output_file, "w", encoding="utf-8") as report:
            json.dump(data_to_export, report, indent=4, allow_nan=False)

    def customize(self, **kwargs):
        """Update global plot defaults and this instance's current theme."""
        plt.rcParams.update(kwargs)
        self.theme_settings.update(kwargs)
