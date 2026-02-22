import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings
import json
import base64
import io
# from statsmodels.tsa.stattools import acf # Removed as _detect_seasonality is removed

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
        """
        Initializes the AutoPlot object by loading the dataset and classifying columns.

        Args:
            csv_path (str): Path to the CSV file to load and analyze.
        """
        self.data = pd.read_csv(csv_path)
        self._try_parse_dates()
        self.categorical = []
        self.numeric = []
        self.time_series = []
        self._detect_column_types()
        self._apply_theme("light") # Initialize default theme

    def _try_parse_dates(self):
        """Attempt to parse object columns as dates."""
        for col in self.data.select_dtypes(include=['object']):
            # Basic check to see if it's likely a date column
            # We avoid purely numeric strings that might be interpreted as dates
            if self.data[col].astype(str).str.contains(r'[-/:]').any():
                try:
                    converted = pd.to_datetime(self.data[col], errors='coerce')
                    # If more than 50% parsed successfully and it's not all NaT
                    if converted.notna().sum() > len(self.data) * 0.5:
                        self.data[col] = converted
                except (ValueError, TypeError):
                    pass

    def _detect_column_types(self):
        """Detect columns as categorical, numeric, or time-series based on their content."""
        self.categorical = []
        self.numeric = []
        self.time_series = []

        for column in self.data.columns:
            # Check for time series (datetime-like columns)
            if pd.api.types.is_datetime64_any_dtype(self.data[column]):
                self.time_series.append(column)
            # Check for numeric columns (numeric dtype)
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                # If it has very few unique values, it might be better as categorical (e.g., flags 0/1)
                # but usually numeric is fine. Let's stick to the rule but prioritize numeric.
                if self.data[column].nunique() < 10:
                     self.categorical.append(column)
                else:
                     self.numeric.append(column)
            # Check for categorical columns (object dtype)
            elif self.data[column].dtype == 'object' or self.data[column].nunique() < 15:
                self.categorical.append(column)

        # Check for unclassified columns
        for column in self.data.columns:
            if column not in self.numeric and column not in self.categorical and column not in self.time_series:
                warnings.warn(f"Warning: Column '{column}' was not classified and will be ignored.")

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

        if drop_na:
            self.data.dropna(subset=columns, inplace=True)
        else:
            for col in columns:
                if self.data[col].isnull().any():
                    if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.data[col]):
                        self.data[col] = self.data[col].fillna(self.data[col].mean())
                    elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.data[col]):
                        self.data[col] = self.data[col].fillna(self.data[col].median())
                    elif strategy == 'mode':
                        mode_vals = self.data[col].mode()
                        if not mode_vals.empty:
                            self.data[col] = self.data[col].fillna(mode_vals[0])
                    elif strategy == 'constant':
                        self.data[col] = self.data[col].fillna(fill_value)

        # Re-detect column types after cleanup
        self._detect_column_types()

    def _apply_theme(self, theme):
        """Apply theme settings for light, dark, or custom themes."""
        if theme == "dark":
            self.theme_settings = {
                "axes.facecolor": "#222222",
                "axes.edgecolor": "#ffffff",
                "axes.labelcolor": "#ffffff",
                "figure.facecolor": "#222222",
                "grid.color": "#444444",
                "text.color": "#ffffff",
                "xtick.color": "#ffffff",
                "ytick.color": "#ffffff",
                "legend.frameon": False
            }
            plt.rcParams.update(self.theme_settings)

        elif theme == "light":
            self.theme_settings = {
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
            plt.rcParams.update(self.theme_settings)

        elif isinstance(theme, dict):
            # Check if the theme is a dictionary (custom theme)
            try:
                self.theme_settings = theme
                plt.rcParams.update(self.theme_settings)
            except Exception as e:
                print(f"Error applying custom theme: {e}")
                # Optionally, revert to default or light theme if custom fails
                self._apply_theme("light")

        else:
            print("Invalid theme specified. Falling back to default (light) theme.")
            self._apply_theme("light")

    def _generate_analysis(self):
        """Generate detailed analysis of the dataset."""
        analysis = {}

        # Numeric Analysis
        for column in self.numeric:
            data_column = self.data[column].dropna()  # Drop NaN values for the analysis
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
                "Missing Values": self.data[column].isnull().sum(),
            }
            analysis[column] = stats

        # Categorical Analysis
        for column in self.categorical:
            value_counts = self.data[column].value_counts()
            stats = {
                "Type": "Categorical",
                "Count": self.data[column].count(),
                "Unique": self.data[column].nunique(),
                "Most Common": value_counts.idxmax(),
                "Most Common Count": value_counts.max(),
                "Top 3 Most Common": value_counts.head(3).to_dict(),
                "Balance Ratio": self._calculate_balance_ratio(self.data[column]),
                "Frequency Distribution": value_counts.to_dict(),
                "Mode Variability": value_counts[value_counts == value_counts.max()].index.tolist(),
                "Missing Values": self.data[column].isnull().sum(),
            }
            analysis[column] = stats

        # Time Series Analysis
        for column in self.time_series:
            data_column = self.data[column].dropna()

            # seasonality_result logic removed
            autocorrelation_result = [np.nan] * 10 # Default for autocorrelation

            if not data_column.empty:
                # seasonality_result = self._detect_seasonality(data_column) # Removed
                autocorrelation_result = self._calculate_autocorrelation(data_column)

            stats = {
                "Type": "Time Series",
                "Count": len(data_column), # This will be 0 if empty
                "Min": data_column.min() if not data_column.empty else np.nan,
                "Max": data_column.max() if not data_column.empty else np.nan,
                "Mean": data_column.mean() if not data_column.empty else np.nan,
                "Median": data_column.median() if not data_column.empty else np.nan,
                "Missing Values": self.data[column].isnull().sum(),
                # "Seasonality": seasonality_result, # Removed
                "Autocorrelation": autocorrelation_result,
            }
            analysis[column] = stats

        return analysis

    @staticmethod
    def _calculate_skewness(data, mean, stddev, n):
        """Calculate skewness of a dataset."""
        if stddev == 0 or n == 0:
            return np.nan
        skewness = np.sum(((data - mean) / stddev) ** 3) / n
        return skewness

    @staticmethod
    def _calculate_kurtosis(data, mean, stddev, n):
        """Calculate kurtosis of a dataset."""
        if stddev == 0 or n == 0:
            return np.nan
        kurtosis = (np.sum(((data - mean) / stddev) ** 4) / n) - 3
        return kurtosis

    @staticmethod
    def _calculate_autocorrelation(data):
        """Calculate autocorrelation at various lags."""
        if len(data) < 2 or data.var() == 0:
            return [np.nan] * 10  # Return list of NaNs for 10 lags
        mean = data.mean()
        autocorrelations = []
        for lag in range(1, 11):  # Calculate autocorrelation for lags 1 through 10
            lagged_data = data.shift(lag)
            correlation = data.corr(lagged_data)
            autocorrelations.append(correlation)
        return autocorrelations

    # _detect_seasonality method removed

    @staticmethod
    def _detect_outliers(series, threshold=3):
        """Detect outliers in a numeric series based on standard deviation."""
        mean = series.mean()
        std_dev = series.std()
        outliers = series[(series < mean - threshold * std_dev) | (series > mean + threshold * std_dev)]
        return outliers.tolist()

    @staticmethod
    def _calculate_balance_ratio(series):
        """Calculate balance ratio for a categorical series."""
        value_counts = series.value_counts()
        most_common = value_counts.max()
        least_common = value_counts.min()
        return f"{most_common}:{least_common} (Most:Least)"

    def _plot_correlation_heatmap(self, output_file=None, **kwargs):
        """Generate a correlation heatmap for numeric variables."""
        if len(self.numeric) < 2:
            return

        corr = self.data[self.numeric].corr()
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

        # Add colorbar
        fig.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(self.numeric)))
        ax.set_yticks(np.arange(len(self.numeric)))
        ax.set_xticklabels(self.numeric, rotation=45, ha="right")
        ax.set_yticklabels(self.numeric)

        # Add text annotations
        for i in range(len(self.numeric)):
            for j in range(len(self.numeric)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black")

        ax.set_title("Correlation Heatmap", fontsize=16, weight="bold")

        if output_file:
            plt.savefig(output_file, dpi=kwargs.get("dpi", 300))

    def _plot_violin(self, output_file=None, **kwargs):
        """Generate violin plots for numeric variables."""
        if not self.numeric:
            return

        num_cols = len(self.numeric)
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, num_cols * 4), constrained_layout=True)
        axes = axes if num_cols > 1 else [axes]

        for i, col in enumerate(self.numeric):
            data_to_plot = self.data[col].dropna()
            if not data_to_plot.empty:
                axes[i].violinplot(data_to_plot)
                axes[i].set_title(f"Violin Plot of {col}", weight="bold")
                axes[i].set_ylabel("Value")
                axes[i].set_xticks([]) # Remove x ticks for single violin

        if output_file:
            plt.savefig(output_file, dpi=kwargs.get("dpi", 300))

    def _plot_detailed_analysis(self, analysis, output_file=None):
        """Create a summary plot showing the detailed analysis for all variables."""
        total_vars = len(analysis)
        fig, axes = plt.subplots(total_vars, 1, figsize=(20, total_vars * 4), constrained_layout=True)
        axes = axes if total_vars > 1 else [axes]  # Handle a single variable case

        fig.suptitle("Detailed Dataset Analysis", fontsize=20, weight="bold", y=1.02)

        for i, (column, stats) in enumerate(analysis.items()):
            ax = axes[i]
            stat_text = "\n".join(f"{key}: {value}" for key, value in stats.items())

            # Set up bbox with increased padding for larger boxes
            bbox_props = dict(facecolor=self.theme_settings['axes.facecolor'], edgecolor=self.theme_settings['axes.edgecolor'], alpha=0.8)

            # Adjust text positioning to allow more room
            ax.text(0.5, 0.5, stat_text, fontsize=12, verticalalignment="center", horizontalalignment="center", transform=ax.transAxes, bbox=bbox_props)

            ax.axis("off")  # Hide axes since this is a text-only section
            ax.set_title(f"Analysis of {column}", fontsize=16, pad=20, weight="bold")  # Increased padding to make space for the title

        if output_file:
            plt.savefig(output_file, dpi=300)
        # plt.show() will be called from auto_plot

    def auto_plot(self, output_file=None, theme="light", excludes=None, **kwargs):
        """
        Automatically analyze the dataset and generate visualizations.

        This method produces:
        1. Detailed analysis summary as a text-based plot.
        2. Numeric visualizations: Histograms, Boxplots, and Pairwise Scatter Matrix. Note: Pairwise Scatter Matrix can be resource-intensive for datasets with many numeric columns. Consider using the `excludes=['pairwise_scatter']` option for large datasets.
        3. Categorical visualizations: Enhanced Bar Plots and Pie Charts.
        4. Time-series visualizations: Line and Stacked Area Plots.

        Parameters:
            output_file (str, optional): Base filename for saving plots. Default is None (no saving).
            theme (str, optional): Plot theme ("light", "dark", or a custom dictionary). Default is "light".
            excludes (list, optional): Sections to exclude from plotting, e.g., ["histograms"]. Default is None.
            **kwargs: Additional keyword arguments passed to plot methods.

        Examples:
            autoplot.auto_plot(output_file="dataset_output", theme="dark", excludes=["pie_charts"])
        """

        # Initialize the excludes parameter if not provided
        if excludes is None:
            excludes = []

        self._apply_theme(theme)

        # Handle the base filename and extension
        if output_file:
            base_filename, file_extension = os.path.splitext(output_file)

        # Section 1: Detailed Analysis Summary
        if "detailed_analysis" not in excludes:
            analysis = self._generate_analysis()
            self._plot_detailed_analysis(analysis, output_file=f"{base_filename}_analysis{file_extension}" if output_file else None)
            plt.show()
            plt.close('all')

        # Section 2: Numeric Distributions and Boxplots
        if self.numeric and "numeric" not in excludes:
            fig, axes = plt.subplots(len(self.numeric), 2, figsize=(18, len(self.numeric) * 5), constrained_layout=True)
            axes = axes if len(self.numeric) > 1 else [axes]  # Handle single numeric column

            fig.suptitle("Numeric Variables: Distributions and Boxplots", fontsize=20, weight="bold", y=1.02)

            for i, column in enumerate(self.numeric):
                # Histogram
                self.data[column].plot(kind='hist', ax=axes[i][0], title=f"Distribution of {column}", **kwargs)
                axes[i][0].set_xlabel(column)
                axes[i][0].set_ylabel("Frequency")

                # Boxplot
                self.data.boxplot(column=column, ax=axes[i][1], **kwargs)
                axes[i][1].set_title(f"Boxplot of {column}")

            if output_file:
                # Save Numeric section plots
                plt.savefig(f"{base_filename}_numeric{file_extension}", dpi=kwargs.get("dpi", 300))
            plt.show()
            plt.close('all')

        # Section: Correlation Heatmap
        if "correlation" not in excludes:
            self._plot_correlation_heatmap(output_file=f"{base_filename}_correlation{file_extension}" if output_file else None, **kwargs)
            plt.show()
            plt.close('all')

        # Section: Violin Plots
        if "violin" not in excludes:
            self._plot_violin(output_file=f"{base_filename}_violin{file_extension}" if output_file else None, **kwargs)
            plt.show()
            plt.close('all')

        # Section 3: Enhanced Bar Plots for Categorical Variables
        if self.categorical and "categorical" not in excludes:
            fig_cat, ax_cat = plt.subplots(len(self.categorical), 1, figsize=(18, len(self.categorical) * 5), constrained_layout=True)
            ax_cat = ax_cat if len(self.categorical) > 1 else [ax_cat]

            fig_cat.suptitle("Categorical Variables: Enhanced Bar Plots", fontsize=20, weight="bold", y=1.02)

            for j, cat_column in enumerate(self.categorical):
                value_counts = self.data[cat_column].value_counts()
                max_value = value_counts.max()
                min_value = value_counts.min()
                avg_value = value_counts.mean()

                value_counts.plot(kind="bar", ax=ax_cat[j], title=f"Bar Plot of {cat_column}", **kwargs)
                ax_cat[j].set_xlabel(cat_column)
                ax_cat[j].set_ylabel("Count")

                # Annotate bar plot with statistics
                ax_cat[j].text(
                    0.02, 0.95,
                    f"Max: {max_value}\nMin: {min_value}\nAvg: {avg_value:.2f}",
                    transform=ax_cat[j].transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=dict(facecolor=self.theme_settings['axes.facecolor'], edgecolor=self.theme_settings['axes.edgecolor'], alpha=0.8)
                )
            if output_file:
                # Save Categorical section plots
                plt.savefig(f"{base_filename}_categorical{file_extension}", dpi=kwargs.get("dpi", 300))
            plt.show()
            plt.close('all')

        # Section 4: Pairwise Scatter Plots for Numeric Variables
        if len(self.numeric) > 1 and "pairwise_scatter" not in excludes:
            if len(self.numeric) > 10:
                warnings.warn(f"Large number of numeric columns ({len(self.numeric)}) detected. Pairwise scatter plots may be slow.")

            fig_scatter, axes_scatter = plt.subplots(len(self.numeric), len(self.numeric), figsize=(20, 20), constrained_layout=True)

            fig_scatter.suptitle("Pairwise Scatter Plots of Numeric Variables", fontsize=20, weight="bold", y=1.02)

            for i, col_x in enumerate(self.numeric):
                for j, col_y in enumerate(self.numeric):
                    if i == j:
                        # Diagonal: Histogram for the variable
                        self.data[col_x].plot(kind="hist", ax=axes_scatter[i][j], title=f"{col_x} Distribution", **kwargs)
                        axes_scatter[i][j].set_xlabel(col_x)
                    else:
                        # Off-diagonal: Scatter plot
                        self.data.plot.scatter(x=col_x, y=col_y, ax=axes_scatter[i][j], **kwargs)
                        axes_scatter[i][j].set_title(f"{col_x} vs {col_y}")

            if output_file:
                # Save Pairwise section plots
                plt.savefig(f"{base_filename}_pairwise{file_extension}", dpi=kwargs.get("dpi", 300))
            plt.show()
            plt.close('all')

        # Section 5: Pie Charts for Categorical Variables
        if self.categorical and "pie_charts" not in excludes:
            for cat_column in self.categorical:
                value_counts = self.data[cat_column].value_counts()
                fig_pie, ax_pie = plt.subplots(figsize=(8, 8))

                value_counts.plot(kind="pie", ax=ax_pie, autopct='%1.1f%%', title=f"Pie Chart of {cat_column}", **kwargs)
                ax_pie.set_ylabel("")  # Remove the default ylabel (it can clutter the plot)

                if output_file:
                    # Save Pie section plots
                    plt.savefig(f"{base_filename}_pie_{cat_column}{file_extension}", dpi=kwargs.get("dpi", 300))
                plt.show()
                plt.close('all')

        # Section 6: Line Plots for Time Series Data
        if self.time_series and "line_plots" not in excludes:  # Assuming self.time_series contains time series columns
            for time_col in self.time_series:
                fig_line, ax_line = plt.subplots(figsize=(10, 6))
                self.data[time_col].plot(kind="line", ax=ax_line, title=f"Line Plot of {time_col}", **kwargs)
                ax_line.set_xlabel("Time")
                ax_line.set_ylabel(time_col)

                if output_file:
                    # Save Line plot section
                    plt.savefig(f"{base_filename}_line_{time_col}{file_extension}", dpi=kwargs.get("dpi", 300))
                plt.show()
                plt.close('all')

        # Section 7: Stacked Area Plots for Time Series Data
        if self.time_series and "stacked_area" not in excludes:  # If self.time_series contains multiple time series columns for stacking
            fig_area, ax_area = plt.subplots(figsize=(12, 8))
            self.data[self.time_series].plot.area(ax=ax_area, title="Stacked Area Plot of Time Series Data", **kwargs)
            ax_area.set_xlabel("Time")
            ax_area.set_ylabel("Values")

            if output_file:
                # Save Stacked Area section plot
                plt.savefig(f"{base_filename}_stacked_area{file_extension}", dpi=kwargs.get("dpi", 300))
            plt.show()
            plt.close('all')

        # No final plt.show() here anymore

    def plot(self, plot_type, x=None, y=None, **kwargs):
        """
        Generate a specific type of plot manually.

        Parameters:
            plot_type (str): Type of plot to generate. Options include:
                - "scatter": Scatter plot (requires x and y).
                - "distribution": Histogram (requires x).
                - "boxplot": Boxplot (requires x).
                - "bar": Bar chart (requires x).
            x (str, optional): Name of the column for the x-axis or primary data. Default is None.
            y (str, optional): Name of the column for the y-axis (for scatter plots). Default is None.
            **kwargs: Additional keyword arguments for the plot.

        Examples:
            autoplot.plot(plot_type="scatter", x="age", y="income")
            autoplot.plot(plot_type="distribution", x="salary", bins=20)
        """
        if plot_type == "scatter":
            if x is None:
                raise ValueError("Argument 'x' is required for scatter plot.")
            if y is None:
                raise ValueError("Argument 'y' is required for scatter plot.")
            if x not in self.data.columns:
                raise ValueError(f"Column '{x}' not found in dataset.")
            if y not in self.data.columns:
                raise ValueError(f"Column '{y}' not found in dataset.")
            self.data.plot.scatter(x=x, y=y, **kwargs)
        elif plot_type in ["distribution", "boxplot", "bar", "violin"]:
            if x is None:
                raise ValueError("Argument 'x' is required for this plot type.")
            if x not in self.data.columns:
                raise ValueError(f"Column '{x}' not found in dataset.")

            if plot_type == "distribution":
                self.data[x].plot(kind="hist", **kwargs)
            elif plot_type == "boxplot":
                self.data.boxplot(column=x, **kwargs)
            elif plot_type == "bar":
                self.data[x].value_counts().plot(kind="bar", **kwargs)
            elif plot_type == "violin":
                plt.violinplot(self.data[x].dropna())
                plt.title(f"Violin Plot of {x}")
        elif plot_type == "correlation":
            self._plot_correlation_heatmap(**kwargs)
        else:
            # Handling other plot types or raising an error for unsupported ones
            # If x is generally required for other plots, this check can be more generic.
            if x is None: # A more generic check if x is usually required
                 raise ValueError("Argument 'x' is required for this plot type.")
            # If x is provided, but plot_type is unknown
            if x is not None and x not in self.data.columns:
                 raise ValueError(f"Column '{x}' not found in dataset.")
            # Defaulting to a simple plot or error if plot_type is not recognized
            else:
                raise ValueError(f"Plot type '{plot_type}' is not supported or invalid arguments provided.")

        plt.show()

    def export_report(self, output_file, format='html'):
        """
        Export a comprehensive report of the analysis and visualizations.

        Parameters:
            output_file (str): Path to the output file.
            format (str): Format of the report ('html' or 'pdf'). Default is 'html'.
        """
        if format.lower() == 'html':
            self._export_html(output_file)
        elif format.lower() == 'pdf':
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
                        <tr><td>Numeric Columns</td><td>{", ".join(self.numeric) if self.numeric else "None"}</td></tr>
                        <tr><td>Categorical Columns</td><td>{", ".join(self.categorical) if self.categorical else "None"}</td></tr>
                        <tr><td>Time Series Columns</td><td>{", ".join(self.time_series) if self.time_series else "None"}</td></tr>
                    </table>
                </div>

                <div id="detailed-analysis" class="section">
                    <h2>2. Statistical Analysis</h2>
        """

        for col, stats in analysis.items():
            html_content += f"<h3>Variable: {col} ({stats['Type']})</h3><table>"
            for k, v in stats.items():
                if k != "Type":
                    html_content += f"<tr><td>{k}</td><td>{v}</td></tr>"
            html_content += "</table>"

        html_content += '<div id="visualizations" class="section"><h2>3. Visualizations</h2>'

        def get_plot_base64():
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close('all')
            return img_str

        # Correlation Heatmap
        if len(self.numeric) >= 2:
            self._plot_correlation_heatmap()
            html_content += f'<h3>3.1 Correlation Heatmap</h3><div class="plot-container"><img src="data:image/png;base64,{get_plot_base64()}"></div>'

        # Numeric plots
        if self.numeric:
            html_content += '<h3>3.2 Numeric Distributions and Boxplots</h3>'
            for column in self.numeric:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
                self.data[column].plot(kind='hist', ax=axes[0], title=f"Distribution of {column}")
                self.data.boxplot(column=column, ax=axes[1])
                html_content += f'<div class="plot-container"><img src="data:image/png;base64,{get_plot_base64()}"></div>'

            html_content += '<h3>3.3 Violin Plots</h3>'
            self._plot_violin()
            html_content += f'<div class="plot-container"><img src="data:image/png;base64,{get_plot_base64()}"></div>'

        # Categorical plots
        if self.categorical:
            html_content += '<h3>3.4 Categorical Variable Bar Plots</h3>'
            for cat_column in self.categorical:
                fig, ax = plt.subplots(figsize=(10, 6))
                self.data[cat_column].value_counts().plot(kind="bar", ax=ax, title=f"Bar Plot of {cat_column}")
                html_content += f'<div class="plot-container"><img src="data:image/png;base64,{get_plot_base64()}"></div>'

        html_content += "</div></div></body></html>"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _export_pdf(self, output_file):
        """Internal method to export PDF report."""
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output_file) as pdf:
            # 1. Detailed Analysis
            analysis = self._generate_analysis()
            self._plot_detailed_analysis(analysis)
            pdf.savefig(bbox_inches='tight')
            plt.close('all')

            # 2. Correlation
            if len(self.numeric) >= 2:
                self._plot_correlation_heatmap()
                pdf.savefig(bbox_inches='tight')
                plt.close('all')

            # 3. Numeric
            if self.numeric:
                for column in self.numeric:
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
                    self.data[column].plot(kind='hist', ax=axes[0], title=f"Distribution of {column}")
                    self.data.boxplot(column=column, ax=axes[1])
                    pdf.savefig(bbox_inches='tight')
                    plt.close('all')

                self._plot_violin()
                pdf.savefig(bbox_inches='tight')
                plt.close('all')

            # 4. Categorical
            if self.categorical:
                for cat_column in self.categorical:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    self.data[cat_column].value_counts().plot(kind="bar", ax=ax, title=f"Bar Plot of {cat_column}")
                    pdf.savefig(bbox_inches='tight')
                    plt.close('all')

    def export_json(self, output_file):
        """Export statistical analysis and metadata to a JSON file."""
        analysis = self._generate_analysis()

        # Convert any non-serializable objects (like numpy types or pandas Series/DataFrames)
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                if np.isnan(obj):
                    return None
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj]
            if isinstance(obj, dict):
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            if pd.isna(obj):
                return None
            return obj

        serializable_analysis = convert_to_serializable(analysis)

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
            "analysis": serializable_analysis
        }

        with open(output_file, 'w') as f:
            json.dump(data_to_export, f, indent=4)

    def customize(self, **kwargs):
        """Update global plot settings."""
        plt.rcParams.update(kwargs)