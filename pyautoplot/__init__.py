import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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
        self.categorical = []
        self.numeric = []
        self.time_series = []
        self._detect_column_types()

    def _detect_column_types(self):
        """Detect columns as categorical or numeric based on their content."""
        for column in self.data.columns:
            # Check for time series (datetime-like columns)
            if pd.api.types.is_datetime64_any_dtype(self.data[column]):
                self.time_series.append(column)
            # Check for categorical columns (object dtype or less than 15 unique values)
            elif self.data[column].dtype == 'object' or self.data[column].nunique() < 15:
                self.categorical.append(column)
            # Check for numeric columns (numeric dtype)
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                self.numeric.append(column)

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
            skewness = self._calculate_skewness(self, data=data_column, mean=mean, stddev=stddev, n=n)
            kurtosis = self._calculate_kurtosis(self, data=data_column, mean=mean, stddev=stddev, n=n)

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
                "Missing Values": data_column.isnull().sum(),
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
                "Missing Values": data_column.isnull().sum(),
            }
            analysis[column] = stats

        # Time Series Analysis
        for column in self.time_series:
            data_column = self.data[column].dropna()
            stats = {
                "Type": "Time Series",
                "Count": len(data_column),
                "Min": data_column.min(),
                "Max": data_column.max(),
                "Mean": data_column.mean(),
                "Median": data_column.median(),
                "Missing Values": data_column.isnull().sum(),
                "Seasonality": self._detect_seasonality(data_column),
                "Autocorrelation": self._calculate_autocorrelation(data_column),
            }
            analysis[column] = stats

        return analysis

    @staticmethod
    def _calculate_skewness(self, data, mean, stddev, n):
        """Calculate skewness of a dataset."""
        skewness = np.sum(((data - mean) / stddev) ** 3) / n
        return skewness

    @staticmethod
    def _calculate_kurtosis(self, data, mean, stddev, n):
        """Calculate kurtosis of a dataset."""
        kurtosis = (np.sum(((data - mean) / stddev) ** 4) / n) - 3
        return kurtosis

    @staticmethod
    def _calculate_autocorrelation(self, data):
        """Calculate autocorrelation at various lags."""
        mean = data.mean()
        autocorrelations = []
        for lag in range(1, 11):  # Calculate autocorrelation for lags 1 through 10
            lagged_data = data.shift(lag)
            correlation = data.corr(lagged_data)
            autocorrelations.append(correlation)
        return autocorrelations


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
        plt.show()

    def auto_plot(self, output_file=None, theme="light", excludes=None, **kwargs):
        """
        Automatically analyze the dataset and generate visualizations.

        This method produces:
        1. Detailed analysis summary as a text-based plot.
        2. Numeric visualizations: Histograms, Boxplots, and Pairwise Scatter Matrix.
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
            self._plot_detailed_analysis(analysis, output_file=f"{base_filename}_analysis{file_extension}")

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

        # Section 4: Pairwise Scatter Plots for Numeric Variables
        if len(self.numeric) > 1 and "pairwise_scatter" not in excludes:
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

        # Section 7: Stacked Area Plots for Time Series Data
        if self.time_series and "stacked_area" not in excludes:  # If self.time_series contains multiple time series columns for stacking
            fig_area, ax_area = plt.subplots(figsize=(12, 8))
            self.data[self.time_series].plot.area(ax=ax_area, title="Stacked Area Plot of Time Series Data", **kwargs)
            ax_area.set_xlabel("Time")
            ax_area.set_ylabel("Values")

            if output_file:
                # Save Stacked Area section plot
                plt.savefig(f"{base_filename}_stacked_area{file_extension}", dpi=kwargs.get("dpi", 300))

            # Show all generated plots
            plt.show()

    def plot(self, plot_type, x=None, y=None, **kwargs):
        """
        Generate a specific type of plot manually.

        Args:
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
        if plot_type == "scatter" and x and y:
            self.data.plot.scatter(x=x, y=y, **kwargs)
        elif plot_type == "distribution" and x:
            self.data[x].plot(kind="hist", **kwargs)
        elif plot_type == "boxplot" and x:
            self.data.boxplot(column=x, **kwargs)
        elif plot_type == "bar" and x:
            self.data[x].value_counts().plot(kind="bar", **kwargs)
        plt.show()

    def customize(self, **kwargs):
        """Update global plot settings."""
        plt.rcParams.update(kwargs)