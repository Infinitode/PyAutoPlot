import json
import unittest
import warnings
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from pyautoplot import AutoPlot


class AutoPlotTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.addCleanup(plt.close, "all")

    def read_analysis(self, plotter):
        path = self.output / "analysis.json"
        plotter.export_json(path)

        def reject_nonfinite(value):
            self.fail(f"Non-standard JSON number: {value}")

        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)

    def test_sparse_mixed_format_dates_and_autocorrelation(self):
        csv = ("when,value\n2024-01-01,1\n,2\n,3\n2024-01-02T00:00:00,4\n"
               ",5\n2024-01-03,6\n,7\n2024-01-04,8\n,9\n")
        plotter = AutoPlot(StringIO(csv))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            report = self.read_analysis(plotter)
        self.assertEqual(report["metadata"]["column_types"]["time_series"], ["when"])
        stats = report["analysis"]["when"]
        self.assertEqual(stats["Count"], 4)
        self.assertEqual(stats["Missing Values"], 5)
        self.assertEqual(stats["Min"], "2024-01-01T00:00:00")
        self.assertEqual(stats["Median"], "2024-01-02T12:00:00")
        self.assertAlmostEqual(stats["Autocorrelation"][0], 1.0)
        self.assertAlmostEqual(stats["Autocorrelation"][1], 1.0)
        self.assertEqual(stats["Autocorrelation"][2:], [None] * 8)

    def test_mixed_offsets_and_constant_timestamps(self):
        plotter = AutoPlot(StringIO("when\n2024-01-01T11:00:00+01:00\n"
                                   "2024-01-01T12:00:00+02:00\n2024-01-01T10:00:00Z\n"))
        stats = self.read_analysis(plotter)["analysis"]["when"]
        self.assertEqual(stats["Count"], 3)
        self.assertEqual(stats["Min"], "2024-01-01T10:00:00+00:00")
        self.assertEqual(stats["Autocorrelation"], [None] * 10)

    def test_text_columns_are_not_destroyed_by_date_detection(self):
        rows = [f"group-{i:02d},08:00,01/02,{date}" for i, date in enumerate(
            ["2024-01-01"] * 19 + ["2024-02-30"]
        )]
        plotter = AutoPlot(StringIO("group,time,code,date\n" + "\n".join(rows)))
        self.assertEqual(plotter.time_series, [])
        self.assertEqual(plotter.categorical, ["group", "time", "code", "date"])
        self.assertEqual(plotter.data["date"].iloc[-1], "2024-02-30")
        self.assertEqual(plotter.data["time"].iloc[0], "08:00")

    def test_header_only_csv_generates_reports_but_no_empty_charts(self):
        path = self.output / "empty.csv"
        path.write_text("category,measurement\n", encoding="utf-8")
        plotter = AutoPlot(path)
        report = self.read_analysis(plotter)
        self.assertEqual(report["metadata"]["num_rows"], 0)
        for stats in report["analysis"].values():
            self.assertEqual(stats["Count"], 0)
            self.assertIsNone(stats["Most Common"])
            self.assertEqual(stats["Frequency Distribution"], {})

        plotter.auto_plot(self.output / "empty", show=False, dpi=25)
        self.assertEqual([p.name for p in self.output.glob("*.png")], ["empty_analysis.png"])
        html = self.output / "empty.html"
        pdf = self.output / "empty.pdf"
        plotter.export_report(html)
        plotter.export_report(pdf, format="pdf")
        self.assertIn("Variable: category", html.read_text(encoding="utf-8"))
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))

    def test_json_exports_bool_and_handles_nonfinite_numbers(self):
        rows = [f"{n},True" for n in range(12)] + ["inf,False", "-inf,False", ","]
        plotter = AutoPlot(StringIO("number,flag\n" + "\n".join(rows)))
        analysis = self.read_analysis(plotter)["analysis"]
        self.assertEqual(analysis["number"]["Count"], 12)
        self.assertEqual(analysis["number"]["Max"], 11)
        self.assertEqual(analysis["number"]["Missing Values"], 3)
        self.assertIs(analysis["flag"]["Most Common"], True)
        self.assertEqual(analysis["flag"]["Most Common Count"], 12)

    def test_cleanup_validation_and_empty_timestamps_after_dropping_rows(self):
        plotter = AutoPlot(StringIO("when,value,empty\n2024-01-01,1,\n2024-01-02,,\n2024-01-03,5,\n"))
        with self.assertRaisesRegex(ValueError, "Unknown cleanup strategy"):
            plotter.cleanup(strategy="unknown")
        with self.assertRaisesRegex(ValueError, "fill_value"):
            plotter.cleanup(strategy="constant")
        with self.assertRaisesRegex(ValueError, "not found"):
            plotter.cleanup(columns=["invalid"])
        plotter.cleanup(strategy="median", columns="value")
        self.assertEqual(plotter.data["value"].tolist(), [1, 3, 5])
        self.assertEqual(self.read_analysis(plotter)["analysis"]["empty"]["Missing Values"], 3)

        plotter.cleanup(drop_na=True, columns=["empty"])
        stats = self.read_analysis(plotter)["analysis"]["when"]
        self.assertEqual(stats["Count"], 0)
        self.assertIsNone(stats["Min"])
        self.assertEqual(stats["Autocorrelation"], [None] * 10)

    def test_auto_plot_consumes_dpi_and_bins_and_scopes_the_theme(self):
        csv = "value,other,group\n" + "\n".join(f"{n},{2*n},a" for n in range(12))
        user_figure = plt.figure()
        saved = {}
        savefig = Figure.savefig

        def record_save(fig, path, **kwargs):
            axis = fig.axes[0]
            saved[Path(path).name] = (kwargs["dpi"], axis.get_facecolor(), len(axis.patches))
            return savefig(fig, path, **kwargs)

        with plt.rc_context({"axes.facecolor": "#eeffee"}):
            plotter = AutoPlot(StringIO(csv))
            self.assertEqual(plt.rcParams["axes.facecolor"], "#eeffee")
            with patch.object(Figure, "savefig", record_save), patch.object(plt, "show") as show:
                plotter.auto_plot(self.output / "charts", theme={"axes.facecolor": "#ddeeff"},
                                  show=False, dpi=40, bins=4, color="orange")
                show.assert_not_called()
            self.assertEqual(plt.rcParams["axes.facecolor"], "#eeffee")

        self.assertEqual(set(saved), {f"charts_{name}.png" for name in (
            "analysis", "numeric", "categorical", "pairwise", "pie_group", "violin", "correlation"
        )})
        self.assertEqual(saved["charts_numeric.png"], (40, to_rgba("#ddeeff"), 4))
        self.assertEqual(saved["charts_pairwise.png"][2], 4)
        self.assertTrue(all(dpi == 40 for dpi, _, _ in saved.values()))
        self.assertEqual(plt.get_fignums(), [user_figure.number])

    def test_time_plots_use_sorted_timestamps_and_handle_mixed_signs(self):
        plotter = AutoPlot(StringIO("when,measure\n2024-01-03,3\n2024-01-01,-4\n,10\n2024-01-02,2\n"))
        saved = {}
        savefig = Figure.savefig

        def record_save(fig, path, **kwargs):
            saved[Path(path).name] = fig.axes[0]
            return savefig(fig, path, **kwargs)

        with patch.object(Figure, "savefig", record_save):
            plotter.auto_plot(self.output / "timeline.png", dpi=30, show=False,
                              excludes=["detailed_analysis", "categorical", "pie_charts"])
        line = saved["timeline_line_when.png"]
        self.assertEqual(line.get_xlabel(), "when")
        self.assertEqual(line.lines[0].get_ydata().tolist(), [-4, 2, 3])
        self.assertEqual(saved["timeline_stacked_area.png"].get_title(), "Area Plot over when")

    def test_column_names_are_safe_and_unique_in_output_paths(self):
        plotter = AutoPlot(StringIO("a/b,a\\b\nx,y\ny,z\n"))
        plotter.auto_plot(self.output / "out.png", show=False, dpi=30,
                          excludes=["detailed_analysis", "categorical"])
        self.assertEqual({path.name for path in self.output.iterdir()}, {"out_pie_a_b.png", "out_pie_a_b_2.png"})

    def test_reports_escape_csv_text_and_keep_external_figures_open(self):
        plotter = AutoPlot(StringIO("<group & name>,when\nÉté < 30 & warm,2024-01-01\n"
                                   "Cool & cloudy,2024-01-02\n"))
        user_figure = plt.figure()
        html = self.output / "report.html"
        pdf = self.output / "report.pdf"
        plotter.export_report(html)
        plotter.export_report(pdf, format="PDF")

        text = html.read_text(encoding="utf-8")
        self.assertIn("&lt;group &amp; name&gt;", text)
        self.assertIn("Été &lt; 30 &amp; warm", text)
        self.assertNotIn("<group & name>", text)
        self.assertIn('src="data:image/png;base64,', text)
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
        self.assertEqual(plt.get_fignums(), [user_figure.number])

    def test_failed_save_does_not_leak_figures_or_theme(self):
        plotter = AutoPlot(StringIO("group\na\nb\n"))
        user_figure = plt.figure()
        original = plt.rcParams["axes.facecolor"]
        with self.assertRaises(FileNotFoundError):
            plotter.auto_plot(self.output / "missing" / "charts", theme="dark", show=False, dpi=25)
        self.assertEqual(plt.rcParams["axes.facecolor"], original)
        self.assertEqual(plt.get_fignums(), [user_figure.number])

    def test_manual_plot_validation_and_violin_options(self):
        plotter = AutoPlot(StringIO("name,value\na,1\nb,2\nc,3\n"))
        with self.assertRaisesRegex(ValueError, "not supported"):
            plotter.plot("unknown", show=False)
        with self.assertRaisesRegex(ValueError, "'y' is required"):
            plotter.plot("scatter", x="value", show=False)
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            plotter.plot("distribution", x="name", show=False)
        _, ax = plt.subplots()
        plotter.plot("violin", x="value", ax=ax, showmeans=True, show=False)
        self.assertEqual(ax.get_title(), "Violin Plot of value")
        self.assertEqual(len(ax.collections), 5)


if __name__ == "__main__":
    unittest.main()
