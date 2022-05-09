from astropy.table import Table
import matplotlib
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pkg_resources
from scipy import stats

from pcigale.utils.console import console, INFO
from pcigale.utils.counter import Counter
from pcigale_plots.plot_types import Plotter

# Name of the file containing the best models information
BEST_RESULTS = "results.fits"
MOCK_RESULTS = "results_mock.fits"


class Mock(Plotter):
    def __init__(self, config, nologo, outdir):
        """Plot the comparison of input/output values of analysed variables."""
        best_results_file = outdir / BEST_RESULTS
        mock_results_file = outdir / MOCK_RESULTS

        try:
            exact = Table.read(best_results_file)
        except FileNotFoundError:
            raise Exception(f"Best models file {best_results_file} not found.")

        try:
            estimated = Table.read(mock_results_file)
        except FileNotFoundError:
            raise Exception(f"Mock models file {mock_results_file} not found.")

        self.configuration = config.configuration
        params = self.configuration["analysis_params"]["variables"]

        for param in params:
            if param.endswith("_log"):
                param = f"best.{param}"
                exact[param] = np.log10(exact[param[:-4]])

        logo = (
            False
            if nologo
            else plt.imread(
                pkg_resources.resource_filename(
                    __name__, "../../resources/CIGALE.png"
                )
            )
        )

        arguments = [
            (
                exact[f"best.{param}"],
                estimated[f"bayes.{param}"],
                param,
                logo,
                outdir,
            )
            for param in params
        ]

        counter = Counter(len(arguments), 1, "Parameter")

        self._parallel_job(arguments, counter)

        # Print the final value as it may not otherwise be printed
        counter.global_counter.value = len(arguments)
        counter.progress.join()
        console.print(f"{INFO} Done.")

    @staticmethod
    def initializer(counter):
        """Initializer of the pool of processes to share variables between workers.
        Parameters
        ----------
        :param counter: Counter class object for the number of models plotted
        """
        global gbl_counter

        gbl_counter = counter

    @staticmethod
    def worker(exact, estimated, param, logo, outdir):
        """Plot the exact and estimated values of a parameter for the mock analysis

        Parameters
        ----------
        exact: Table column
            Exact values of the parameter.
        estimated: Table column
            Estimated values of the parameter.
        param: string
            Name of the parameter
        nologo: boolean
            Do not add the logo when set to true.
        outdir: Path
            Path to outdir

        """
        gbl_counter.inc()
        range_exact = np.linspace(np.min(exact), np.max(exact), 100)

        # We compute the linear regression
        if np.min(exact) < np.max(exact):
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                exact, estimated
            )
        else:
            slope = 0.0
            intercept = 1.0
            r_value = 0.0
        figure = plt.figure()
        plt.errorbar(
            exact,
            estimated,
            marker=".",
            label=param,
            color="k",
            linestyle="None",
            capsize=0.0,
        )
        plt.plot(range_exact, range_exact, color="r", label="1-to-1")
        plt.plot(
            range_exact,
            slope * range_exact + intercept,
            color="b",
            label=f"exact-fit $r^2$ = {r_value**2:.2f}",
        )
        plt.xlabel("Exact")
        plt.ylabel("Estimated")
        plt.title(param)
        plt.legend(loc="best", fancybox=True, framealpha=0.5, numpoints=1)
        plt.minorticks_on()

        if logo is not False:
            figure.figimage(logo, 0, 0, origin="upper", zorder=0, alpha=1)

        plt.tight_layout()
        plt.savefig(outdir / f"mock_{param}.pdf", dpi=figure.dpi * 2.0)

        plt.close()
