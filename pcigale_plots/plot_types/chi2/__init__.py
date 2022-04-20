from itertools import product
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

from pcigale.utils.io import read_table
from pcigale.utils.console import console, INFO
from pcigale.utils.counter import Counter
from pcigale_plots.plot_types import Plotter


class Chi2(Plotter):
    def __init__(self, config, format, outdir):
        """Plot the χ² values of analysed variables."""
        self.configuration = config.configuration
        file = outdir.parent / self.configuration["data_file"]
        input_data = read_table(file)
        save_chi2 = self.configuration["analysis_params"]["save_chi2"]

        chi2_vars = []
        if "all" in save_chi2 or "properties" in save_chi2:
            chi2_vars += self.configuration["analysis_params"]["variables"]
        if "all" in save_chi2 or "fluxes" in save_chi2:
            chi2_vars += self.configuration["analysis_params"]["bands"]

        items = list(product(input_data["id"], chi2_vars, [format], [outdir]))
        counter = Counter(len(items), 1, "Item")

        self._parallel_job(items, counter)

        # Print the final value as it may not otherwise be printed
        counter.global_counter.value = len(items)
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
    def worker(obj_name, var_name, format, outdir):
        """Plot the reduced χ² associated with a given analysed variable

        Parameters
        ----------
        obj_name: string
            Name of the object.
        var_name: string
            Name of the analysed variable..
        outdir: Path
            Path to outdir

        """
        gbl_counter.inc()
        figure = plt.figure()
        ax = figure.add_subplot(111)

        var_name = var_name.replace("/", "_")
        fnames = outdir.glob(f"{obj_name}_{var_name}_chi2-block-*.npy")
        for fname in fnames:
            data = np.memmap(fname, dtype=np.float64)
            data = np.memmap(fname, dtype=np.float64, shape=(2, data.size // 2))
            ax.scatter(data[1, :], data[0, :], color="k", s=0.1)
        ax.set_xlabel(var_name)
        ax.set_ylabel(r"Reduced $\chi^2$")
        ax.set_ylim(
            0.0,
        )
        ax.minorticks_on()
        figure.suptitle(
            f"Reduced $\chi^2$ distribution of {var_name} for " f"{obj_name}."
        )
        figure.savefig(outdir / f"{obj_name}_{var_name}_chi2.{format}")
        plt.close(figure)
