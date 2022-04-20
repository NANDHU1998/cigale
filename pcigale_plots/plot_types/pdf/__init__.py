from itertools import product
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

from pcigale.utils.io import read_table
from pcigale.utils.console import console, INFO, WARNING
from pcigale.utils.counter import Counter
from pcigale_plots.plot_types import Plotter


class PDF(Plotter):
    def __init__(self, config, format, outdir):
        """Plot the PDF of analysed variables."""
        self.configuration = config.configuration
        save_chi2 = self.configuration["analysis_params"]["save_chi2"]

        pdf_vars = []
        if "all" in save_chi2 or "properties" in save_chi2:
            pdf_vars += self.configuration["analysis_params"]["variables"]
        if "all" in save_chi2 or "fluxes" in save_chi2:
            pdf_vars += self.configuration["analysis_params"]["bands"]

        input_data = read_table(outdir.parent / self.configuration["data_file"])
        items = list(product(input_data["id"], pdf_vars, [format], [outdir]))
        counter = Counter(len(items), 1, "PDF")

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
        """Plot the PDF associated with a given analysed variable

        Parameters
        ----------
        obj_name: string
            Name of the object.
        var_name: string
            Name of the analysed variable..
        outdir: Path
            The absolute path to outdir

        """
        gbl_counter.inc()
        var_name = var_name.replace("/", "_")
        fnames = outdir.glob(f"{obj_name}_{var_name}_chi2-block-*.npy")
        likelihood = []
        model_variable = []
        for fname in fnames:
            data = np.memmap(fname, dtype=np.float64)
            data = np.memmap(fname, dtype=np.float64, shape=(2, data.size // 2))

            likelihood.append(np.exp(-data[0, :] / 2.0))
            model_variable.append(data[1, :])
        if len(likelihood) > 0:
            likelihood = np.concatenate(likelihood)
            model_variable = np.concatenate(model_variable)
            w = np.where(np.isfinite(likelihood) & np.isfinite(model_variable))
            likelihood = likelihood[w]
            model_variable = model_variable[w]

            Npdf = 100
            min_hist = np.min(model_variable)
            max_hist = np.max(model_variable)
            Nhist = min(Npdf, len(np.unique(model_variable)))

            if min_hist == max_hist:
                pdf_grid = np.array([min_hist, max_hist])
                pdf_prob = np.array([1.0, 1.0])
            else:
                pdf_prob, pdf_grid = np.histogram(
                    model_variable,
                    Nhist,
                    (min_hist, max_hist),
                    weights=likelihood,
                    density=True,
                )
                pdf_x = (pdf_grid[1:] + pdf_grid[:-1]) / 2.0

                pdf_grid = np.linspace(min_hist, max_hist, Npdf)
                pdf_prob = np.interp(pdf_grid, pdf_x, pdf_prob)

            figure = plt.figure()
            ax = figure.add_subplot(111)
            ax.plot(pdf_grid, pdf_prob, color="k")
            ax.set_xlabel(var_name)
            ax.set_ylabel("Probability density")
            ax.minorticks_on()
            figure.suptitle(
                f"Probability distribution function of {var_name} for "
                f"{obj_name}"
            )
            figure.savefig(outdir / f"{obj_name}_{var_name}_pdf.{format}")
            plt.close(figure)
        else:
            console.print(
                f"{WARNING} Cannot build the PDF of {var_name} for "
                f"{obj_name}."
            )
