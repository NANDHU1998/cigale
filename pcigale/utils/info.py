from collections import Counter
from copy import deepcopy
from importlib import import_module
import multiprocessing as mp
import pkgutil
from platform import python_version
from rich.panel import Panel
from sysconfig import get_platform

import numpy as np
from rich import box
from rich.table import Table

import pcigale.sed_modules
from pcigale.utils.console import console
from pcigale.utils.io import read_table


class Info:
    def __init__(self, config):
        """Utility class to gather and display information about CIGALE and
        general information about the run as described in pcigale.ini.

        Parameters
        ----------
        config: dict
           Dictionary containing the configuration given in pcigale.ini.

        """
        # Make sure we do not alter the config dictionary
        self.config = deepcopy(config)

        self.sedmodules = {}
        for module in pkgutil.iter_modules(pcigale.sed_modules.__path__):
            try:
                imported = import_module(f"pcigale.sed_modules.{module.name}")
                if hasattr(imported, "__category__"):
                    category = imported.__category__
                    if category not in self.sedmodules:
                        self.sedmodules[category] = [module.name]
                    else:
                        self.sedmodules[category].append(module.name)
            except Exception:
                continue

    def print_tables(self):
        """Print the summary CIGALE info table and the SED modules table."""

        console.print("\n")
        self.print_summaryinfo_table()

        console.print("\n")
        self.print_sedmodules_table()

    @staticmethod
    def print_panel():
        """Print the general information panel about CIGALE along with essential
        information such as the python version and the platform.
        """

        url = "https://cigale.lam.fr"
        console.print(
            Panel(
                "[bold]Code Investigating GALaxy Emission[/bold]\n"
                f"Boquien et al. (2019) ([link={url}]{url}[/link])\n"
                f"CIGALE version: [data]{pcigale.__version__}[/data] "
                f"— Python version: [data]{python_version()}[/data] "
                f"— Platform: [data]{get_platform()}[/data]",
                box=box.ROUNDED,
                style="panel",
                highlight=False,
            ),
            justify="center",
        )

    def print_summaryinfo_table(self):
        """Print the summary information table about the run. It contains data
        extracted from pcigale.ini
        """

        if len(self.config["data_file"]) == 0:
            fname_data = "[warning]None[/warning]"
            ntargets = "[warning]None[/warning]"
            z = self.config["sed_modules_params"]["redshifting"]["redshift"]
            if len(z) == 0:
                z = "Not indicated"
            else:
                nz = len(z)
                zmin = np.nanmin(z)
                zmax = np.nanmax(z)
                if zmin != zmax:
                    z = f"{zmin:.2f}—{zmax:.2f} ({nz} redshifts)"
                else:
                    z = f"{zmin}"
        else:
            fname_data = self.config["data_file"]
            data = read_table(self.config["data_file"])
            ntargets = f"{len(data)}"
            zmin = np.min(data["redshift"])
            zmax = np.max(data["redshift"])
            if zmin >= 0.0:
                z = f"{zmin:.2f} to {zmax:.2f}"
            else:
                z = "Photometric"

        table = Table(
            title="General information",
            style="panel",
            show_header=False,
            box=box.ROUNDED,
        )
        table.add_column(justify="left", min_width=None)
        table.add_column(min_width=None, style="data")
        table.add_row("Data file", fname_data)
        table.add_row("Parameters file", self._param_file())
        table.add_row("Number of objects", ntargets)
        table.add_row("Redshift", z)
        table.add_row("Bands fitted", self._bands())
        table.add_row("Properties fitted", self._props())
        table.add_row("Number of models", self._nmodels())
        table.add_row("Cores used", self._cores())
        table.add_row("Analysis module", self.config["analysis_method"])
        console.print(table, justify="center")

    def print_sedmodules_table(self):
        """Print the list of SED modules used in the run. When a component has
        no associated module a list of modules is provided.
        """

        table = Table(
            title="SED modules",
            style="panel",
            show_header=False,
            box=box.ROUNDED,
        )
        categories = [
            "SFH",
            "SSP",
            "nebular",
            "dust attenuation",
            "dust emission",
            "AGN",
            "X-ray",
            "radio",
            "restframe_parameters",
            "redshifting",
        ]
        sed_modules = set(self.config["sed_modules"])
        table.add_column(justify="left", min_width=None)
        table.add_column(min_width=None, style="data")

        for category in categories:
            module = set(self.sedmodules[category]) & sed_modules
            if len(module) == 0:
                module = (
                    "[warning]None[/warning]. Options are: "
                    f"[data]{', '.join(self.sedmodules[category])}.[/data]"
                )
            else:
                module = f"{module.pop()}"
            table.add_row(f"{category}", module)

        console.print(table, justify="center")

    def _param_file(self):
        """Return the formatted name of the parameters file for printing."""

        if len(self.config["parameters_file"]) == 0:
            return "[warning]None[/warning]"
        return self.config["parameters_files"]

    def _bands(self):
        """Return the formatted list of bands to fit for printing."""

        bands = [
            band.split(".")[0]
            for band in self.config["bands"]
            if band.endswith("_err") is False
        ]
        bands = [f"{item} ({count})" for item, count in Counter(bands).items()]
        if len(bands) > 0:
            bands = " — ".join(bands)
        else:
            bands = "[warning]None[/warning]"

        return bands

    def _props(self):
        """Return the formatted list of properties to fit for printing."""

        props = [
            prop
            for prop in self.config["properties"]
            if prop.endswith("_err") is False
        ]

        if len(props) == 0:
            return "[warning]None[/warning]"
        return " — ".join(props)

    def _nmodels(self):
        """Return the formatted number of models for printing."""

        nmodels = 1
        for module in self.config["sed_modules_params"].values():
            for parameter in module:
                if isinstance(module[parameter], list):
                    nparams = len(module[parameter])
                    nmodels *= nparams

        nz = len(self.config["sed_modules_params"]["redshifting"]["redshift"])
        if nz > 0:
            nmodels = f"{nmodels} ({nmodels // nz} per redshift)"
        else:
            nmodels = f"{nmodels}"

        return nmodels

    def _cores(self):
        """Return the formatted number of cores used for printing."""

        cores = f"{self.config['cores']}/{mp.cpu_count()}"
        if self.config["cores"] > mp.cpu_count():
            cores = f"[warning][b]{cores}[/b][/warning] (risk of slow down)"

        return cores
