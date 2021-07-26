"""
Save fluxes analysis module
===========================

This module does not perform a statistical analysis. It computes and save the
fluxes in a set of filters for all the possible combinations of input SED
parameters.

The data file is used only to get the list of fluxes to be computed.

"""

import multiprocessing as mp

from .. import AnalysisModule
from utils.counter import Counter
from .workers import init_fluxes as init_worker_fluxes
from .workers import fluxes as worker_fluxes
from ...managers.models import ModelsManager
from ...managers.observations import ObservationsManager
from ...managers.parameters import ParametersManager
from pcigale.utils.console import console, INFO

class SaveFluxes(AnalysisModule):
    """Save fluxes analysis module

    This module saves a table containing all the parameters and desired fluxes
    for all the computed models.

    """

    parameter_list = {
        "variables": (
            "cigale_string_list()",
            "List of the physical properties to save. Leave empty to save all "
            "the physical properties (not recommended when there are many "
            "models).",
            None
        ),
        "save_sed": (
            "boolean()",
            "If True, save the generated spectrum for each model.",
            False
        ),
        "blocks": (
            "integer(min=1)",
            "Number of blocks to compute the models. Having a number of blocks"
            " larger than 1 can be useful when computing a very large number "
            "of models or to split the result file into smaller files.",
            1
        )
    }

    @staticmethod
    def _parallel_job(worker, items, initargs, initializer, ncores):
        if ncores == 1:  # Do not create a new process
            initializer(*initargs)
            for idx, item in enumerate(items):
                worker(idx, item)
        else:  # run in parallel
            with mp.Pool(processes=ncores, initializer=initializer,
                         initargs=initargs) as pool:
                pool.starmap(worker, enumerate(items))


    def _compute_models(self, conf, obs, params):
        nblocks = len(params.blocks)
        for iblock in range(nblocks):
            console.rule(f"Block {iblock + 1}/{nblocks}")
            console.print(f"{INFO} Starting the computation of the models.")

            models = ModelsManager(conf, obs, params, iblock)
            counter = Counter(len(params.blocks[iblock]), 50, "Model")

            initargs = (models, counter)
            self._parallel_job(worker_fluxes, params.blocks[iblock], initargs,
                               init_worker_fluxes, conf['cores'])

            # Print the final value as it may not otherwise be printed
            counter.global_counter.value = len(params.blocks[iblock])
            counter.progress.join()
            console.print(f"{INFO} Done.")
            console.print(f"{INFO} Saving the models.")
            models.save(f"models-block-{iblock}")

    def process(self, conf):
        """Process with the savedfluxes analysis.

        All the possible theoretical SED are created and the fluxes in the
        filters from the list of bands are computed and saved to a table,
        alongside the parameter values.

        Parameters
        ----------
        conf: dictionary
            Contents of pcigale.ini in the form of a dictionary
        """

        # Rename the output directory if it exists
        self.prepare_dirs()

        # Read the observations information in order to retrieve the list of
        # bands to compute the fluxes.
        observations = ObservationsManager(conf)

        # The parameters manager allows us to retrieve the models parameters
        # from a 1D index. This is useful in that we do not have to create
        # a list of parameters as they are computed on-the-fly. It also has
        # nice goodies such as finding the index of the first parameter to
        # have changed between two indices or the number of models.
        params = ParametersManager(conf)

        self._compute_models(conf, observations, params)

        console.print(f"{INFO} Run completed! :thumbs_up:")



# AnalysisModule to be returned by get_module
Module = SaveFluxes
