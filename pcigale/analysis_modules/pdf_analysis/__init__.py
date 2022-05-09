"""
Probability Density Function analysis module
============================================

This module builds the probability density functions (PDF) of the SED
parameters to compute their moments.

The models corresponding to all possible combinations of parameters are
computed and their fluxes in the same filters as the observations are
integrated. These fluxes are compared to the observed ones to compute the
χ² value of the fitting. This χ² give a probability that is associated with
the model values for the parameters.

At the end, for each parameter, the probability-weighted mean and standard
deviation are computed and the best fitting model (the one with the least
reduced χ²) is given for each observation.

"""

import multiprocessing as mp

import numpy as np

from .. import AnalysisModule
from pcigale.utils.counter import Counter
from .workers import sed as worker_sed
from .workers import init_sed as init_worker_sed
from .workers import init_analysis as init_worker_analysis
from .workers import init_bestfit as init_worker_bestfit
from .workers import analysis as worker_analysis
from .workers import bestfit as worker_bestfit
from ...managers.results import ResultsManager
from ...managers.models import ModelsManager
from ...managers.observations import ObservationsManager
from ...managers.parameters import ParametersManager
from pcigale.utils.console import console, INFO


class PdfAnalysis(AnalysisModule):
    """PDF analysis module"""

    parameter_list = {
        "variables": (
            "cigale_string_list()",
            "List of the physical properties to estimate. Leave empty to "
            "analyse all the physical properties (not recommended when there "
            "are many models).",
            ["sfh.sfr", "sfh.sfr10Myrs", "sfh.sfr100Myrs"]
        ),
        "bands": (
            "cigale_string_list()",
            "List of bands for which to estimate the fluxes. Note that this is "
            "independent from the fluxes actually fitted to estimate the "
            "physical properties.",
            None
        ),
        "save_best_sed": (
            "boolean()",
            "If true, save the best SED for each observation to a file.",
            False
        ),
        "save_chi2": (
            "option('all', 'none', 'properties', 'fluxes')",
            "Save the raw chi2. It occupies ~15 MB/million models/variable. "
            "Allowed values are 'all', 'none', 'properties', and 'fluxes'.",
            "none"
        ),
        "lim_flag": (
            "option('full', 'noscaling', 'none')",
            "Take into account upper limits. If 'full', the exact computation "
            "is done. If 'noscaling', the scaling of the models will not be "
            "adjusted but the χ² will include the upper limits adequately. "
            "Waiving the adjustment makes the fitting much faster compared to "
            "the 'full' option while generally not affecting the results in "
            "any substantial manner. This is the recommended option as it "
            "achieves a good balance between speed and reliability. Finally, "
            "'none' simply discards bands with upper limits.",
            "noscaling"
        ),
        "mock_flag": (
            "boolean()",
            "If true, for each object we create a mock object "
            "and analyse them.",
            False
        ),
        "redshift_decimals": (
            "integer()",
            "When redshifts are not given explicitly in the redshifting "
            "module, number of decimals to round the observed redshifts to "
            "compute the grid of models. To disable rounding give a negative "
            "value. Do not round if you use narrow-band filters.",
            2
        ),
        "blocks": (
            "integer(min=1)",
            "Number of blocks to compute the models and analyse the "
            "observations. If there is enough memory, we strongly recommend "
            "this to be set to 1.",
            1
        )
    }

    def _compute_models(self, conf, obs, params, iblock):
        models = ModelsManager(conf, obs, params, iblock)
        counter = Counter(len(params.blocks[iblock]), 50, "Model")
        initargs = (models, counter)

        self._parallel_job(
            worker_sed,
            params.blocks[iblock],
            initargs,
            init_worker_sed,
            conf["cores"]
        )

        # Print the final value as it may not otherwise be printed
        counter.global_counter.value = len(params.blocks[iblock])
        counter.progress.join()
        console.print(f"{INFO} Done.")

        return models

    def _compute_bayes(self, conf, obs, models):
        results = ResultsManager(models)
        counter = Counter(len(obs), 1, "Object")
        initargs = (models, results, counter)
        self._parallel_job(
            worker_analysis,
            obs,
            initargs,
            init_worker_analysis,
            conf["cores"],
            1
        )
        counter.progress.join()
        console.print(f"{INFO} Done.")

        return results

    def _compute_best(self, conf, obs, params, results):
        counter = Counter(len(obs), 1, "Object")
        initargs = (conf, params, obs, results, counter)
        self._parallel_job(
            worker_bestfit, obs, initargs, init_worker_bestfit, conf["cores"], 1
        )
        counter.progress.join()
        console.print(f"{INFO} Done.")

    def _parallel_job(self, worker, items, initargs, initializer, ncores,
                      chunksize=None):
        if ncores == 1:  # Do not create a new process
            initializer(*initargs)
            for idx, item in enumerate(items):
                worker(idx, item)
        else:  # run in parallel
            # Temporarily remove the counter sub-process that updates the
            # progress bar as it cannot be pickled when creating the parallel
            # processes when using the "spawn" starting method.
            for arg in initargs:
                if isinstance(arg, Counter):
                    counter = arg
                    progress = counter.progress
                    counter.progress = None

            with mp.Pool(
                processes=ncores, initializer=initializer, initargs=initargs
            ) as pool:
                pool.starmap(worker, enumerate(items), chunksize)

            # After the parallel processes have exited, it can be restored
            counter.progress = progress

    def _compute(self, conf, obs, params):
        results = []
        nblocks = len(params.blocks)
        for iblock in range(nblocks):
            console.rule(f"Block {iblock + 1}/{nblocks}")
            # We keep the models if there is only one block. This allows to
            # avoid recomputing the models when we do a mock analysis
            if not hasattr(self, "_models"):
                console.print(f"{INFO} Computing models.")
                models = self._compute_models(conf, obs, params, iblock)
                if nblocks == 1:
                    self._models = models
            else:
                console.print(f"{INFO} Loading precomputed models.")
                models = self._models

            console.print(f"{INFO} Estimating the physical properties.")
            result = self._compute_bayes(conf, obs, models)
            results.append(result)
            console.print(f"{INFO} Block processed.")

        console.rule("Global analysis")
        console.print(f"{INFO} Estimating the physical properties.")
        results = ResultsManager.merge(results)

        console.print(f"{INFO} Computing the best fit spectra.")
        self._compute_best(conf, obs, params, results)

        return results

    def process(self, conf):
        """Process with the psum analysis.

        The analysis is done in two steps which can both run on multiple
        processors to run faster. The first step is to compute all the fluxes
        associated with each model as well as ancillary data such as the SED
        information. The second step is to carry out the analysis of each
        object, considering all models at once.

        Parameters
        ----------
        conf: dictionary
            Contents of pcigale.ini in the form of a dictionary

        """
        np.seterr(invalid="ignore")

        console.print(f"{INFO} Initialising the analysis module.")

        # Rename the output directory if it exists
        self.prepare_dirs()

        # Store the grid of parameters in a manager to facilitate the
        # computation of the models
        params = ParametersManager(conf)

        # Store the observations in a manager which sanitises the data, checks
        # all the required fluxes are present, adding errors if needed,
        # discarding invalid fluxes, etc.
        obs = ObservationsManager(conf, params)
        obs.save("observations")

        results = self._compute(conf, obs, params)
        console.print(f"{INFO} Sanity check of the analysis results.")
        results.best.analyse_chi2()

        console.print(f"{INFO} Saving the analysis results.")
        results.save("results")

        if conf["analysis_params"]["mock_flag"] is True:
            console.print(f"{INFO} Analysing the mock observations.")

            # For the mock analysis we do not save the ancillary files.
            for k in ["best_sed", "chi2"]:
                conf["analysis_params"][f"save_{k}"] = False

            # We replace the observations with a mock catalogue..
            obs.generate_mock(results)
            obs.save("mock_observations")

            results = self._compute(conf, obs, params)

            console.print(f"{INFO} Saving the mock analysis results.")
            results.save("results_mock")

        console.print(f"{INFO} Run completed! :thumbs_up:")


# AnalysisModule to be returned by get_module
Module = PdfAnalysis
