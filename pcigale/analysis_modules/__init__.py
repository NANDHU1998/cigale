# -*- coding: utf-8 -*-
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille, AMU
# Copyright (C) 2012, 2014 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly & Denis Burgarella

from datetime import datetime
from importlib import import_module
import os
import shutil


class AnalysisModule(object):
    """Abstract class, the pCigale analysis modules are based on.
    """

    # parameter_list is a dictionary containing all the parameters
    # used by the module. Each parameter name is associate to a tuple
    # (variable type, description [string], default value). Each module must
    # define its parameter list, unless it does not use any parameter. Using
    # None means that there is no description, unit or default value. If None
    # should be the default value, use the 'None' string instead.
    parameter_list = dict()

    def __init__(self, **kwargs):
        """Instantiate a analysis module

        The module parameters values can be passed as keyword parameters.
        """
        # parameters is a dictionary containing the actual values for each
        # module parameter.
        self.parameters = kwargs

    def _process(self, configuration):
        """Do the actual analysis

        This method is responsible for the fitting / analysis process
        and must be implemented by each real module.

        Parameters
        ----------
        configuration: dictionary
            Configuration file

        Returns
        -------
        The process results are saved to disk by the analysis module.

        """
        raise NotImplementedError()

    def prepare_dirs(self):
        # Create a new out/ directory and move existing one if needed
        if os.path.exists('out/'):
            name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_out/'
            os.rename('out/', name)
            print(f"The out/ directory was renamed to {name}")

        os.mkdir('out/')
        shutil.copy('pcigale.ini', 'out/')
        shutil.copy('pcigale.ini.spec', 'out/')

    def process(self, configuration):
        """Process with the analysis

        This method is responsible for checking the module parameters before
        doing the actual processing (_process method). If a parameter is not
        given but exists in the parameter_list with a default value, this
        value is used.

        Parameters
        ----------
        configuration: dictionary
            Contents of pcigale.ini in the form of a dictionary

        Returns
        -------
        The process results are saved to disk by the analysis module

        Raises
        ------
        KeyError: when not all the needed parameters are given.

        """
        parameters = configuration['analysis_params']
        # For parameters that are present on the parameter_list with a default
        # value and that are not in the parameters dictionary, we add them
        # with their default value.
        for key in self.parameter_list:
            if (key not in parameters) and (
                    self.parameter_list[key][2] is not None):
                parameters[key] = self.parameter_list[key][2]

        # If the keys of the parameters dictionary are different from the one
        # of the parameter_list dictionary, we raises a KeyError. That means
        # that a parameter is missing (and has no default value) or that an
        # unexpected one was given.
        if not set(parameters) == set(self.parameter_list):
            missing_parameters = (set(self.parameter_list) - set(parameters))
            unexpected_parameters = (set(parameters) -
                                     set(self.parameter_list))
            message = ""
            if missing_parameters:
                message += ("Missing parameters: " +
                            ", ".join(missing_parameters) +
                            ".")
            if unexpected_parameters:
                message += ("Unexpected parameters: " +
                            ", ".join(unexpected_parameters) +
                            ".")
            raise KeyError("The parameters passed are different from the "
                           "expected one." + message)

        # We do the actual processing
        self._process(configuration)


def get_module(module_name):
    """Return the main class of the module provided

    Parameters
    ----------
    module_name: string
        The name of the module we want to get the class.

    Returns
    -------
    module_class: class
    """

    try:
        module = import_module('.' + module_name, 'pcigale.analysis_modules')
        return module.Module()
    except ImportError:
        print('Module ' + module_name + ' does not exists!')
        raise
