from collections.abc import Iterable
import itertools
import numpy as np

from pcigale.utils.io import read_table


class ParametersManager:
    """Class to abstract the call to the relevant parameters manager depending
    how the physical parameters of the models are provided (directly in the
    pcigale.ini file ).

    A ParametersManager allows to generate a list containing the parameters for
    a modules whose index is passed as an argument. It also allows to know what
    modules have changed their parameters given their indices. Because the
    order of the modules is optimised to minimise the computations, this allows
    to keep the cache of partially computed models in SedWarehouse as small as
    possible by weeding out partial models that will not be used anymore."""

    def __new__(object, conf):
        if conf['parameters_file']:
            return ParametersManagerFile(conf)
        else:
            return ParametersManagerGrid(conf)


class ParametersManagerGrid:
    """Class to generate a parameters manager for a systematic grid using the
    parameters given in the pcigale.ini file."""

    def __init__(self, conf):
        """Instantiate the class.

        Parameters
        ----------
        conf: dictionary
            Contains the modules in the order they are called

        """
        self.modules = conf['sed_modules']
        self.parameters = [self._param_dict_combine(conf['sed_modules_params'][module])
                           for module in self.modules]
        self.shape = tuple(len(parameter) for parameter in self.parameters)
        self.size = int(np.product(self.shape))
        if 'blocks' not in conf['analysis_params']:
            conf['analysis_params']['blocks'] = 1
        self.blocks = self._split(range(self.size),
                                  conf['analysis_params']['blocks'],
                                  len(conf['sed_modules_params']['redshifting']['redshift']))

    def __len__(self):
        return self.size

    @staticmethod
    def _split(l, nb, nz):
        """Split a list l into nb blocks with blocks of such size that all the
        redshifts of a given model as in the same block.

        Parameters
        ----------
        l: list
            List to split.
        nb: int
            Number of blocks.
        nz: int
            Number of redshifts.

        """
        k = len(l) // nb
        step = k + nz - k % nz

        return [l[i * step: (i + 1) * step] for i in range(nb)]

    def _param_dict_combine(self, dictionary):
        """Given a dictionary associating to each key an array, returns all the
        possible dictionaries associating a single element to each key.

        Parameters
        ----------
        dictionary: dict
            Dictionary associating an array to its (or some of its) keys.

        Returns
        -------
        combination_list: list of dictionaries
            List of dictionaries with the same keys but associating one element
            to each.

        """
        # We make a copy of the dictionary as we are modifying it.
        dictionary = dict(dictionary)

        # First, we must ensure that all values are lists; when a value is a
        # single element, we put it in a list.
        # We must take a special care of strings, because they are iterable.

        for key, value in dictionary.items():
            if (not isinstance(value, Iterable)) or isinstance(value, str):
                dictionary[key] = [value]

        # We use itertools.product to make all the possible combinations from
        # the value lists.
        key_list = dictionary.keys()
        value_array_list = [dictionary[key] for key in key_list]
        combination_list = [dict(zip(key_list, combination))
                            for combination in
                            itertools.product(*value_array_list)]

        return combination_list

    def from_index(self, index):
        """Provides the parameters of a model given a 1D index.

        Parameters
        ----------
        index: int
            1D index of the model for which we want the parameters

        Returns
        -------
        params: list
            Parameters of the model corresponding to the index

        """
        # Our problem is isomorph to the conversion between a linear and an nD
        # index of an nD array. Thankfully numpy's unravel_index does the
        # conversion from a 1D index to nD indices.
        indices = np.unravel_index(index, self.shape)
        params = [self.parameters[module][param_idx]
                  for module, param_idx in enumerate(indices)]

        return params


class ParametersManagerFile:
    """Class to generate a parameters manager for list of parameters given in an
    input file."""

    def __init__(self, conf):
        """Instantiate the class.

        Parameters
        ----------
        conf: dictionary
            Contains the name of the file containing the parameters

        """
        table = read_table(conf['parameters_file'])

        self.size = len(table)
        self.modules = conf['sed_modules']
        self.blocks = self._split(range(self.size),
                                  conf['analysis_params']['blocks'])

        # The parameters file is read using astropy.Table. Unfortunately, it
        # stores strings as np.str_, which is not marshalable, which means we
        # cannot use the list of parameter to build a key for the cache. This
        # overcome this, rather than using the table directly, we split it into
        # several dictionaries, one for each module. Each dictionary contains
        # the arguments in the form of lists that we convert to the right type:
        # float for numbers and str for strings.
        self.parameters = []
        for module in self.modules:
            dict_params = {}
            for colname in table.colnames:
                if colname.startswith(module):
                    parname = colname.split('.', 1)[1]
                    if type(table[colname][0]) is np.str_:
                        dict_params[parname] = [str(val) for val in
                                                table[colname]]
                    else:
                        dict_params[parname] = list(table[colname])
            self.parameters.append(dict_params)

        del table

    @staticmethod
    def _split(l, nb):
        """Split a list l into nb blocks.

        Parameters
        ----------
        l: list
            List to split.
        nb: int
            Number of blocks.

        """
        step = len(l) // nb
        if step > 0:
            return [l[i * step: (i + 1) * step] for i in range(nb)]

        raise ValueError("The number of blocks must be no more than the number"
                         "of models.")

    def from_index(self, index):
        """Provides the parameters of a model given an index.

        Parameters
        ----------
        index: int
            index of the model for which we want the parameters

        Returns
        -------
        params: list
            Parameters of the model corresponding to the index

        """

        # As we have a simple file, this corresponds to the line number
        params = [{name: self.parameters[idx_module][name][index] for name in
                   self.parameters[idx_module]} for idx_module, module
                  in enumerate(self.modules)]

        return params
