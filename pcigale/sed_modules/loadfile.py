"""
Read spectrum from file module
==============================

This module reads a SED spectrum from a file.

"""

from pcigale.utils.io import read_table
from . import SedModule


class LoadSpecFile(SedModule):
    """Module reading a spectrum from a file and adding it to the SED.

    """

    parameter_list = {
        "filename": (
            'string()',
            "Name of the file to load and to add to the SED table. This "
            "file must be loadable with astropy",
            None
        ),
        "lambda_column": (
            'string()',
            "Name of the column containing the wavelength in nm.",
            None
        ),
        "l_lambda_column": (
            'string()',
            "Name of the column containing the Lλ luminosity in W/nm.",
            None
        )
    }

    def process(self, sed):
        """Add the spectrum from the file to the SED object

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        filename = self.parameters['filename']
        table = read_table(filename)

        sed.add_module(self.name, self.parameters)

        sed.add_contribution(
            filename,
            table[self.parameters['lambda_column']],
            table[self.parameters['l_lambda_column']]
        )


# SedModule to be returned by get_module
Module = LoadSpecFile
