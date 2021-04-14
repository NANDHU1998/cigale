# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2014 Institute of Astronomy, University of Cambridge
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien


class NebularLines_AGN(object):
    """Nebular lines templates.

    This class holds the data associated with the line templates

    """

    def __init__(self, region, metallicity, logU, name, wave, ratio):
        """Create a new nebular lines template

        Parameters
        ----------
        region: string
            NLR or BLR region
        metallicity: float
            Gas phase metallicity
        logU: float
            Ionisation parameter
        name: array
            Name of each line
        wave: array
            Vector of the λ grid used in the templates [nm]
        ratio: array
            Line intensities relative to Hβ

        """

        self.region = region
        self.metallicity = metallicity
        self.logU = logU
        self.name = name
        self.wave = wave
        self.ratio = ratio
