"""
Bruzual and Charlot (2003) stellar emission module for an SSP
=============================================================

This module implements the Bruzual and Charlot (2003) Single Stellar
Populations.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule
from ..data import Database


class YggdrasilSSP(SedModule):

    parameter_list = OrderedDict([
        ("metallicity", (
            "cigale_list(options=0.004 & 0.008 & 0.02)",
            "Metalicity. Possible values are: 0.004, 0.008, and 0.02.",
            0.02
        )),
        ("separation_age", (
            "cigale_list(dtype=int, minvalue=0)",
            "Age [Myr] of the separation between the young and the old star "
            "populations. The default value in 10^7 years (10 Myr). Set "
            "to 0 not to differentiate ages (only an old population).",
            10
        ))
    ])

    def _init_code(self):
        """Read the SSP from the database."""
        self.metallicity = float(self.parameters["metallicity"])
        self.separation_age = int(self.parameters["separation_age"])

        with Database() as database:
            self.ssp = database.get_yggdrasil_ssp(self.metallicity)

    def process(self, sed):
        """Add the convolution of a Bruzual and Charlot SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """
        if 'ssp.index' in sed.info:
            index = sed.info['ssp.index']
        else:
            raise Exception('The stellar models do not correspond to pure SSP.')

        if self.ssp.time_grid[index] <= self.separation_age:
            spec_young = self.ssp.spec_table[:, index]
            info_young = self.ssp.info_table[index]
            spec_old = np.zeros_like(spec_young)
            info_old = np.zeros_like(info_young)
        else:
            spec_old = self.ssp.spec_table[:, index]
            info_old = self.ssp.info_table[index]
            spec_young = np.zeros_like(spec_old)
            info_young = np.zeros_like(info_old)
        info_all = info_young + info_old

        wave = self.ssp.wavelength_grid
        lum_young, lum_old = np.trapz([spec_young, spec_old], wave)

        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.metallicity", self.metallicity)
        sed.add_info("stellar.old_young_separation_age", self.separation_age)
        sed.add_info("stellar.age", self.ssp.time_grid[index])

        sed.add_info("stellar.m_star_young", info_young, True)
        sed.add_info("stellar.lum_young", lum_young, True)

        sed.add_info("stellar.m_star_old", info_old, True)
        sed.add_info("stellar.lum_old", lum_old, True)

        sed.add_info("stellar.m_star", info_all, True)
        sed.add_info("stellar.lum", lum_young + lum_old, True)

        sed.add_contribution("stellar.old", wave, spec_old)
        sed.add_contribution("stellar.young", wave, spec_young)


# SedModule to be returned by get_module
Module = YggdrasilSSP
