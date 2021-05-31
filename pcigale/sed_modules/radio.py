"""
Radio module
=============================

This module implements the synchrotron emission of galaxies assuming the
FIR/radio correlation and the power law of the synchrotron spectrum.

"""

import numpy as np
import scipy.constants as cst

from . import SedModule


class Radio(SedModule):
    """Radio emission

    This module computes the synchrotron (non-thermal) emission of galaxies.

    """

    parameter_list = {
        "qir_sf": (
            "cigale_list(minvalue=0.)",
            "The value of the FIR/radio correlation coefficient for star "
            "formation.",
            2.58
        ),
        "alpha_sf": (
            "cigale_list()",
            "The slope of the power-law synchrotron emission related to star "
            "formation, Lν∝ν^-α.",
            0.8
        ),
        "R_agn": (
            "cigale_list(minvalue=0.)",
            "The radio-loudness parameter for AGN, defined as "
            "R=Lν_5GHz/Lν_2500A, where Lν_2500A is the AGN 2500 Å intrinsic "
            "disk luminosity measured at viewing angle=30°.",
            0.
        ),
        "alpha_agn": (
            "cigale_list()",
            "The slope of the power-law AGN radio emission (assumed "
            "isotropic), Lν∝ν^-α.",
            0.7
        )
    }

    def _init_code(self):
        """Build the model for a given set of parameters."""

        self.qir_sf = float(self.parameters["qir_sf"])
        self.alpha_sf = float(self.parameters["alpha_sf"])
        self.R_agn = float(self.parameters["R_agn"])
        self.alpha_agn = float(self.parameters["alpha_agn"])

        # We define various constants necessary to compute the model. For
        # consistency, we define the speed of light in nm s¯¹ rather than in
        # m s¯¹.
        c = cst.c * 1e9
        # We define the wavelength range for the non thermal emission
        self.wave = np.logspace(5., 9., 1000)

        # We compute the SF synchrotron emission normalised at 21cm
        self.lumin_nonthermal_sf = (1. / self.wave)**(-self.alpha_sf + 2.) / \
                                   (1. / 2.1e8)**(-self.alpha_sf + 2.)

        # Normalisation factor from the FIR/radio correlation to apply to the
        # IR luminosity
        S21cm = (1. / (10.**self.qir_sf * 3.75e12)) * (c / 2.1e8**2.)
        self.lumin_nonthermal_sf *= S21cm

        # We compute the AGN emission normalized at 5 GHz
        self.lumin_agn = (1. / self.wave)**(-self.alpha_agn + 2.) / \
                         (5e9 / c)**(-self.alpha_agn + 2.)

        # Normalisation factor from the 2500 A-5 GHz relation to apply to the
        # AGN 2500 A Lnu
        S5GHz = self.R_agn * 5e9**2. / c
        self.lumin_agn *= S5GHz

    def process(self, sed):
        """Add the radio contribution.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        if 'dust.luminosity' not in sed.info:
            sed.add_info('dust.luminosity', 1., True, unit='W')
        luminosity = sed.info['dust.luminosity']

        if 'agn.intrin_Lnu_2500A_30deg' not in sed.info:
            sed.add_info('agn.intrin_Lnu_2500A_30deg', 1., True, unit='W/Hz')
        Lnu_2500A = sed.info['agn.intrin_Lnu_2500A_30deg']

        sed.add_module(self.name, self.parameters)
        sed.add_info("radio.qir_sf", self.qir_sf)
        sed.add_info("radio.alpha_sf", self.alpha_sf)
        sed.add_contribution('radio.sf_nonthermal', self.wave,
                             self.lumin_nonthermal_sf * luminosity)
        sed.add_info("radio.R_agn", self.R_agn)
        sed.add_info("radio.alpha_agn", self.alpha_agn)
        sed.add_contribution('radio.agn', self.wave,
                             self.lumin_agn * Lnu_2500A)


# SedModule to be returned by get_module
Module = Radio
