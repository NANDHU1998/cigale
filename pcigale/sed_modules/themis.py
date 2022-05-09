# DustPedia,  http://www.dustpedia.com/
# Author: Angelos Nersesian, Frederic Galliano, Anthony Jones, Pieter De Vis

"""
Jones et al (2017) IR models module
=====================================

This module implements the Jones et al (2017) infrared models.

"""

import numpy as np

from pcigale.data import SimpleDatabase as Database
from . import SedModule

__category__ = "dust emission"


class THEMIS(SedModule):
    """Jones et al (2017) templates IR re-emission module

    Given an amount of attenuation (e.g. resulting from the action of a dust
    attenuation module) this module normalises the Jones et al (2017)
    model corresponding to a given set of parameters to this amount of energy
    and add it to the SED.

    Information added to the SED: NAME_alpha.

    """

    parameter_list = {
        'qhac': (
            'cigale_list(options=0.02 & 0.06 & 0.10 & 0.14 & 0.17 & 0.20 & '
            '0.24 & 0.28 & 0.32 & 0.36 & 0.40)',
            "Mass fraction of hydrocarbon solids i.e., a-C(:H) smaller than "
            "1.5 nm, also known as HAC. Possible values are: 0.02, 0.06, "
            "0.10, 0.14, 0.17, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40.",
            0.17
        ),
        'umin': (
            'cigale_list(options=0.10 & 0.12 & 0.15 & 0.17 & 0.20 & 0.25 & '
            '0.30 & 0.35 & 0.40 & 0.50 & 0.60 & 0.70 & 0.80 & 1.00 & 1.20 & '
            '1.50 & 1.70 & 2.00 & 2.50 & 3.00 & 3.50 & 4.00 & 5.00 & 6.00 & '
            '7.00 & 8.00 & 10.00 & 12.00 & 15.00 & 17.00 & 20.00 & 25.00 & '
            '30.00 & 35.00 & 40.00 & 50.00 & 80.00)',
            "Minimum radiation field. Possible values are: 0.100, 0.120, "
            "0.150, 0.170, 0.200, 0.250, 0.300, 0.350, 0.400, 0.500, 0.600, "
            "0.700, 0.800, 1.000, 1.200, 1.500, 1.700, 2.000, 2.500, 3.000, "
            "3.500, 4.000, 5.000, 6.000, 7.000, 8.000, 10.00, 12.00, 15.00, "
            "17.00, 20.00, 25.00, 30.00, 35.00, 40.00, 50.00, 80.00.",
            1.0
        ),
        'alpha': (
            'cigale_list(options=1.0 & 1.1 & 1.2 & 1.3 & 1.4 & 1.5 & 1.6 & '
            '1.7 & 1.8 & 1.9 & 2.0 & 2.1 & 2.2 & 2.3 & 2.4 & 2.5 & 2.6 & '
            '2.7 & 2.8 & 2.9 & 3.0)',
            "Powerlaw slope dU/dM propto U^alpha. Possible values are: 1.0, "
            "1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, "
            "2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0.",
            2.0
        ),
        'gamma': (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction illuminated from Umin to Umax. Possible values between "
            "0 and 1.",
            0.1
        )
    }

    def _init_code(self):
        """Get the model out of the database"""

        self.qhac = float(self.parameters["qhac"])
        self.umin = float(self.parameters["umin"])
        self.alpha = float(self.parameters["alpha"])
        self.gamma = float(self.parameters["gamma"])
        self.umax = 1e7

        with Database("themis") as db:
            self.model_minmin = db.get(qhac=self.qhac, umin=self.umin,
                                       umax=self.umin, alpha=1.)
            self.model_minmax = db.get(qhac=self.qhac, umin=self.umin,
                                       umax=self.umax, alpha=self.alpha)

        # The models in memory are in W/nm for 1 kg of dust. At the same time
        # we need to normalize them to 1 W here to easily scale them from the
        # power absorbed in the UV-optical. If we want to retrieve the dust
        # mass at a later point, we have to save their "emissivity" per unit
        # mass in W (kg of dust)¯¹, The gamma parameter does not affect the
        # fact that it is for 1 kg because it represents a mass fraction of
        # each component.
        self.emissivity = np.trapz((1. - self.gamma) * self.model_minmin.spec +
                                   self.gamma * self.model_minmax.spec,
                                   x=self.model_minmin.wl)

        # We want to be able to display the respective contributions of both
        # components, therefore we keep they separately.
        self.model_minmin.spec *= (1. - self.gamma) / self.emissivity
        self.model_minmax.spec *= self.gamma / self.emissivity

    def process(self, sed):
        """Add the IR re-emission contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """
        if 'dust.luminosity' not in sed.info:
            sed.add_info('dust.luminosity', 1., True, unit='W')
        luminosity = sed.info['dust.luminosity']

        sed.add_module(self.name, self.parameters)
        sed.add_info('dust.qhac', self.qhac)
        sed.add_info('dust.umin', self.umin)
        sed.add_info('dust.alpha', self.alpha)
        sed.add_info('dust.gamma', self.gamma)
        # To compute the dust mass we simply divide the luminosity in W by the
        # emissivity in W/kg of dust.
        sed.add_info('dust.mass', luminosity / self.emissivity, True, unit='kg')

        sed.add_contribution('dust.Umin_Umin', self.model_minmin.wl,
                             luminosity * self.model_minmin.spec)
        sed.add_contribution('dust.Umin_Umax', self.model_minmax.wl,
                             luminosity * self.model_minmax.spec)


# SedModule to be returned by get_module
Module = THEMIS
