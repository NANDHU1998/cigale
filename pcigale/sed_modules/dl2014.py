"""
Updated Draine and Li (2007) IR models module
=====================================

This module implements the updated Draine and Li (2007) infrared models.

"""

import numpy as np

from pcigale.data import SimpleDatabase as Database
from . import SedModule

__category__ = "dust emission"


class DL2014(SedModule):
    """Updated Draine and Li (2007) templates IR re-emission module

    Given an amount of attenuation (e.g. resulting from the action of a dust
    attenuation module) this module normalises the updated Draine and Li (2007)
    model corresponding to a given set of parameters to this amount of energy
    and add it to the SED.

    Information added to the SED: NAME_alpha.

    """

    parameter_list = {
        'qpah': (
            'cigale_list(options=0.47 & 1.12 & 1.77 & 2.50 & 3.19 & 3.90 & '
            '4.58 & 5.26 & 5.95 & 6.63 & 7.32)',
            "Mass fraction of PAH. Possible values are: 0.47, 1.12, 1.77, "
            "2.50, 3.19, 3.90, 4.58, 5.26, 5.95, 6.63, 7.32.",
            2.50
        ),
        'umin': (
            'cigale_list(options=0.10 & 0.12 & 0.15 & 0.17 & 0.20 & 0.25 & '
            '0.30 & 0.35 & 0.40 & 0.50 & 0.60 & 0.70 & 0.80 & 1.00 & 1.20 & '
            '1.50 & 1.70 & 2.00 & 2.50 & 3.00 & 3.50 & 4.00 & 5.00 & 6.00 & '
            '7.00 & 8.00 & 10.00 & 12.00 & 15.00 & 17.00 & 20.00 & 25.00 & '
            '30.00 & 35.00 & 40.00 & 50.00)',
            "Minimum radiation field. Possible values are: 0.100, 0.120, "
            "0.150, 0.170, 0.200, 0.250, 0.300, 0.350, 0.400, 0.500, 0.600, "
            "0.700, 0.800, 1.000, 1.200, 1.500, 1.700, 2.000, 2.500, 3.000, "
            "3.500, 4.000, 5.000, 6.000, 7.000, 8.000, 10.00, 12.00, 15.00, "
            "17.00, 20.00, 25.00, 30.00, 35.00, 40.00, 50.00.",
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

        self.qpah = float(self.parameters["qpah"])
        self.umin = float(self.parameters["umin"])
        self.alpha = float(self.parameters["alpha"])
        self.gamma = float(self.parameters["gamma"])

        # We also compute <U>. For this we consider Eq. 6 and 15 of Draine & Li
        # (2007) taking into account that α can be different from 2. We then
        # just compute the integral. We have two specific cases, α=1 and α=2.
        self.umax = 1e7
        self.umean = (1. - self.gamma) * self.umin
        if self.alpha == 1.0:
            self.umean += self.gamma * (self.umax - self.umin) / \
                np.log(self.umax / self.umin)
        elif self.alpha == 2.0:
            self.umean += self.gamma * np.log(self.umax / self.umin) / \
                (1. / self.umin - 1. / self.umax)
        else:
            oma = 1. - self.alpha
            tma = 2. - self.alpha
            self.umean += self.gamma * oma / tma * \
                (self.umin**tma - self.umax**tma) / \
                (self.umin**oma - self.umax**oma)

        with Database("dl2014") as db:
            self.model_minmin = db.get(qpah=self.qpah, umin=self.umin,
                                       umax=self.umin, alpha=1.)
            self.model_minmax = db.get(qpah=self.qpah, umin=self.umin,
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
        sed.add_info('dust.qpah', self.qpah)
        sed.add_info('dust.umin', self.umin)
        sed.add_info('dust.umean', self.umean)
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
Module = DL2014
