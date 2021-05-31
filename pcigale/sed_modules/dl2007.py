"""
Draine and Li (2007) IR models module
=====================================

This module implements the Draine and Li (2007) infra-red models.

"""

import numpy as np

from pcigale.data import SimpleDatabase as Database
from . import SedModule


class DL2007(SedModule):
    """Draine and Li (2007) templates IR re-emission module

    Given an amount of attenuation (e.g. resulting from the action of a dust
    attenuation module) this module normalises the Draine and Li (2007)
    template corresponding to a given α to this amount of energy and add it
    to the SED.

    Information added to the SED: NAME_alpha.

    """

    parameter_list = {
        'qpah': (
            'cigale_list(options=0.47 & 1.12 & 1.77 & 2.50 & 3.19 & 3.90 & '
            '4.58)',
            "Mass fraction of PAH. Possible values are: 0.47, 1.12, 1.77, "
            "2.50, 3.19, 3.90, 4.58.",
            2.50
        ),
        'umin': (
            'cigale_list(options=0.10 & 0.15 & 0.20 & 0.30 & 0.40 & 0.50 & '
            '0.70 & 0.80 & 1.00 & 1.20 & 1.50 & 2.00 & 2.50 & 3.00 & 4.00 & '
            '5.00 & 7.00 & 8.00 & 10.0 & 12.0 & 15.0 & 20.0 & 25.0)',
            "Minimum radiation field. Possible values are: 0.10, 0.15, 0.20, "
            "0.30, 0.40, 0.50, 0.70, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, "
            "3.00, 4.00, 5.00, 7.00, 8.00, 10.0, 12.0, 15.0, 20.0, 25.0.",
            1.0
        ),
        'umax': (
            'cigale_list(options=1e3 & 1e4 & 1e5 & 1e6)',
            "Maximum radiation field. Possible values are: 1e3, 1e4, 1e5, "
            "1e6.",
            1e6
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
        self.umax = float(self.parameters["umax"])
        self.gamma = float(self.parameters["gamma"])

        # We also compute <U>
        self.umean = (1. - self.gamma) * self.umin + \
            self.gamma * np.log(self.umax / self.umin) / \
            (1. / self.umin - 1. / self.umax)

        with Database("dl2007") as db:
            self.model_minmin = db.get(qpah=self.qpah, umin=self.umin,
                                       umax=self.umin)
            self.model_minmax = db.get(qpah=self.qpah, umin=self.umin,
                                       umax=self.umax)

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
        sed.add_info('dust.umax', self.umax)
        sed.add_info('dust.umean', self.umean)
        sed.add_info('dust.gamma', self.gamma)
        # To compute the dust mass we simply divide the luminosity in W by the
        # emissivity in W/kg of dust.
        sed.add_info('dust.mass', luminosity / self.emissivity, True, unit='kg')

        sed.add_contribution('dust.Umin_Umin', self.model_minmin.wl,
                             luminosity * self.model_minmin.spec)
        sed.add_contribution('dust.Umin_Umax', self.model_minmax.wl,
                             luminosity * self.model_minmax.spec)


# SedModule to be returned by get_module
Module = DL2007
