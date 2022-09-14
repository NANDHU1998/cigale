"""
Draine and Li (2021) IR models module
=====================================

This module implements the Draine and Li (2021) infra-red models.

"""

import numpy as np
from scipy.interpolate import interp1d

from pcigale.data import SimpleDatabase as Database
from pcigale.sed_modules import SedModule
from pcigale.sed_modules.dustextPHANGS import FitzIndeb

__category__ = "dust emission"


class DL2021(SedModule):
    """Updated Draine and Li (2007) templates IR re-emission module

    Given an amount of attenuation (e.g. resulting from the action of a dust
    attenuation module) this module normalises the updated Draine and Li (2007)
    model corresponding to a given set of parameters to this amount of energy
    and add it to the SED.

    Information added to the SED: NAME_alpha.

    """

    parameter_list = {
        "lgu": (
            "cigale_list(options=0.0 & 0.5 & 1.0 & 1.5 & 2.0 & 2.5 & 3.0 & 3.5 & 4.0 & 4.5 & 5.0 & 5.5 & 6.0 & 6.5 & 7.0)",
            "log U. Possible values are: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0",
            0.0,
        ),
        "fion": (
            "cigale_list(options=0 & 1 & 2)",
            "PAH ionization. Possible values are: 0 (low), 1 (standard), and 2 (high)",
            1,
        ),
        "fsize": (
            "cigale_list(options=0 & 1 & 2)",
            "Grain size distribution. Possible values are: 0 (small), 1 (standard), and 2 (large)",
            1,
        ),
        "self_abs": ("boolean()", "Take self-absorption into account.", False),
    }

    def _init_code(self):
        """Get the model out of the database"""

        self.lgu = float(self.parameters["lgu"])
        self.fion = int(self.parameters["fion"])
        self.fsize = int(self.parameters["fsize"])
        self.self_abs = bool(self.parameters["self_abs"])

        fion = {0: "lo", 1: "st", 2: "hi"}[self.fion]
        fsize = {0: "sma", 1: "std", 2: "lrg"}[self.fsize]

        with Database("dl2021") as db:
            ages = db.parameters["age"]

            models = {
                age: db.get(age=age, lgu=self.lgu, fion=fion, fsize=fsize)
                for age in ages
            }
        self.wl = models[3e6].wl

        beta0 = np.array([model.beta0 for model in models.values()])
        # The models in memory are in W/nm for 1 kg of dust. At the same time
        # we need to normalize them to 1 W here to easily scale them from the
        # power absorbed in the UV-optical. If we want to retrieve the dust
        # mass at a later point, we have to save their "emissivity" per unit
        # mass in W (kg of dust)¯¹, The gamma parameter does not affect the
        # fact that it is for 1 kg because it represents a mass fraction of
        # each component.
        emissivity = np.array(
            [np.trapz(np.sum(models[age].spec, axis=-1), x=self.wl) for age in ages]
        )
        self.emissivity = interp1d(
            beta0,
            emissivity,
            bounds_error=False,
            fill_value=(emissivity[0], emissivity[-1]),
        )

        # We want to be able to display the respective contributions of both
        # components, therefore we keep they separately.
        for age in ages:
            models[age].spec /= self.emissivity(models[age].beta0)

        spectra = np.moveaxis(
            np.array([model.spec for model in models.values()]), -2, -1
        )

        self.spec = interp1d(
            beta0,
            spectra,
            axis=0,
            bounds_error=False,
            fill_value=(spectra[0, ...], spectra[-1, ...]),
        )

        if self.self_abs is True:
            self.att = FitzIndeb().interp(self.wl)

    def process(self, sed):
        """Add the IR re-emission contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """
        if "dust.luminosity" not in sed.info:
            sed.add_info("dust.luminosity", 1.0, True, unit="W")
        luminosity = sed.info["dust.luminosity"]

        sed.add_module(self.name, self.parameters)
        sed.add_info("dust.lgu", self.lgu)
        sed.add_info("dust.fion", self.fion)
        sed.add_info("dust.fsize", self.fsize)

        # To compute the dust mass we simply divide the luminosity in W by the
        # emissivity in W/kg of dust.
        beta0 = sed.info["stellar.beta0"]
        sed.add_info("dust.mass", luminosity / self.emissivity(beta0), True, unit="kg")

        spectra = self.spec(beta0)
        if self.self_abs is True:
            att = 10.0 ** (-0.4 * self.att * sed.info["attenuation.A550"]) - 1.0
            luminosity /= 1.0 + np.trapz(att * np.sum(spectra, axis=0), x=self.wl)
            for component, spectrum in zip(["astrodust", "PAH0", "PAH+"], spectra):
                sed.add_contribution(
                    f"dust.att_{component}", self.wl, luminosity * spectrum * att
                )

        for component, spectrum in zip(["astrodust", "PAH0", "PAH+"], spectra):
            sed.add_contribution(f"dust.{component}", self.wl, luminosity * spectrum)


# SedModule to be returned by get_module
Module = DL2021
