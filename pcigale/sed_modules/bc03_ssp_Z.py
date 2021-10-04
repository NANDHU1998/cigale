"""
Bruzual and Charlot (2003) stellar emission module for an SSP
=============================================================

This module implements the Bruzual and Charlot (2003) Single Stellar
Populations.

"""

from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d

from . import SedModule
from ..data import SimpleDatabase as Database


class BC03SSPZ(SedModule):

    parameter_list = OrderedDict(
        [
            (
                "imf",
                (
                    "cigale_list(dtype=int, options=0. & 1.)",
                    "Initial mass function: 0 (Salpeter) or 1 (Chabrier).",
                    0,
                ),
            ),
            (
                "minmetallicity",
                (
                    "cigale_list(minvalue=0.0001)",
                    "Minimum metallicity. It must be larger than 0.0001.",
                    0.0004,
                ),
            ),
            (
                "maxmetallicity",
                (
                    "cigale_list(maxvalue=0.05)",
                    "Maximum metallicity. It must be smaller than 0.05.",
                    0.02,
                ),
            ),
            (
                "minmetallicityage",
                (
                    "cigale_list(dtype=int, minvalue=1)",
                    "Age of the minimum metallicity. Clusters of that age will have "
                    "the corresponding metallicity.",
                    13000,
                ),
            ),
            (
                "separation_age",
                (
                    "cigale_list(dtype=int, minvalue=0)",
                    "Age [Myr] of the separation between the young and the old star "
                    "populations. The default value in 10^7 years (10 Myr). Set "
                    "to 0 not to differentiate ages (only an old population).",
                    10,
                ),
            ),
        ]
    )

    def _init_code(self):
        """Read the SSP from the database."""
        self.imf = int(self.parameters["imf"])
        self.minmetallicity = float(self.parameters["minmetallicity"])
        self.maxmetallicity = float(self.parameters["maxmetallicity"])
        self.minmetallicityage = float(self.parameters["minmetallicityage"])
        self.separation_age = int(self.parameters["separation_age"])

        with Database("bc03_SSP") as db:
            metallicities = np.sort(db.parameters["Z"])
            if (
                self.minmetallicity < metallicities[0]
                or self.maxmetallicity > metallicities[-1]
            ):
                raise Exception("Metallicity is not in the expected range.")

            imf = {0: "salp", 1: "chab"}
            if self.imf in imf:
                self.ssp = {
                    Z: db.get(imf=imf[self.imf], Z=Z) for Z in metallicities
                }
            else:
                raise Exception("IMF #{} unknown".format(self.imf))

    def process(self, sed):
        """Add the convolution of a Bruzual and Charlot SSP to the SED

        Parameters
        ----------
        sed: pcigale.sed.SED
            SED object.

        """
        if "ssp.index" in sed.info:
            index = sed.info["ssp.index"]
        else:
            raise Exception(
                "The stellar models do not correspond to pure SSP."
            )

        age = list(self.ssp.values())[0].t[index]

        Z = (
            self.minmetallicity
            + (self.maxmetallicity - self.minmetallicity)
            * (self.minmetallicityage - age)
            / self.minmetallicityage
        )
        if Z < self.minmetallicity:
            Z = self.minmetallicity
        if Z > self.maxmetallicity:
            Z = self.maxmetallicity

        spec = interp1d(
            np.array([m for m in self.ssp]),
            np.array([self.ssp[m].spec[:, index] for m in self.ssp]).T,
        )(Z)
        info = interp1d(
            np.array([m for m in self.ssp]),
            np.array([self.ssp[m].info[:, index] for m in self.ssp]).T,
        )(Z)

        if age <= self.separation_age:
            spec_young = spec
            info_young = info
            spec_old = np.zeros_like(spec_young)
            info_old = np.zeros_like(info_young)
        else:
            spec_old = spec
            info_old = info
            spec_young = np.zeros_like(spec_old)
            info_young = np.zeros_like(info_old)
        info_all = info_young + info_old

        info_young = dict(zip(["m_star", "m_gas", "n_ly"], info_young))
        info_old = dict(zip(["m_star", "m_gas", "n_ly"], info_old))
        info_all = dict(zip(["m_star", "m_gas", "n_ly"], info_all))
        # We compute the Lyman continuum luminosity as it is important to
        # compute the energy absorbed by the dust before ionising gas.
        wave = list(self.ssp.values())[0].wl
        w = np.where(wave <= 91.1)
        lum_lyc_young, lum_lyc_old = np.trapz(
            [spec_young[w], spec_old[w]], wave[w]
        )

        # We do similarly for the total stellar luminosity
        lum_young, lum_old = np.trapz([spec_young, spec_old], wave)

        sed.add_module(self.name, self.parameters)

        sed.add_info("stellar.imf", self.imf)
        sed.add_info("stellar.metallicity", Z)
        sed.add_info("stellar.old_young_separation_age", self.separation_age)
        sed.add_info("stellar.age", age)

        sed.add_info("stellar.m_star_young", info_young["m_star"], True)
        sed.add_info("stellar.m_gas_young", info_young["m_gas"], True)
        sed.add_info("stellar.n_ly_young", info_young["n_ly"], True)
        sed.add_info("stellar.lum_ly_young", lum_lyc_young, True)
        sed.add_info("stellar.lum_young", lum_young, True)

        sed.add_info("stellar.m_star_old", info_old["m_star"], True)
        sed.add_info("stellar.m_gas_old", info_old["m_gas"], True)
        sed.add_info("stellar.n_ly_old", info_old["n_ly"], True)
        sed.add_info("stellar.lum_ly_old", lum_lyc_old, True)
        sed.add_info("stellar.lum_old", lum_old, True)

        sed.add_info("stellar.m_star", info_all["m_star"], True)
        sed.add_info("stellar.m_gas", info_all["m_gas"], True)
        sed.add_info("stellar.n_ly", info_all["n_ly"], True)
        sed.add_info("stellar.lum_ly", lum_lyc_young + lum_lyc_old, True)
        sed.add_info("stellar.lum", lum_young + lum_old, True)

        sed.add_contribution("stellar.old", wave, spec_old)
        sed.add_contribution("stellar.young", wave, spec_young)


# SedModule to be returned by get_module
Module = BC03SSPZ
