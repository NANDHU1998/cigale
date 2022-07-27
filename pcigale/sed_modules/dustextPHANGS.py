"""
Simple screen extinction module
=====================================================================

This module implements a simple dust extinction with self-absorption

"""

from collections import OrderedDict
import pkg_resources

import numpy as np

from . import SedModule

from astropy.table import Table
from scipy.interpolate import interp1d


class FitzIndeb:
    tableFitzIndeb = Table.read(
        pkg_resources.resource_filename(
            __name__, f"curves/FitzIndeb_3.1.fits"
        )
    )
    tableFitzIndeb["opacity"][tableFitzIndeb["wave"] < 911.0] = 0.0
    interpolator = interp1d(
        tableFitzIndeb["wave"].data * 0.1,
        tableFitzIndeb["opacity"].data / 211.4,
        fill_value=0.0,
        bounds_error=False,
    )

    def __init__(self):
        pass

    def interp(self, wl):
        return self.interpolator(wl)


class DustExtinction(SedModule):
    """Screen extinction law

    This module computes the screen extinction with the Indebetouw curve.

    The extinction is computed for all the components and is added to the SED as
    a negative contribution.

    """

    parameter_list = OrderedDict(
        [
            (
                "A550",
                ("cigale_list(minvalue=0.)", "Attenuation at 550 nm.", 0.3),
            ),
            (
                "filters",
                (
                    "string()",
                    "Filters for which the extinction will be computed and added to "
                    "the SED information dictionary. You can give several filter "
                    "names separated by a & (don't use commas).",
                    "B_B90 & V_B90 & FUV",
                ),
            ),
        ]
    )

    def _init_code(self):
        """Get the filters from the database"""

        self.A550 = float(self.parameters["A550"])
        self.filter_list = [
            item.strip() for item in self.parameters["filters"].split("&")
        ]
        # We cannot compute the extinction until we know the wavelengths. Yet,
        # we reserve the object.
        self.att = None
        self.lineatt = {}
        self.fitzindeb = FitzIndeb().interp

    def process(self, sed):
        """Add the extinction component to each of the emission components.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        wl = sed.wavelength_grid

        # Fλ fluxes (only from continuum) in each filter before extinction.
        flux_noatt = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Compute stellar extinction curve
        if self.att is None:
            self.att = 10 ** (-0.4 * self.fitzindeb(wl) * self.A550)

        # Compute nebular extinction curves
        if len(self.lineatt) == 0:
            names = [k for k in sed.lines]
            linewl = np.array([sed.lines[k][0] for k in names])
            self.lineatt["nebular"] = self.fitzindeb(wl)
            for name, att in zip(names, self.fitzindeb(linewl)):
                self.lineatt[name] = att
            for k, v in self.lineatt.items():
                self.lineatt[k] = 10.0 ** (-0.4 * v * self.A550)

        dust_lumin = 0.0
        contribs = [
            contrib
            for contrib in sed.luminosities
            if "absorption" not in contrib
        ]

        for contrib in contribs:
            luminosity = sed.luminosities[contrib]
            if "nebular" in contrib:
                extinction_spec = luminosity * (self.lineatt["nebular"] - 1.0)
            else:
                extinction_spec = luminosity * (self.att - 1.0)
            dust_lumin -= np.trapz(extinction_spec, wl)

            sed.add_module(self.name, self.parameters)
            sed.add_contribution("attenuation." + contrib, wl, extinction_spec)

        for name, (linewl, old, young) in sed.lines.items():
            sed.lines[name] = (
                linewl,
                old * self.lineatt[name],
                young * self.lineatt[name],
            )

        # Total extinction
        if "dust.luminosity" in sed.info:
            sed.add_info(
                "dust.luminosity",
                sed.info["dust.luminosity"] + dust_lumin,
                True,
                True,
            )
        else:
            sed.add_info("dust.luminosity", dust_lumin, True)

        # Fλ fluxes (only from continuum) in each filter after extinction.
        flux_att = {filt: sed.compute_fnu(filt) for filt in self.filter_list}

        # Attenuation in each filter
        for filt in self.filter_list:
            sed.add_info(
                "attenuation." + filt,
                -2.5 * np.log10(flux_att[filt] / flux_noatt[filt]),
            )

        sed.add_info("attenuation.A550", self.A550)


# SedModule to be returned by get_module
Module = DustExtinction
