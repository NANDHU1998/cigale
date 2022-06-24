import numpy as np
import scipy.constants as cst

from pcigale.data import SimpleDatabase as Database
from . import SedModule

default_lines = [
    "Ly-alpha",
    "CII-133.5",
    "SiIV-139.7",
    "CIV-154.9",
    "HeII-164.0",
    "OIII-166.5",
    "CIII-190.9",
    "CII-232.6",
    "MgII-279.8",
    "OII-372.7",
    "H-10",
    "H-9",
    "NeIII-386.9",
    "HeI-388.9",
    "H-epsilon",
    "SII-407.0",
    "H-delta",
    "H-gamma",
    "H-beta",
    "OIII-495.9",
    "OIII-500.7",
    "OI-630.0",
    "NII-654.8",
    "H-alpha",
    "NII-658.4",
    "SII-671.6",
    "SII-673.1",
]


class NebularEmissionAGN(SedModule):
    """
    Module computing the nebular emission from the ultraviolet to the
    near-infrared for AGN. It includes both the nebular lines and the nebular
    continuum (optional) for both the NLR and BLR regions. It takes into account
    the escape fraction and the absorption by dust.

    Both the nebular continuum and the flux of each line are scaled directly
    from the number of ionizing photons NLy given by the AGN radiation field.

    """

    parameter_list = {
        "metallicity": (
            "cigale_list(options=-0.8 & -0.7 & -0.6 & -0.5 & -0.4 & -0.3 & "
            "-0.2 & -0.1 & 0.0 & 0.1 & 0.4 & 0.5 & 0.3 & 0.2 & -2.0 & -3.0 "
            "& -1.0 & -1.5)",
            "zetaO metallicity",
            0.0,
        ),
        "delta": ("cigale_list(options=0.0)", "Radiation field shape parameter", 0.0),
        "f_NLR": ("cigale_list(minvalue=0., maxvalue=1.)", "NLR covering factor", 0.2),
        "nH_NLR": ("cigale_list(options=2.0 & 3.0 & 4.0)", "NLR density", 3.0),
        "f_BLR": ("cigale_list(minvalue=0., maxvalue=1.)", "NLR covering factor", 0.1),
        "nH_BLR": ("cigale_list(options=8.0 & 10.0 & 12.0)", "BLR density", 10.0),
        "logU": (
            "cigale_list(options=-4.0 & -3.9 & -3.8 & -3.7 & -3.6 & -3.5 & "
            "-3.4 & -3.3 & -3.2 & -3.1 & -3.0 & -2.9 & -2.8 & -2.7 & -2.6 & "
            "-2.5)",
            "Ionisation parameter",
            -3.0,
        ),
        "lines_width_NLR": ("cigale_list(minvalue=0.)", "Line width in km/s", 300.0),
        "lines_width_BLR": ("cigale_list(minvalue=0.)", "Line width in km/s", 800.0),
    }

    def _init_code(self):
        """Get the nebular emission lines out of the database and resample
        them to see the line profile. Compute scaling coefficients.
        """

        self.regions = ["NLR", "BLR"]

        self.metallicity = float(self.parameters["metallicity"])
        self.delta = float(self.parameters["delta"])
        self.covfact = {
            region: float(self.parameters[f"f_{region}"]) for region in self.regions
        }
        self.density = {
            region: float(self.parameters[f"nH_{region}"]) for region in self.regions
        }
        self.logU = float(self.parameters["logU"])
        self.lines_width = {
            region: float(self.parameters[f"lines_width_{region}"])
            for region in self.regions
        }

        # Get the model from the database if we don't have it. All the SEDs are
        # expected to use the same disk type.
        with Database("nebular_agn") as db:
            self.models = {
                (region, disk_type): db.get(
                    region=region,
                    disk=disk_type,
                    delta=self.delta,
                    Z=self.metallicity,
                    logU=self.logU,
                    density=self.density[region],
                )
                for region in self.regions
                for disk_type in ["SKIRTOR", "Schartmann"]
            }

        self.invhc = 1.0 / (cst.h * cst.c)
        self.lines_wl = {}
        self.lines_flux = {}

        # Compute lines wavelength grid and flux for NLR and BLR lines.
        k1 = -4.0 * np.log(2)
        k2 = 0.5 * np.sqrt(np.pi / np.log(2))
        for region, disk_type in self.models:
            lines_wave = self.models[(region, disk_type)].lines_wave
            lines_spec = self.models[(region, disk_type)].lines_spec
            lines_width = lines_wave * self.lines_width[region] * 1e3 / cst.c

            # Wavelength grid of lines contribution. We take 9 points around
            # each central wavelength, in 6 times the width range.
            new_wl = np.sort(
                np.concatenate(
                    [
                        np.linspace(wave - 3 * width, wave + 3 * width, 9)
                        for wave, width in zip(lines_wave, lines_width)
                    ]
                )
            )

            # Lines luminosity as Gaussian.
            new_flux = np.zeros_like(new_wl)
            for wave, lumin, width in zip(lines_wave, lines_spec, lines_width):
                new_flux += (
                    lumin
                    * np.exp(k1 / (width * width) * (new_wl - wave) ** 2.0)
                    / (width * k2)
                )

            self.lines_wl[(region, disk_type)] = new_wl
            self.lines_flux[(region, disk_type)] = new_flux

    def process(self, sed):
        """Add the nebular emission lines

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        # calculate the number of hydrogen-ionizing photons Q(H)
        wavelength = sed.wavelength_grid
        mask_Q = wavelength <= 91.5  # select the photons ionizing hydrogen,
        # below 91.5 nm
        wvl_H = wavelength[mask_Q]
        luminosity_H = sed.luminosities["agn.SKIRTOR2016_disk"][mask_Q]
        NLy = np.trapz(wvl_H * luminosity_H, x=wvl_H) * self.invhc * 1e-9

        sed.add_module(self.name, self.parameters)

        disk_type = {
            0: "SKIRTOR",
            1: "Schartmann",
        }[sed.info["agn.disk_type"]]

        for region in self.regions:
            key = (region, disk_type)
            sed.add_contribution(
                f"nebular.continuum_{region}",
                self.models[key].cont_wave,
                self.models[key].cont_spec * NLy * self.covfact[region],
            )
            sed.add_contribution(
                f"nebular.lines_{region}",
                self.lines_wl[key],
                self.lines_flux[key] * NLy * self.covfact[region],
            )

            sed.add_info(
                f"AGN.nebular.lines_width_{region}",
                self.lines_width[region],
                unit="km/s",
            )

        sed.add_info("AGN.nebular.logU", self.logU)


# SedModule to be returned by get_module
Module = NebularEmissionAGN
