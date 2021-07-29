import numpy as np
import scipy.constants as cst

from pcigale.data import SimpleDatabase as Database
from . import SedModule

default_lines = ['Ly-alpha',
                 'CII-133.5',
                 'SiIV-139.7',
                 'CIV-154.9',
                 'HeII-164.0',
                 'OIII-166.5',
                 'CIII-190.9',
                 'CII-232.6',
                 'MgII-279.8',
                 'OII-372.7',
                 'H-10',
                 'H-9',
                 'NeIII-386.9',
                 'HeI-388.9',
                 'H-epsilon',
                 'SII-407.0',
                 'H-delta',
                 'H-gamma',
                 'H-beta',
                 'OIII-495.9',
                 'OIII-500.7',
                 'OI-630.0',
                 'NII-654.8',
                 'H-alpha',
                 'NII-658.4',
                 'SII-671.6',
                 'SII-673.1'
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
        'metallicity': (
             'cigale_list(options=-0.8 & -0.7 & -0.6 & -0.5 & -0.4 & -0.3 & '
             '-0.2 & -0.1 & 0.0 & 0.1 & 0.4 & 0.5 & 0.3 & 0.2 & -2.0 & -3.0 '
             '& -1.0 & -1.5)',
             "zetaO metallicity",
             0.0
        ),
        'delta': (
            'cigale_list(options=0.0)',
            'Radiation field shape parameter',
            0.0
        ),
        'f_NLR': (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "NLR covering factor",
            0.2
        ),
        'nH_NLR': (
            'cigale_list(options=2.0 & 3.0 & 4.0',
            'NLR density',
            3.0
        ),
        'f_BLR': (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "NLR covering factor",
            0.1
        ),
        'nH_BLR': (
            'cigale_list(options=8.0 & 10.0 & 12.0',
            'BLR density',
            10.0
        ),
        'logU': (
            'cigale_list(options=-4.0 & -3.9 & -3.8 & -3.7 & -3.6 & -3.5 & '
            '-3.4 & -3.3 & -3.2 & -3.1 & -3.0 & -2.9 & -2.8 & -2.7 & -2.6 & '
            '-2.5)',
            "Ionisation parameter",
            -3.0
        ),
        'lines_width_NLR': (
            'cigale_list(minvalue=0.)',
            "Line width in km/s",
            300.
        ),
        'lines_width_BLR': (
            'cigale_list(minvalue=0.)',
            "Line width in km/s",
            800.
        ),
    }

    def _init_code(self):
        """Get the nebular emission lines out of the database and resample
           them to see the line profile. Compute scaling coefficients.
        """
        self.metallicity = float(self.parameters['metallicity'])
        self.disk = self.parameters['disk_model']
        self.delta = float(self.parameters['delta'])
        self.covfact = {
            "NLR": float(self.parameters['f_NLR']),
            "BLR": float(self.parameters['f_BLR']),
        }
        self.density = {
            "NLR": float(self.parameters['nH_NLR']),
            "BLR": float(self.parameters['nH_BLR']),
        }
        self.logU = float(self.parameters['logU'])
        self.lines_width = {
            "NLR": float(self.parameters['lines_width_NLR']),
            "BLR": float(self.parameters['lines_width_BLR'])
        }

        # Cache for getting model from the database.
        self.model = None
        self.disk_type = None

    def process(self, sed):
        """Add the nebular emission lines

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """

        # Get the model from the database if we don't have it. All the SEDs are
        # expected to use the same disk type.
        if self.model is None:

            self.disk_type = {
                0: "SKIRTOR",
                1: "Schartmann",
            }[sed.info['agn.disk_type']]

            with Database('nebular_AGN') as db:
                self.model = {
                    "NLR": db.get(
                        region='NLR',
                        disk=self.disk,
                        delta=self.delta,
                        Z=self.metallicity,
                        logU=self.logU,
                        density=self.density['NLR']),
                    "BLR": db.get(
                        region='BLR',
                        disk=self.disk,
                        delta=self.delta,
                        Z=self.metallicity,
                        logU=self.logU,
                        density=self.density['BLR']),
                }

        # calculate the number of hydrogen-ionizing photons Q(H)
        wavelength = sed.wavelength_grid
        mask_Q = (wavelength <= 91.5)  # select the photons ionizing hydrogen,
                                       # below 91.5 nm
        wvl_H = wavelength[mask_Q]
        luminosity_H = sed.get_lumin_contribution(
            'agn.SKIRTOR2016_disk')[mask_Q]
        hc = 6.62607004e-34 * 2.99792458e8  # kg m2 s-1 * m s-1  (SI)
        NLy = np.trapz(wvl_H*luminosity_H/hc, x=wvl_H)  # lambda*F_lambda is
                                                        # in W
        sed.add_module(self.name, self.parameters)

        sed.add_contribution(
            'nebular.continuum_NLR',
            self.model['NLR'].cont_wave,
            self.model['NLR'].cont_spec * NLy * self.covfact['NLR']
        )
        sed.add_contribution(
            'nebular.continuum_BLR',
            self.model['BLR'].cont_wave,
            self.model['BLR'].cont_spec * NLy * self.covfact['BLR']
        )

        # Compute lines wavelength grid and flux for NLR and BLR lines.
        lines = {}
        for region in ['BLR', 'NLR']:
            lines_wave = self.model[region].lines_wave
            lines_spec = self.model[region].lines_spec
            lines_width = lines_wave * self.lines_width[region] * 1e3 / cst.c

            # Wavelength grid of lines contribution. We take 9 points around
            # each central wavelength, in 6 times the width range.
            wavelength_grid = sorted(np.concatenate([
                np.linspace(wave - 3 * width, wave + 3 * width,  9) for
                wave, width in zip(lines_wave, lines_width)
            ]))

            # Lines luminosity as Gaussian.
            lumin = np.zeros_like(wavelength_grid)
            for wave, lumin, width in zip(lines_wave, lines_spec, lines_width):
                lumin += (
                    lumin * np.exp(
                        - 4. * np.log(2.) * (wavelength_grid - wave) ** 2. /
                        (width * width)) / (width * np.sqrt(np.pi / np.log(2.))
                                            / 2.)
                )

            sed.add_contribution(
                f"nebular_lines_{region}",
                wavelength_grid,
                lumin * NLy * self.covfact[region]
            )

        sed.add_info('AGN.nebular.lines_width_NLR', self.lines_width['NLR'],
                     unit='km/s')
        sed.add_info('AGN.nebular.lines_width_BLR', self.lines_width['BLR'],
                     unit='km/s')
        sed.add_info('AGN.nebular.logU', self.logU)

# SedModule to be returned by get_module
Module = NebularEmissionAGN
