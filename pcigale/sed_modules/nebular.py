import numpy as np
import scipy.constants as cst

from pcigale.data import SimpleDatabase as Database
from . import SedModule

__category__ = "nebular"

default_lines = ['Ly-alpha',
                 'HeII-164.0',
                 'OII-372.6',
                 'OII-372.9',
                 'H-delta',
                 'H-gamma',
                 'H-beta',
                 'OIII-495.9',
                 'OIII-500.7',
                 'OI-630.0',
                 'NII-654.8',
                 'H-alpha',
                 'NII-658.3',
                 'SII-671.6',
                 'SII-673.1'
                 ]


class NebularEmission(SedModule):
    """
    Module computing the nebular emission from the ultraviolet to the
    near-infrared. It includes both the nebular lines and the nubular
    continuum (optional). It takes into account the escape fraction and the
    absorption by dust.

    Given the number of Lyman continuum photons, we compute the Hβ line
    luminosity. We then compute the other lines using the
    metallicity-dependent templates that provide the ratio between individual
    lines and Hβ. The nebular continuum is scaled directly from the number of
    ionizing photons.

    """

    parameter_list = {
        'logU': (
            'cigale_list(options=-4.0 & -3.9 & -3.8 & -3.7 & -3.6 & -3.5 & '
            '-3.4 & -3.3 & -3.2 & -3.1 & -3.0 & -2.9 & -2.8 & -2.7 & -2.6 & '
            '-2.5 & -2.4 & -2.3 & -2.2 & -2.1 & -2.0 & -1.9 & -1.8 & -1.7 & '
            '-1.6 & -1.5 & -1.4 & -1.3 & -1.2 & -1.1 & -1.0)',
            "Ionisation parameter",
            -2.
        ),
        'zgas': (
            'cigale_list(options=0.0001 & 0.0004 & 0.001 & 0.002 & 0.0025 & '
            '0.003 & 0.004 & 0.005 & 0.006 & 0.007 & 0.008 & 0.009 & 0.011 & '
            '0.012 & 0.014 & 0.016 & 0.019 & 0.020 & 0.022 & 0.025 & 0.03 & '
            '0.033 & 0.037 & 0.041 & 0.046 & 0.051)',
            "Gas metallicity",
            0.02
        ),
        'ne': (
            'cigale_list(options=10 & 100 & 1000)',
            "Electron density",
            100
        ),
        'f_esc': (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons escaping the galaxy",
            0.
        ),
        'f_dust': (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons absorbed by dust",
            0.
        ),
        'lines_width': (
            'cigale_list(minvalue=0.)',
            "Line width in km/s",
            300.
        ),
        'emission': (
            'boolean()',
            "Include nebular emission.",
            True
        )
    }

    def _init_code(self):
        """Get the nebular emission lines out of the database and resample
           them to see the line profile. Compute scaling coefficients.
        """
        self.logU = float(self.parameters['logU'])
        self.zgas = float(self.parameters['zgas'])
        self.ne = float(self.parameters['ne'])
        self.fesc = float(self.parameters['f_esc'])
        self.fdust = float(self.parameters['f_dust'])
        self.lines_width = float(self.parameters['lines_width'])
        if type(self.parameters["emission"]) is str:
            self.emission = self.parameters["emission"].lower() == 'true'
        else:
            self.emission = bool(self.parameters["emission"])

        if self.fesc < 0. or self.fesc > 1:
            raise Exception("Escape fraction must be between 0 and 1")

        if self.fdust < 0 or self.fdust > 1:
            raise Exception("Fraction of lyman photons absorbed by dust must "
                            "be between 0 and 1")

        if self.fesc + self.fdust > 1:
            raise Exception("Escape fraction+f_dust>1")

        if self.emission:
            with Database("nebular_continuum") as db:
                metallicities = db.parameters['Z']
                self.cont_template = {m: db.get(Z=m, logU=self.logU, ne=self.ne)
                                      for m in metallicities}

            with Database("nebular_lines") as db:
                self.lines_template = {m: db.get(Z=m, logU=self.logU, ne=self.ne)
                                       for m in metallicities}

            self.linesdict = {m: dict(zip(self.lines_template[m].name,
                                          zip(self.lines_template[m].wl,
                                              self.lines_template[m].spec)))
                              for m in metallicities}

            for lines in self.lines_template.values():
                new_wave = np.array([])
                for line_wave in lines.wl:
                    width = line_wave * self.lines_width * 1e3 / cst.c
                    new_wave = np.concatenate((new_wave,
                                               np.linspace(line_wave - 3. * width,
                                                           line_wave + 3. * width,
                                                           9)))
                new_wave.sort()
                new_flux = np.zeros_like(new_wave)
                log2 = np.log(2)
                for line_flux, line_wave in zip(lines.spec, lines.wl):
                    width = line_wave * self.lines_width * 1e3 / cst.c
                    new_flux += (line_flux * np.exp(- 4. * log2 *
                                (new_wave - line_wave) ** 2. / (width * width)) /
                                (width * np.sqrt(np.pi / log2) / 2.))
                lines.wl = new_wave
                lines.spec = new_flux

            # To take into acount the escape fraction and the fraction of Lyman
            # continuum photons absorbed by dust we correct by a factor
            # k=(1-fesc-fdust)/(1+(α1/αβ)*(fesc+fdust))
            alpha_B = 2.58e-19  # Ferland 1980, m³ s¯¹
            alpha_1 = 1.54e-19  # αA-αB, Ferland 1980, m³ s¯¹
            k = (1. - self.fesc - self.fdust) / (1. + alpha_1 / alpha_B * (
                self.fesc + self.fdust))

            self.corr = k
        self.idx_Ly_break = None
        self.absorbed_old = None
        self.absorbed_young = None

    def process(self, sed):
        """Add the nebular emission lines

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """
        if self.idx_Ly_break is None:
            self.idx_Ly_break = np.searchsorted(sed.wavelength_grid, 91.2)
            self.absorbed_old = np.zeros(sed.wavelength_grid.size)
            self.absorbed_young = np.zeros(sed.wavelength_grid.size)

        self.absorbed_old[:self.idx_Ly_break] = -(
            sed.luminosities['stellar.old'][:self.idx_Ly_break] *
            (1. - self.fesc))
        self.absorbed_young[:self.idx_Ly_break] = -(
            sed.luminosities['stellar.young'][:self.idx_Ly_break] *
            (1. - self.fesc))

        sed.add_module(self.name, self.parameters)
        sed.add_info('nebular.f_esc', self.fesc)
        sed.add_info('nebular.f_dust', self.fdust)
        sed.add_info('dust.luminosity', (sed.info['stellar.lum_ly_young'] +
                     sed.info['stellar.lum_ly_old']) * self.fdust, True,
                     unit='W')

        sed.add_contribution('nebular.absorption_old', sed.wavelength_grid,
                             self.absorbed_old)
        sed.add_contribution('nebular.absorption_young', sed.wavelength_grid,
                             self.absorbed_young)

        if self.emission:
            NLy_old = sed.info['stellar.n_ly_old']
            NLy_young = sed.info['stellar.n_ly_young']
            metallicity = self.zgas
            lines = self.lines_template[metallicity]
            linesdict = self.linesdict[metallicity]
            cont = self.cont_template[metallicity]

            sed.add_info('nebular.lines_width', self.lines_width, unit='km/s')
            sed.add_info('nebular.logU', self.logU)
            sed.add_info('nebular.zgas', self.zgas)
            sed.add_info('nebular.ne', self.ne, unit='cm^-3')

            for line in default_lines:
                wave, ratio = linesdict[line]
                sed.lines[line] = (wave,
                                   ratio * NLy_old * self.corr,
                                   ratio * NLy_young * self.corr)

            sed.add_contribution('nebular.lines_old', lines.wl,
                                 lines.spec * NLy_old * self.corr)
            sed.add_contribution('nebular.lines_young', lines.wl,
                                 lines.spec * NLy_young * self.corr)

            sed.add_contribution('nebular.continuum_old', cont.wl,
                                 cont.spec * NLy_old * self.corr)
            sed.add_contribution('nebular.continuum_young', cont.wl,
                                 cont.spec * NLy_young * self.corr)


# SedModule to be returned by get_module
Module = NebularEmission
