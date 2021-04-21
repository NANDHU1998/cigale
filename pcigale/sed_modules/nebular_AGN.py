# -*- coding: utf-8 -*-
# Copyright (C) 2014 University of Cambridge
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien <mboquien@ast.cam.ac.uk>

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import scipy.constants as cst

from pcigale.data import Database
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
#default_lines = [] to deactivate the previous list

class NebularEmissionAGN(SedModule):
    """
    Module computing the nebular emission from the ultraviolet to the
    near-infrared for AGN. It includes both the nebular lines and the nebular
    continuum (optional) for both the NLR and BLR regions. It takes into account the escape fraction and the
    absorption by dust.

    Both the nebular continuum and the flux of each line are scaled directly from the number of ionizing photons NLy given by the AGN radiation field (!!!! set to 1 for now (L229) !!!!).

    """

    parameter_list = OrderedDict([
        ('metallicity', (
             'cigale_list(options=-0.5 & 0.0 & 0.5)',
             "dzetaO metallicity",
             0.0
         )),
        ('f_NLR', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "NLR covering factor",
            0.2
        )),
         ('f_BLR', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "NLR covering factor",
            0.1
        )),
        
        ('logU', (
            'cigale_list(options=-4.0 & -3.9 & -3.8 & -3.7 & -3.6 & -3.5 & '
            '-3.4 & -3.3 & -3.2 & -3.1 & -3.0 & -2.9 & -2.8 & -2.7 & -2.6 & '
            '-2.5 & -2.4 & -2.3 & -2.2 & -2.1 & -2.0 & -1.9 & -1.8 & -1.7 & '
            '-1.6 & -1.5 & -1.4 & -1.3 & -1.2 & -1.1 & -1.0)',
            "Ionisation parameter",
            -3.
        )),
        ('f_esc', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons escaping the galaxy",
            0.
        )),
        ('f_dust', (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Fraction of Lyman continuum photons absorbed by dust",
            0.
        )),
        ('lines_width_NLR', (
            'cigale_list(minvalue=0.)',
            "Line width in km/s",
            300.
        )),
        ('lines_width_BLR', (
            'cigale_list(minvalue=0.)',
            "Line width in km/s",
            800.
        )),
        ('emission', (
            'boolean()',
            "Include nebular emission.",
            True
        ))
    ])

    def _init_code(self):
        """Get the nebular emission lines out of the database and resample
           them to see the line profile. Compute scaling coefficients.
        """
        self.metallicity = float(self.parameters['metallicity'])
        self.f_NLR = float(self.parameters['f_NLR'])
        self.f_BLR = float(self.parameters['f_BLR'])
        self.logU = float(self.parameters['logU'])
        self.fesc = float(self.parameters['f_esc'])
        self.fdust = float(self.parameters['f_dust'])
        self.lines_width_NLR = float(self.parameters['lines_width_NLR'])
        self.lines_width_BLR = float(self.parameters['lines_width_BLR'])
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
            with Database() as db:
                metallicities = db.get_nebular_continuum_AGN_parameters()['metallicity']               
                self.lines_template_NLR = {m: db.get_nebular_lines_AGN('NLR', m, self.logU)
                                    for m in metallicities}               
                self.cont_template_NLR = {m: db.get_nebular_continuum_AGN('NLR', m,self.logU)
                                    for m in metallicities}
                self.lines_template_BLR = {m: db.get_nebular_lines_AGN('BLR', m, self.logU)
                                    for m in metallicities}
                self.cont_template_BLR = {m: db.get_nebular_continuum_AGN('BLR', m,self.logU)
                                    for m in metallicities}

            self.linesdict_NLR = {m: dict(zip(self.lines_template_NLR[m].name,
                                          zip(self.lines_template_NLR[m].wave,
                                              self.lines_template_NLR[m].ratio)))
                              for m in metallicities}
            self.linesdict_BLR = {m: dict(zip(self.lines_template_BLR[m].name,
                                          zip(self.lines_template_BLR[m].wave,
                                              self.lines_template_BLR[m].ratio)))
                              for m in metallicities}

            for lines in self.lines_template_NLR.values():
                new_wave = np.array([])
                for line_wave in lines.wave:
                    width = line_wave * self.lines_width_NLR * 1e3 / cst.c
                    new_wave = np.concatenate((new_wave,
                                            np.linspace(line_wave - 3. * width,
                                                        line_wave + 3. * width,
                                                        9)))
                new_wave.sort()
                new_flux = np.zeros_like(new_wave)
                for line_flux, line_wave in zip(lines.ratio, lines.wave):
                    width = line_wave * self.lines_width_NLR * 1e3 / cst.c
                    new_flux += (line_flux * np.exp(- 4. * np.log(2.) *
                                (new_wave - line_wave) ** 2. / (width * width)) /
                                (width * np.sqrt(np.pi / np.log(2.)) / 2.))
                lines.wave = new_wave
                lines.ratio = new_flux* self.f_NLR

            for lines in self.lines_template_BLR.values():
                new_wave = np.array([])
                for line_wave in lines.wave:
                    width = line_wave * self.lines_width_BLR * 1e3 / cst.c
                    new_wave = np.concatenate((new_wave,
                                            np.linspace(line_wave - 3. * width,
                                                        line_wave + 3. * width,
                                                        9)))
                new_wave.sort()
                new_flux = np.zeros_like(new_wave)
                for line_flux, line_wave in zip(lines.ratio, lines.wave):
                    width = line_wave * self.lines_width_BLR * 1e3 / cst.c
                    new_flux += (line_flux * np.exp(- 4. * np.log(2.) *
                                (new_wave - line_wave) ** 2. / (width * width)) /
                                (width * np.sqrt(np.pi / np.log(2.)) / 2.))
                lines.wave = new_wave
                lines.ratio = new_flux* self.f_BLR

            for NLR,BLR in zip(self.cont_template_NLR.values(), self.cont_template_BLR.values()):
                NLR.lumin *= self.f_NLR
                BLR.lumin *= self.f_BLR
          
            
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
        

        sed.add_module(self.name, self.parameters)
             

        if self.emission:
           
            #NLy = sed.info['AGN.n_ly'] # to be written
            NLy = 1.
            metallicity = self.metallicity
            lines_NLR = self.lines_template_NLR[metallicity]
            lines_BLR = self.lines_template_BLR[metallicity]
            linesdict_NLR = self.linesdict_NLR[metallicity]
            linesdict_BLR = self.linesdict_BLR[metallicity]
            cont_NLR = self.cont_template_NLR[metallicity]
            cont_BLR = self.cont_template_BLR[metallicity]

            sed.add_info('AGN.nebular.lines_width_NLR', self.lines_width_NLR, unit='km/s')
            sed.add_info('AGN.nebular.lines_width_BLR', self.lines_width_BLR, unit='km/s')
            sed.add_info('AGN.nebular.logU', self.logU)

            for line in default_lines:
                wave_NLR, ratio_NLR = linesdict_NLR[line]
                wave_BLR, ratio_BLR = linesdict_BLR[line]
                sed.lines_AGN[line] = (wave_NLR,
                                   ratio_NLR * NLy,
                                   ratio_BLR * NLy)

        

            sed.add_contribution('nebular.lines_NLR', lines_NLR.wave,
                                 lines_NLR.ratio * NLy)
            sed.add_contribution('nebular.lines_BLR', lines_BLR.wave,
                                 lines_BLR.ratio * NLy)

            
            
            sed.add_contribution('nebular.continuum_NLR', cont_NLR.wave,
                                 cont_NLR.lumin * NLy)
            sed.add_contribution('nebular.continuum_BLR', cont_BLR.wave,
                                 cont_BLR.lumin * NLy)


# SedModule to be returned by get_module
Module = NebularEmissionAGN
