# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

import numpy as np


class M2005:
    """Single Stellar Population as defined in Maraston (2005)

    This class holds the data associated with a single stellar population
    (SSP) as defined in Maraston (2005). Compared to the pristine Maraston's
    SSP:
        * The time grid ranges from 1 Myr to 13.7 Gyr with 1 Myr steps. This
        excludes the metallicities -2.25 and 0.67 for which the age grid starts
        at 1 Gyr.
        * The SSP are all interpolated on this new grid.
        * The wavelength is given in nm rather than Å.
        * The spectra are given in W/nm rather than erg/s.

     """

    def __init__(self, imf, metallicity, time_grid, wavelength_grid,
                 info_table, spec_table):
        """Create a new single stellar population as defined in Maraston 2005.

        Parameters
        ----------
        imf: string
            Initial mass function (IMF): either 'salp' for single Salpeter
            (1955) or 'krou' for Kroupa (2001).
        metallicity: float
            The metallicity. Possible values are 0.001, 0.01, 0.02, and 0.04.
        time_grid: array of floats
            The time grid in Myr used in the info_table and the spec_table.
        wavelength_grid: array of floats
            The wavelength grid in nm used in spec_table.
        info_table: (6, n) array of floats
            The 2D table giving the various masses at a given age. The
            first axis is the king of mass, the second is the age based on the
            time_grid.
                * info_table[0]: total mass
                * info_table[1]: alive stellar mass
                * info_table[2]: white dwarf stars mass
                * info_table[3]: neutron stars mass
                * info_table[4]: black holes mass
                * info_table[5]: turn-off mass
        spec_table: 2D array of floats
            Spectrum of the SSP in W/nm (first axis) every 1 Myr (second axis).

        """

        if imf in ['salp', 'krou']:
            self.imf = imf
        else:
            raise ValueError("IMF must be either salp for Salpeter or krou "
                             "for Kroupa.")
        self.metallicity = metallicity
        self.time_grid = time_grid
        self.wavelength_grid = wavelength_grid
        self.info_table = info_table
        self.spec_table = spec_table

    def convolve(self, sfh, separation_age):
        """Convolve the SSP with a Star Formation History

        Convolves the SSP and the associated info with the SFH.

        Parameters
        ----------
        sfh: array of floats
            Star Formation History in Msun/yr.
        separation_age: float
            Age separating the young from the old stellar populations in Myr.

        Returns
        -------
        spec_young: array of floats
            Spectrum in W/nm of the young stellar populations.
        spec_old: array of floats
            Same as spec_young but for the old stellar populations.
        info_young: array of floats
            Contains the info for the young stellar populations:
            * 0: total stellar mass
            * 1: alive stellar mass
            * 2: white dwarf stars mass
            * 3: neutron stars mass
            * 4: black holes mass
        info_old: array of floats
            Same as info_young but for the old stellar populations.
        info_all: array of floats
            Same as info_young but for the entire stellar population. Also
            contains 5: stellar mass-weighted age

        """
        # We cut the SSP to the maximum age considered to simplify the
        # computation.
        info_table = self.info_table[:5, :sfh.size]
        spec_table = self.spec_table[:, :sfh.size]

        # As both the SFH and the SSP (limited to the age of the SFH) data now
        # share the same time grid, the convolution is just a matter of
        # reverting one and computing the sum of the one to one product; this
        # is done using the dot product. The 1e6 factor is because the SFH is
        # in solar mass per year.
        info_young = 1e6 * np.dot(info_table[:, :separation_age],
                                  sfh[-separation_age:][::-1])
        spec_young = 1e6 * np.dot(spec_table[:, :separation_age],
                                  sfh[-separation_age:][::-1])

        info_old = 1e6 * np.dot(info_table[:, separation_age:],
                                sfh[:-separation_age][::-1])
        spec_old = 1e6 * np.dot(spec_table[:, separation_age:],
                                sfh[:-separation_age][::-1])

        info_all = info_young + info_old
        info_all = np.append(info_all, np.average(self.time_grid[:sfh.size],
                                                  weights=info_table[1, :] *
                                                  sfh[::-1]))

        return spec_young, spec_old, info_young, info_old, info_all
