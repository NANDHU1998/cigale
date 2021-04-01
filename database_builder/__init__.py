# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Authors: Yannick Roehlly, Médéric Boquien, Laure Ciesla

"""
This script is used the build pcigale internal database containing:
- The various filter transmission tables;
- The Maraston 2005 single stellar population (SSP) data;
- The Dale and Helou 2002 infra-red templates.

"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import glob
import io
import itertools
import numpy as np
from scipy import interpolate
import scipy.constants as cst
from astropy.table import Table
from pcigale.data import (Database, Filter, M2005, BC03, Fritz2006,
                          Dale2014, DL2007, DL2014, NebularLines,
                          NebularContinuum, NebularLines_AGN, NebularContinuum_AGN, SKIRTOR2016, Schreiber2016, THEMIS)


def read_bc03_ssp(filename):
    """Read a Bruzual and Charlot 2003 ASCII SSP file

    The ASCII SSP files of Bruzual and Charlot 2003 have se special structure.
    A vector is stored with the number of values followed by the values
    separated by a space (or a carriage return). There are the time vector, 5
    (for Chabrier IMF) or 6 lines (for Salpeter IMF) that we don't care of,
    then the wavelength vector, then the luminosity vectors, each followed by
    a 52 value table, then a bunch of other table of information that are also
    in the *colors files.

    Parameters
    ----------
    filename : string

    Returns
    -------
    time_grid: numpy 1D array of floats
              Vector of the time grid of the SSP in Myr.
    wavelength: numpy 1D array of floats
                Vector of the wavelength grid of the SSP in nm.
    spectra: numpy 2D array of floats
             Array containing the SSP spectra, first axis is the wavelength,
             second one is the time.

    """

    def file_structure_generator():
        """Generator used to identify table lines in the SSP file

        In the SSP file, the vectors are store one next to the other, but
        there are 5 informational lines after the time vector. We use this
        generator to the if we are on lines to read or not.
        """
        if "chab" in filename:
            bad_line_number = 5
        else:
            bad_line_number = 6
        yield("data")
        for i in range(bad_line_number):
            yield("bad")
        while True:
            yield("data")

    file_structure = file_structure_generator()
    # Are we in a data line or a bad one.
    what_line = next(file_structure)
    # Variable conting, in reverse order, the number of value still to
    # read for the read vector.
    counter = 0

    time_grid = []
    full_table = []
    tmp_table = []

    with open(filename) as file_:
        # We read the file line by line.
        for line in file_:
            if what_line == "data":
                # If we are in a "data" line, we analyse each number.
                for item in line.split():
                    if counter == 0:
                        # If counter is 0, then we are not reading a vector
                        # and the first number is the length of the next
                        # vector.
                        counter = int(item)
                    else:
                        # If counter > 0, we are currently reading a vector.
                        tmp_table.append(float(item))
                        counter -= 1
                        if counter == 0:
                            # We reached the end of the vector. If we have not
                            # yet store the time grid (the first table) we are
                            # currently reading it.
                            if time_grid == []:
                                time_grid = tmp_table[:]
                            # Else, we store the vector in the full table,
                            # only if its length is superior to 250 to get rid
                            # of the 52 item unknown vector and the 221 (time
                            # grid length) item vectors at the end of the
                            # file.
                            elif len(tmp_table) > 250:
                                full_table.append(tmp_table[:])

                            tmp_table = []

            # If at the end of a line, we have finished reading a vector, it's
            # time to change to the next structure context.
            if counter == 0:
                what_line = next(file_structure)

    # The time grid is in year, we want Myr.
    time_grid = np.array(time_grid, dtype=float)
    time_grid *= 1.e-6

    # The first "long" vector encountered is the wavelength grid. The value
    # are in Ångström, we convert it to nano-meter.
    wavelength = np.array(full_table.pop(0), dtype=float)
    wavelength *= 0.1

    # The luminosities are in Solar luminosity (3.826.10^33 ergs.s-1) per
    # Ångström, we convert it to W/nm.
    luminosity = np.array(full_table, dtype=float)
    luminosity *= 3.826e27
    # Transposition to have the time in the second axis.
    luminosity = luminosity.transpose()

    # In the SSP, the time grid begins at 0, but not in the *colors file, so
    # we remove t=0 from the SSP.
    return time_grid[1:], wavelength, luminosity[:, 1:]


def build_filters(base):
    filters = []
    filters_dir = os.path.join(os.path.dirname(__file__), 'filters/')
    for filter_file in glob.glob(filters_dir + '**/*.dat', recursive=True):
        with open(filter_file, 'r') as filter_file_read:
            filter_name = filter_file_read.readline().strip('# \n\t')
            filter_type = filter_file_read.readline().strip('# \n\t')
            filter_description = filter_file_read.readline().strip('# \n\t')

        # Make the name dynamic for filters in subdirectories
        tmp_name = filter_file.replace(filters_dir, '')[:-4]
        if '/' in tmp_name:
            filter_name = tmp_name.replace('/', '.')
        filter_table = np.genfromtxt(filter_file)
        # The table is transposed to have table[0] containing the wavelength
        # and table[1] containing the transmission.
        filter_table = filter_table.transpose()

        # We convert the wavelength from Å to nm.
        filter_table[0] *= 0.1

        # We convert to energy if needed
        if filter_type == 'photon':
            filter_table[1] *= filter_table[0]
        elif filter_type != 'energy':
            raise ValueError("Filter transmission type can only be "
                             "'energy' or 'photon'.")

        print("Importing %s... (%s points)" % (filter_name,
                                               filter_table.shape[1]))

        new_filter = Filter(filter_name, filter_description, filter_table)

        # We normalise the filter and compute the pivot wavelength. If the
        # filter is a pseudo-filter used to compute line fluxes, it should not
        # be normalised.
        if not (filter_name.startswith('PSEUDO') or
                filter_name.startswith('linefilter')):
            new_filter.normalise()
        else:
            new_filter.pivot_wavelength = np.mean(
                filter_table[0][filter_table[1] > 0]
            )
        filters.append(new_filter)

    base.add_filters(filters)

def build_filters_gazpar(base):
    filters = []
    filters_dir = os.path.join(os.path.dirname(__file__), 'filters_gazpar/')
    for filter_file in glob.glob(filters_dir + '**/*.pb', recursive=True):
        with open(filter_file, 'r') as filter_file_read:
            _ = filter_file_read.readline() # We use the filename for the name
            filter_type = filter_file_read.readline().strip('# \n\t')
            _ = filter_file_read.readline() # We do not yet use the calib type
            filter_desc = filter_file_read.readline().strip('# \n\t')

        filter_name = filter_file.replace(filters_dir, '')[:-3]
        filter_name = filter_name.replace('/', '.')

        filter_table = np.genfromtxt(filter_file)
        # The table is transposed to have table[0] containing the wavelength
        # and table[1] containing the transmission.
        filter_table = filter_table.transpose()

        # We convert the wavelength from Å to nm.
        filter_table[0] *= 0.1

        # We convert to energy if needed
        if filter_type == 'photon':
            filter_table[1] *= filter_table[0]
        elif filter_type != 'energy':
            raise ValueError("Filter transmission type can only be "
                             "'energy' or 'photon'.")

        print("Importing %s... (%s points)" % (filter_name,
                                               filter_table.shape[1]))

        new_filter = Filter(filter_name, filter_desc, filter_table)

        # We normalise the filter and compute the pivot wavelength. If the
        # filter is a pseudo-filter used to compute line fluxes, it should not
        # be normalised.
        if not filter_name.startswith('PSEUDO'):
            new_filter.normalise()
        else:
            new_filter.pivot_wavelength = np.mean(
                filter_table[0][filter_table[1] > 0]
            )
        filters.append(new_filter)

    base.add_filters(filters)

def build_m2005(base):
    m2005_dir = os.path.join(os.path.dirname(__file__), 'maraston2005/')

    # Age grid (1 Myr to 13.7 Gyr with 1 Myr step)
    time_grid = np.arange(1, 13701)
    fine_time_grid = np.linspace(0.1, 13700, 137000)

    # Transpose the table to have access to each value vector on the first
    # axis
    kroupa_mass = np.genfromtxt(m2005_dir + 'stellarmass.kroupa').transpose()
    salpeter_mass = \
        np.genfromtxt(m2005_dir + '/stellarmass.salpeter').transpose()

    for spec_file in glob.glob(m2005_dir + '*.rhb'):

        print("Importing %s..." % spec_file)

        spec_table = np.genfromtxt(spec_file).transpose()
        metallicity = spec_table[1, 0]

        if 'krz' in spec_file:
            imf = 'krou'
            mass_table = np.copy(kroupa_mass)
        elif 'ssz' in spec_file:
            imf = 'salp'
            mass_table = np.copy(salpeter_mass)
        else:
            raise ValueError('Unknown IMF!!!')

        # Keep only the actual metallicity values in the mass table
        # we don't take the first column which contains metallicity.
        # We also eliminate the turn-off mas which makes no send for composite
        # populations.
        mass_table = mass_table[1:7, mass_table[0] == metallicity]

        # Regrid the SSP data to the evenly spaced time grid. In doing so we
        # assume 10 bursts every 0.1 Myr over a period of 1 Myr in order to
        # capture short evolutionary phases.
        # The time grid starts after 0.1 Myr, so we assume the value is the same
        # as the first actual time step.
        mass_table = interpolate.interp1d(mass_table[0] * 1e3, mass_table[1:],
                                          assume_sorted=True)(fine_time_grid)
        mass_table = np.mean(mass_table.reshape(5, -1, 10), axis=-1)

        # Extract the age and convert from Gyr to Myr
        ssp_time = np.unique(spec_table[0]) * 1e3
        spec_table = spec_table[1:]

        # Remove the metallicity column from the spec table
        spec_table = spec_table[1:]

        # Extract the wavelength and convert from Å to nm
        ssp_wave = spec_table[0][:1221] * 0.1
        spec_table = spec_table[1:]

        # Extra the fluxes and convert from erg/s/Å to W/nm
        ssp_lumin = spec_table[0].reshape(ssp_time.size, ssp_wave.size).T
        ssp_lumin *= 10 * 1e-7

        # We have to do the interpolation-averaging in several blocks as it is
        # a bit RAM intensive
        ssp_lumin_interp = np.empty((ssp_wave.size, time_grid.size))
        for i in range(0, ssp_wave.size, 100):
            fill_value = (ssp_lumin[i:i+100, 0], ssp_lumin[i:i+100, -1])
            ssp_interp = interpolate.interp1d(ssp_time, ssp_lumin[i:i+100, :],
                                              fill_value=fill_value,
                                              bounds_error=False,
                                              assume_sorted=True)(fine_time_grid)
            ssp_interp = ssp_interp.reshape(ssp_interp.shape[0], -1, 10)
            ssp_lumin_interp[i:i+100, :] = np.mean(ssp_interp, axis=-1)

        # To avoid the creation of waves when interpolating, we refine the grid
        # beyond 10 μm following a log scale in wavelength. The interpolation
        # is also done in log space as the spectrum is power-law-like
        ssp_wave_resamp = np.around(np.logspace(np.log10(10000),
                                                   np.log10(160000), 50))
        argmin = np.argmin(10000.-ssp_wave > 0)-1
        ssp_lumin_resamp = 10.**interpolate.interp1d(
                                    np.log10(ssp_wave[argmin:]),
                                    np.log10(ssp_lumin_interp[argmin:, :]),
                                    assume_sorted=True,
                                    axis=0)(np.log10(ssp_wave_resamp))

        ssp_wave = np.hstack([ssp_wave[:argmin+1], ssp_wave_resamp])
        ssp_lumin = np.vstack([ssp_lumin_interp[:argmin+1, :],
                               ssp_lumin_resamp])

        # Use Z value for metallicity, not log([Z/H])
        metallicity = {-1.35: 0.001,
                       -0.33: 0.01,
                       0.0: 0.02,
                       0.35: 0.04}[metallicity]

        base.add_m2005(M2005(imf, metallicity, time_grid, ssp_wave,
                             mass_table, ssp_lumin))


def build_bc2003(base, res):
    bc03_dir = os.path.join(os.path.dirname(__file__), 'bc03/')

    # Time grid (1 Myr to 14 Gyr with 1 Myr step)
    time_grid = np.arange(1, 14000)
    fine_time_grid = np.linspace(0.1, 13999, 139990)

    # Metallicities associated to each key
    metallicity = {
        "m22": 0.0001,
        "m32": 0.0004,
        "m42": 0.004,
        "m52": 0.008,
        "m62": 0.02,
        "m72": 0.05
    }

    for key, imf in itertools.product(metallicity, ["salp", "chab"]):
        ssp_filename = "{}bc2003_{}_{}_{}_ssp.ised_ASCII".format(bc03_dir, res,
                                                                 key, imf)
        color3_filename = "{}bc2003_lr_{}_{}_ssp.3color".format(bc03_dir, key,
                                                                imf)
        color4_filename = "{}bc2003_lr_{}_{}_ssp.4color".format(bc03_dir, key,
                                                                imf)

        print("Importing {}...".format(ssp_filename))

        # Read the desired information from the color files
        color_table = []
        color3_table = np.genfromtxt(color3_filename).transpose()
        color4_table = np.genfromtxt(color4_filename).transpose()
        color_table.append(color4_table[6])        # Mstar
        color_table.append(color4_table[7])        # Mgas
        color_table.append(10 ** color3_table[5])  # NLy

        color_table = np.array(color_table)

        ssp_time, ssp_wave, ssp_lumin = read_bc03_ssp(ssp_filename)

        # Regrid the SSP data to the evenly spaced time grid. In doing so we
        # assume 10 bursts every 0.1 Myr over a period of 1 Myr in order to
        # capture short evolutionary phases.
        # The time grid starts after 0.1 Myr, so we assume the value is the same
        # as the first actual time step.
        fill_value = (color_table[:, 0], color_table[:, -1])
        color_table = interpolate.interp1d(ssp_time, color_table,
                                           fill_value=fill_value,
                                           bounds_error=False,
                                           assume_sorted=True)(fine_time_grid)
        color_table = np.mean(color_table.reshape(3, -1, 10), axis=-1)

        # We have to do the interpolation-averaging in several blocks as it is
        # a bit RAM intensive
        ssp_lumin_interp = np.empty((ssp_wave.size, time_grid.size))
        for i in range(0, ssp_wave.size, 100):
            fill_value = (ssp_lumin[i:i+100, 0], ssp_lumin[i:i+100, -1])
            ssp_interp = interpolate.interp1d(ssp_time, ssp_lumin[i:i+100, :],
                                              fill_value=fill_value,
                                              bounds_error=False,
                                              assume_sorted=True)(fine_time_grid)
            ssp_interp = ssp_interp.reshape(ssp_interp.shape[0], -1, 10)
            ssp_lumin_interp[i:i+100, :] = np.mean(ssp_interp, axis=-1)

        # To avoid the creation of waves when interpolating, we refine the grid
        # beyond 10 μm following a log scale in wavelength. The interpolation
        # is also done in log space as the spectrum is power-law-like
        ssp_wave_resamp = np.around(np.logspace(np.log10(10000),
                                                np.log10(160000), 50))
        argmin = np.argmin(10000.-ssp_wave > 0)-1
        ssp_lumin_resamp = 10.**interpolate.interp1d(
                                    np.log10(ssp_wave[argmin:]),
                                    np.log10(ssp_lumin_interp[argmin:, :]),
                                    assume_sorted=True,
                                    axis=0)(np.log10(ssp_wave_resamp))

        ssp_wave = np.hstack([ssp_wave[:argmin+1], ssp_wave_resamp])
        ssp_lumin = np.vstack([ssp_lumin_interp[:argmin+1, :],
                               ssp_lumin_resamp])

        base.add_bc03(BC03(
            imf,
            metallicity[key],
            time_grid,
            ssp_wave,
            color_table,
            ssp_lumin
        ))


def build_dale2014(base):
    models = []
    dale2014_dir = os.path.join(os.path.dirname(__file__), 'dale2014/')

    # Getting the alpha grid for the templates
    d14cal = np.genfromtxt(dale2014_dir + 'dhcal.dat')
    alpha_grid = d14cal[:, 1]

    # Getting the lambda grid for the templates and convert from microns to nm.
    first_template = np.genfromtxt(dale2014_dir + 'spectra.0.00AGN.dat')
    wave = first_template[:, 0] * 1E3

    # Getting the stellar emission and interpolate it at the same wavelength
    # grid
    stell_emission_file = np.genfromtxt(dale2014_dir +
                                        'stellar_SED_age13Gyr_tau10Gyr.spec')
    # A -> to nm
    wave_stell = stell_emission_file[:, 0] * 0.1
    # W/A -> W/nm
    stell_emission = stell_emission_file[:, 1] * 10
    stell_emission_interp = np.interp(wave, wave_stell, stell_emission)

    # The models are in nuFnu and contain stellar emission.
    # We convert this to W/nm and remove the stellar emission.

    # Emission from dust heated by SB
    fraction = 0.0
    filename = dale2014_dir + "spectra.0.00AGN.dat"
    print("Importing {}...".format(filename))
    datafile = open(filename)
    data = "".join(datafile.readlines())
    datafile.close()

    for al in range(1, len(alpha_grid)+1, 1):
        lumin_with_stell = np.genfromtxt(io.BytesIO(data.encode()),
                                         usecols=(al))
        lumin_with_stell = pow(10, lumin_with_stell) / wave
        constant = lumin_with_stell[7] / stell_emission_interp[7]
        lumin = lumin_with_stell - stell_emission_interp * constant
        lumin[lumin < 0] = 0
        lumin[wave < 2E3] = 0
        norm = np.trapz(lumin, x=wave)
        lumin /= norm

        models.append(Dale2014(fraction, alpha_grid[al-1], wave, lumin))
    # Emission from dust heated by AGN - Quasar template
    filename = dale2014_dir + "shi_agn.regridded.extended.dat"
    print("Importing {}...".format(filename))

    wave, lumin_quasar = np.genfromtxt(filename, unpack=True)
    wave *= 1e3
    lumin_quasar = 10**lumin_quasar / wave
    norm = np.trapz(lumin_quasar, x=wave)
    lumin_quasar /= norm

    models.append(Dale2014(1.0, 0.0, wave, lumin_quasar))

    base.add_dale2014(models)


def build_dl2007(base):
    models = []
    dl2007_dir = os.path.join(os.path.dirname(__file__), 'dl2007/')

    qpah = {
        "00": 0.47,
        "10": 1.12,
        "20": 1.77,
        "30": 2.50,
        "40": 3.19,
        "50": 3.90,
        "60": 4.58
    }

    umaximum = ["1e3", "1e4", "1e5", "1e6"]
    uminimum = ["0.10", "0.15", "0.20", "0.30", "0.40", "0.50", "0.70",
                "0.80", "1.00", "1.20", "1.50", "2.00", "2.50", "3.00",
                "4.00", "5.00", "7.00", "8.00", "10.0", "12.0", "15.0",
                "20.0", "25.0"]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {"00": 0.0100, "10": 0.0100, "20": 0.0101, "30": 0.0102,
            "40": 0.0102, "50": 0.0103, "60": 0.0104}

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    datafile = open(dl2007_dir + "U{}/U{}_{}_MW3.1_{}.txt".format(umaximum[0],
                                                                  umaximum[0],
                                                                  umaximum[0],
                                                                  "00"))
    data = "".join(datafile.readlines()[-1001:])
    datafile.close()

    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # For some reason wavelengths are decreasing in the model files
    wave = wave[::-1]
    # We convert wavelengths from μm to nm
    wave *= 1000.

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4. * np.pi * 1e-30 / (cst.m_p+cst.m_e) * cst.c / (wave*wave) * 1e9

    for model in sorted(qpah.keys()):
        for umin in uminimum:
            filename = dl2007_dir + "U{}/U{}_{}_MW3.1_{}.txt".format(umin,
                                                                     umin,
                                                                     umin,
                                                                     model)
            print("Importing {}...".format(filename))
            datafile = open(filename)
            data = "".join(datafile.readlines()[-1001:])
            datafile.close()
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            # For some reason fluxes are decreasing in the model files
            lumin = lumin[::-1]
            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv/MdMH[model]

            models.append(DL2007(qpah[model], umin, umin, wave, lumin))
            for umax in umaximum:
                filename = dl2007_dir + "U{}/U{}_{}_MW3.1_{}.txt".format(umin,
                                                                         umin,
                                                                         umax,
                                                                         model)
                print("Importing {}...".format(filename))
                datafile = open(filename)
                data = "".join(datafile.readlines()[-1001:])
                datafile.close()
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                # For some reason fluxes are decreasing in the model files
                lumin = lumin[::-1]

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv/MdMH[model]

                models.append(DL2007(qpah[model], umin, umax, wave, lumin))
    base.add_dl2007(models)


def build_dl2014(base):
    models = []
    dl2014_dir = os.path.join(os.path.dirname(__file__), 'dl2014/')

    qpah = {"000": 0.47, "010": 1.12, "020": 1.77, "030": 2.50, "040": 3.19,
            "050": 3.90, "060": 4.58, "070": 5.26, "080": 5.95, "090": 6.63,
            "100": 7.32}

    uminimum = ["0.100", "0.120", "0.150", "0.170", "0.200", "0.250", "0.300",
                "0.350", "0.400", "0.500", "0.600", "0.700", "0.800", "1.000",
                "1.200", "1.500", "1.700", "2.000", "2.500", "3.000", "3.500",
                "4.000", "5.000", "6.000", "7.000", "8.000", "10.00", "12.00",
                "15.00", "17.00", "20.00", "25.00", "30.00", "35.00", "40.00",
                "50.00"]

    alpha = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8",
             "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7",
             "2.8", "2.9", "3.0"]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {"000": 0.0100, "010": 0.0100, "020": 0.0101, "030": 0.0102,
            "040": 0.0102, "050": 0.0103, "060": 0.0104, "070": 0.0105,
            "080": 0.0106, "090": 0.0107, "100": 0.0108}

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    datafile = open(dl2014_dir + "U{}_{}_MW3.1_{}/spec_1.0.dat"
                    .format(uminimum[0], uminimum[0], "000"))

    data = "".join(datafile.readlines()[-1001:])
    datafile.close()

    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # For some reason wavelengths are decreasing in the model files
    wave = wave[::-1]
    # We convert wavelengths from μm to nm
    wave *= 1000.

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4. * np.pi * 1e-30 / (cst.m_p+cst.m_e) * cst.c / (wave*wave) * 1e9

    for model in sorted(qpah.keys()):
        for umin in uminimum:
            filename = (dl2014_dir + "U{}_{}_MW3.1_{}/spec_1.0.dat"
                        .format(umin, umin, model))
            print("Importing {}...".format(filename))
            with open(filename) as datafile:
                data = "".join(datafile.readlines()[-1001:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            # For some reason fluxes are decreasing in the model files
            lumin = lumin[::-1]

            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv/MdMH[model]

            models.append(DL2014(qpah[model], umin, umin, 1.0, wave, lumin))
            for al in alpha:
                filename = (dl2014_dir + "U{}_1e7_MW3.1_{}/spec_{}.dat"
                            .format(umin, model, al))
                print("Importing {}...".format(filename))
                with open(filename) as datafile:
                    data = "".join(datafile.readlines()[-1001:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                # For some reason fluxes are decreasing in the model files
                lumin = lumin[::-1]

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv/MdMH[model]

                models.append(DL2014(qpah[model], umin, 1e7, al, wave, lumin))

    base.add_dl2014(models)

def build_fritz2006(base):
    models = []
    fritz2006_dir = os.path.join(os.path.dirname(__file__), 'fritz2006/')

    # Parameters of Fritz+2006
    psy = [0.001, 10.100, 20.100, 30.100, 40.100, 50.100, 60.100, 70.100,
           80.100, 89.990]  # Viewing angle in degrees
    opening_angle = ["20", "40", "60"]  # Theta = 2*(90 - opening_angle)
    gamma = ["0.0", "2.0", "4.0", "6.0"]
    beta = ["-1.00", "-0.75", "-0.50", "-0.25", "0.00"]
    tau = ["0.1", "0.3", "0.6", "1.0", "2.0", "3.0", "6.0", "10.0"]
    r_ratio = ["10", "30", "60", "100", "150"]

    # Read and convert the wavelength
    datafile = open(fritz2006_dir + "ct{}al{}be{}ta{}rm{}.tot"
                    .format(opening_angle[0], gamma[0], beta[0], tau[0],
                            r_ratio[0]))
    data = "".join(datafile.readlines()[-178:])
    datafile.close()
    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    wave *= 1e3
    # Number of wavelengths: 178; Number of comments lines: 28
    nskip = 28
    blocksize = 178

    iter_params = ((oa, gam, be, ta, rm)
                   for oa in opening_angle
                   for gam in gamma
                   for be in beta
                   for ta in tau
                   for rm in r_ratio)

    for params in iter_params:
        filename = fritz2006_dir + "ct{}al{}be{}ta{}rm{}.tot".format(*params)
        print("Importing {}...".format(filename))
        try:
            datafile = open(filename)
        except IOError:
            continue
        data = datafile.readlines()
        datafile.close()

        for n in range(len(psy)):
            block = data[nskip + blocksize * n + 4 * (n + 1) - 1:
                         nskip + blocksize * (n+1) + 4 * (n + 1) - 1]
            lumin_therm, lumin_scatt, lumin_agn = np.genfromtxt(
                io.BytesIO("".join(block).encode()), usecols=(2, 3, 4),
                unpack=True)
            # Remove NaN
            lumin_therm = np.nan_to_num(lumin_therm)
            lumin_scatt = np.nan_to_num(lumin_scatt)
            lumin_agn = np.nan_to_num(lumin_agn)
            # Conversion from erg/s/microns to W/nm
            lumin_therm *= 1e-4
            lumin_scatt *= 1e-4
            lumin_agn *= 1e-4
            # Normalization of the lumin_therm to 1W
            norm = np.trapz(lumin_therm, x=wave)
            lumin_therm /= norm
            lumin_scatt /= norm
            lumin_agn /= norm

            models.append(Fritz2006(params[4], params[3], params[2],
                                         params[1], params[0], psy[n], wave,
                                         lumin_therm, lumin_scatt, lumin_agn))

    base.add_fritz2006(models)

def build_skirtor2016(base):
    models = []
    skirtor2016_dir = os.path.join(os.path.dirname(__file__), 'skirtor2016/')

    files = glob.glob(skirtor2016_dir + '/*')
    files = [file.split('/')[-1] for file in files]
    params = [f.split('_')[:-1] for f in files]

    # Parameters of SKIRTOR 2016
    t = list({param[0][1:] for param in params})
    p = list({param[1][1:] for param in params})
    q = list({param[2][1:] for param in params})
    oa = list({param[3][2:] for param in params})
    R = list({param[4][1:] for param in params})
    Mcl = list({param[5][3:] for param in params})
    i = list({param[6][1:] for param in params})

    iter_params = ((p1, p2, p3, p4, p5, p6, p7)
                   for p1 in t
                   for p2 in p
                   for p3 in q
                   for p4 in oa
                   for p5 in R
                   for p6 in Mcl
                   for p7 in i)

    for params in iter_params:
        filename = skirtor2016_dir + \
                "t{}_p{}_q{}_oa{}_R{}_Mcl{}_i{}_sed.dat".format(*params)
        print("Importing {}...".format(filename))

        wl, disk, scatt, dust = np.genfromtxt(filename, unpack=True,
                                              usecols=(0, 2, 3, 4))
        wl *= 1e3
        disk += scatt
        disk /= wl
        dust /= wl

        # Normalization of the lumin_therm to 1W
        norm = np.trapz(dust, x=wl)
        disk /= norm
        dust /= norm

        models.append(SKIRTOR2016(params[0], params[1], params[2], params[3],
                                  params[4], params[5], params[6], norm, wl,
                                  disk, dust))

    base.add_skirtor2016(models)

def build_nebular(base):
    models_lines = []
    models_cont = []

    nebular_dir = os.path.join(os.path.dirname(__file__), 'nebular/')
    print("Importing {}...".format(nebular_dir + 'lines.dat'))
    lines = np.genfromtxt(nebular_dir + 'lines.dat')

    tmp = Table.read(nebular_dir + 'line_wavelengths.dat', format='ascii')
    wave_lines = tmp['col1'].data
    name_lines = tmp['col2'].data

    print("Importing {}...".format(nebular_dir + 'continuum.dat'))
    cont = np.genfromtxt(nebular_dir + 'continuum.dat')

    # Convert wavelength from Å to nm
    wave_lines *= 0.1
    wave_cont = cont[:3729, 0] * 0.1

    # Get the list of metallicities
    metallicities = np.unique(lines[:, 1])

    # Keep only the fluxes
    lines = lines[:, 2:]
    cont = cont[:, 1:]

    # We select only models with ne=100. Other values could be included later
    lines = lines[:, 1::3]
    cont = cont[:, 1::3]

    # Convert lines to W and to a linear scale
    lines = 10**(lines-7)

    # Convert continuum to W/nm
    cont *= np.tile(1e-7 * cst.c * 1e9 / wave_cont**2,
                    metallicities.size)[:, np.newaxis]

    # Import lines
    for idx, metallicity in enumerate(metallicities):
        spectra = lines[idx::6, :]
        for logU, spectrum in zip(np.around(np.arange(-4., -.9, .1), 1),
                                  spectra.T):
            models_lines.append(NebularLines(metallicity, logU, name_lines,
                                             wave_lines, spectrum))

    # Import continuum
    for idx, metallicity in enumerate(metallicities):
        spectra = cont[3729 * idx: 3729 * (idx+1), :]
        for logU, spectrum in zip(np.around(np.arange(-4., -.9, .1), 1),
                                  spectra.T):
            models_cont.append(NebularContinuum(metallicity, logU, wave_cont,
                                                spectrum))

    base.add_nebular_lines(models_lines)
    base.add_nebular_continuum(models_cont)

def build_nebular_AGN(base):
    models_lines = []
    models_cont = []

    nebular_dir = os.path.join(os.path.dirname(__file__), 'nebular_AGN/')
    print("Importing {}...".format(nebular_dir + 'lines_AGN.dat'))
    lines = np.genfromtxt(nebular_dir + 'lines_AGN.dat')

    tmp = Table.read(nebular_dir + 'line_wavelengths_AGN.dat', format='ascii')
    wave_lines = tmp['col1'].data
    name_lines = tmp['col2'].data

    print("Importing {}...".format(nebular_dir + 'continuum_AGN.dat'))
    cont = np.genfromtxt(nebular_dir + 'continuum_AGN.dat')

    # Convert wavelength from Å to nm
    wave_lines *= 0.1
    wave_cont = cont[:3729, 0] * 0.1

    # Get the list of metallicities
    metallicities = np.unique(lines[:, 1])

    # Keep only the fluxes
    lines = lines[:, 2:]
    cont = cont[:, 1:]

    # We select only models with ne=100. Other values could be included later
    lines = lines[:, 1::3]
    cont = cont[:, 1::3]

    # Convert lines to W and to a linear scale
    lines = 10**(lines-7)

    # Convert continuum to W/nm
    cont *= np.tile(1e-7 * cst.c * 1e9 / wave_cont**2,
                    metallicities.size)[:, np.newaxis]

    # Import lines
    for idx, metallicity in enumerate(metallicities):
        spectra = lines[idx::6, :]
        for logU, spectrum in zip(np.around(np.arange(-4., -.9, .1), 1),
                                  spectra.T):
            models_lines.append(NebularLines_AGN(metallicity, logU, name_lines,
                                             wave_lines, spectrum))

    # Import continuum
    for idx, metallicity in enumerate(metallicities):
        spectra = cont[3729 * idx: 3729 * (idx+1), :]
        for logU, spectrum in zip(np.around(np.arange(-4., -.9, .1), 1),
                                  spectra.T):
            models_cont.append(NebularContinuum_AGN(metallicity, logU, wave_cont,
                                                spectrum))

    base.add_nebular_lines_AGN(models_lines)
    base.add_nebular_continuum_AGN(models_cont)


def build_schreiber2016(base):
    models = []
    schreiber2016_dir = os.path.join(os.path.dirname(__file__),
                                     'schreiber2016/')

    print("Importing {}...".format(schreiber2016_dir + 'g15_pah.fits'))
    pah = Table.read(schreiber2016_dir + 'g15_pah.fits')
    print("Importing {}...".format(schreiber2016_dir + 'g15_dust.fits'))
    dust = Table.read(schreiber2016_dir + 'g15_dust.fits')

    # Getting the lambda grid for the templates and convert from μm to nm.
    wave = dust['LAM'][0, 0, :].data * 1e3

    for td in np.arange(15., 100.):
        # Find the closest temperature in the model list of tdust
        tsed = np.argmin(np.absolute(dust['TDUST'][0].data-td))

        # The models are in νFν.  We convert this to W/nm.
        lumin_dust = dust['SED'][0, tsed, :].data / wave
        lumin_pah = pah['SED'][0, tsed, :].data / wave

        models.append(Schreiber2016(0, td, wave, lumin_dust))
        models.append(Schreiber2016(1, td, wave, lumin_pah))

    base.add_schreiber2016(models)


def build_themis(base):
    models = []
    themis_dir = os.path.join(os.path.dirname(__file__), 'themis/')

    # Mass fraction of hydrocarbon solids i.e., a-C(:H) smaller than 1.5 nm,
    # also known as HAC
    qhac = {"000": 0.02, "010": 0.06, "020": 0.10, "030": 0.14, "040": 0.17,
            "050": 0.20, "060": 0.24, "070": 0.28, "080": 0.32, "090": 0.36,
            "100": 0.40}

    uminimum = ["0.100", "0.120", "0.150", "0.170", "0.200", "0.250", "0.300",
                "0.350", "0.400", "0.500", "0.600", "0.700", "0.800", "1.000",
                "1.200", "1.500", "1.700", "2.000", "2.500", "3.000", "3.500",
                "4.000", "5.000", "6.000", "7.000", "8.000", "10.00", "12.00",
                "15.00", "17.00", "20.00", "25.00", "30.00", "35.00", "40.00",
                "50.00", "80.00"]

    alpha = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8",
             "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7",
             "2.8", "2.9", "3.0"]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {"000": 7.4e-3, "010": 7.4e-3, "020": 7.4e-3, "030": 7.4e-3,
            "040": 7.4e-3, "050": 7.4e-3, "060": 7.4e-3, "070": 7.4e-3,
            "080": 7.4e-3, "090": 7.4e-3, "100": 7.4e-3}

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    datafile = open(themis_dir + "U{}_{}_MW3.1_{}/spec_1.0.dat"
                    .format(uminimum[0], uminimum[0], "000"))

    data = "".join(datafile.readlines()[-576:])
    datafile.close()

    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))

    # We convert wavelengths from μm to nm
    wave *= 1000.

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4. * np.pi * 1e-30 / (cst.m_p+cst.m_e) * cst.c / (wave*wave) * 1e9

    for model in sorted(qhac.keys()):
        for umin in uminimum:
            filename = (themis_dir + "U{}_{}_MW3.1_{}/spec_1.0.dat"
                        .format(umin, umin, model))
            print("Importing {}...".format(filename))
            with open(filename) as datafile:
                data = "".join(datafile.readlines()[-576:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))

            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv / MdMH[model]

            models.append(THEMIS(qhac[model], umin, umin, 1.0, wave, lumin))
            for al in alpha:
                filename = (themis_dir + "U{}_1e7_MW3.1_{}/spec_{}.dat"
                            .format(umin, model, al))
                print("Importing {}...".format(filename))
                with open(filename) as datafile:
                    data = "".join(datafile.readlines()[-576:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv/MdMH[model]

                models.append(THEMIS(qhac[model], umin, 1e7, al, wave, lumin))

    base.add_themis(models)


def build_base(bc03res='lr'):
    base = Database(writable=True)
    base.upgrade_base()

    print('#' * 78)
    print("1- Importing filters...\n")
    build_filters(base)
    build_filters_gazpar(base)
    print("\nDONE\n")
    print('#' * 78)

    print("2- Importing Maraston 2005 SSP\n")
    build_m2005(base)
    print("\nDONE\n")
    print('#' * 78)

    print("3- Importing Bruzual and Charlot 2003 SSP\n")
    build_bc2003(base, bc03res)
    print("\nDONE\n")
    print('#' * 78)

    print("4- Importing Draine and Li (2007) models\n")
    build_dl2007(base)
    print("\nDONE\n")
    print('#' * 78)

    print("5- Importing the updated Draine and Li (2007 models)\n")
    build_dl2014(base)
    print("\nDONE\n")
    print('#' * 78)

    print("6- Importing Fritz et al. (2006) models\n")
    build_fritz2006(base)
    print("\nDONE\n")
    print('#' * 78)

    print("7- Importing SKIRTOR 2016 models\n")
    build_skirtor2016(base)
    print("\nDONE\n")
    print('#' * 78)

    print("8- Importing Dale et al (2014) templates\n")
    build_dale2014(base)
    print("\nDONE\n")
    print('#' * 78)

    print("9- Importing nebular lines and continuum\n")
    build_nebular(base)
    print("\nDONE\n")
    print('#' * 78)

    print("10- Importing nebular lines AGN and continuum AGN\n")
    build_nebular_AGN(base)
    print("\nDONE\n")
    print('#' * 78)

    print("11- Importing Schreiber et al (2016) models\n")
    build_schreiber2016(base)
    print("\nDONE\n")
    print('#' * 78)

    print("12- Importing Jones et al (2017) models)\n")
    build_themis(base)
    print("\nDONE\n")
    print('#' * 78)

    base.session.close_all()


if __name__ == '__main__':
    build_base()
