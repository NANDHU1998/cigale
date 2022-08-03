"""This script is used the build pcigale internal database containing."""

import io
import itertools
from pathlib import Path

import numpy as np
from scipy import interpolate
import scipy.constants as cst
from astropy.table import Table

from pcigale.data import SimpleDatabase


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
    filename : Path

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
        if "chab" in filename.stem:
            bad_line_number = 5
        else:
            bad_line_number = 6
        yield ("data")
        for _ in range(bad_line_number):
            yield ("bad")
        while True:
            yield ("data")

    file_structure = file_structure_generator()
    # Are we in a data line or a bad one.
    what_line = next(file_structure)
    # Variable conting, in reverse order, the number of value still to
    # read for the read vector.
    counter = 0

    time_grid = []
    full_table = []
    tmp_table = []

    with filename.open() as file_:
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
    time_grid *= 1.0e-6

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


def build_filters():
    path = Path(__file__).parent / "filters"
    db = SimpleDatabase("filters", writable=True)
    pathlen = len(path.parts)

    for file in path.rglob("*"):
        if file.suffix not in [".dat", ".pb"]:
            continue

        with file.open() as f:
            _ = f.readline()  # We use the filename for the name
            type_ = f.readline().strip("# \n\t")
            if "gazpar" in str(file):
                _ = f.readline()  # We do not use the calib type
                name = ".".join(file.with_suffix("").parts[pathlen + 1 :])
            else:
                name = ".".join(file.with_suffix("").parts[pathlen:])
            desc = f.readline().strip("# \n\t")

        wl, tr = np.genfromtxt(file, unpack=True)

        # We convert the wavelength from Å to nm.
        wl *= 0.1

        # We convert to energy if needed
        if type_ == "photon":
            tr *= wl
        elif type_ != "energy":
            raise ValueError(
                "Filter transmission type can only be 'energy' or " "'photon'."
            )

        print(f"Importing {name}... ({wl.size} points)")

        # We normalise the filter and compute the pivot wavelength. If the
        # filter is a pseudo-filter used to compute line fluxes, it should not
        # be normalised.
        if not name.startswith("PSEUDO"):
            pivot = np.sqrt(np.trapz(tr, wl) / np.trapz(tr / wl**2, wl))

            # The factor 10²⁰ is so that we get the fluxes directly in mJy when
            # we integrate with the wavelength in units of nm and the spectrum
            # in units of W/m²/nm.
            tr *= 1e20 / (cst.c * np.trapz(tr / wl**2, wl))
        else:
            pivot = np.mean(wl[tr > 0.0])
        db.add({"name": name}, {"wl": wl, "tr": tr, "pivot": pivot, "desc": desc})

    db.close()


def build_m2005():
    path = Path(__file__).parent / "maraston2005"
    db = SimpleDatabase("m2005", writable=True)

    # Age grid (1 Myr to 13.7 Gyr with 1 Myr step)
    time_grid = np.arange(1, 13701)
    fine_time_grid = np.linspace(0.1, 13700, 137000)

    # Transpose the table to have access to each value vector on the first
    # axis
    kroupa_mass = np.genfromtxt(path / "stellarmass.kroupa").transpose()
    salpeter_mass = np.genfromtxt(path / "stellarmass.salpeter").transpose()

    for spec_file in path.glob("*.rhb"):
        print(f"Importing {spec_file}...")

        spec_table = np.genfromtxt(spec_file).transpose()
        metallicity = spec_table[1, 0]

        if "krz" in spec_file.stem:
            imf = "krou"
            mass_table = np.copy(kroupa_mass)
        elif "ssz" in spec_file.stem:
            imf = "salp"
            mass_table = np.copy(salpeter_mass)
        else:
            raise ValueError("Unknown IMF!!!")

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
        mass_table = interpolate.interp1d(
            mass_table[0] * 1e3, mass_table[1:], assume_sorted=True
        )(fine_time_grid)
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
            fill_value = (ssp_lumin[i : i + 100, 0], ssp_lumin[i : i + 100, -1])
            ssp_interp = interpolate.interp1d(
                ssp_time,
                ssp_lumin[i : i + 100, :],
                fill_value=fill_value,
                bounds_error=False,
                assume_sorted=True,
            )(fine_time_grid)
            ssp_interp = ssp_interp.reshape(ssp_interp.shape[0], -1, 10)
            ssp_lumin_interp[i : i + 100, :] = np.mean(ssp_interp, axis=-1)

        # To avoid the creation of waves when interpolating, we refine the grid
        # beyond 10 μm following a log scale in wavelength. The interpolation
        # is also done in log space as the spectrum is power-law-like
        ssp_wave_resamp = np.around(np.logspace(np.log10(10000), np.log10(160000), 50))
        argmin = np.argmin(10000.0 - ssp_wave > 0) - 1
        ssp_lumin_resamp = 10.0 ** interpolate.interp1d(
            np.log10(ssp_wave[argmin:]),
            np.log10(ssp_lumin_interp[argmin:, :]),
            assume_sorted=True,
            axis=0,
        )(np.log10(ssp_wave_resamp))

        ssp_wave = np.hstack([ssp_wave[: argmin + 1], ssp_wave_resamp])
        ssp_lumin = np.vstack([ssp_lumin_interp[: argmin + 1, :], ssp_lumin_resamp])

        # Use Z value for metallicity, not log([Z/H])
        metallicity = {-1.35: 0.001, -0.33: 0.01, 0.0: 0.02, 0.35: 0.04}[metallicity]

        db.add(
            {"imf": imf, "Z": metallicity},
            {"t": time_grid, "wl": ssp_wave, "info": mass_table, "spec": ssp_lumin},
        )
    db.close()


def build_bc2003(res):
    path = Path(__file__).parent / "bc03"
    db = SimpleDatabase("bc03", writable=True)

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
        "m72": 0.05,
    }

    for key, imf in itertools.product(metallicity, ["salp", "chab"]):
        ssp_filename = path / f"bc2003_{res}_{key}_{imf}_ssp.ised_ASCII"
        color3_filename = path / f"bc2003_lr_{key}_{imf}_ssp.3color"
        color4_filename = path / f"bc2003_lr_{key}_{imf}_ssp.4color"

        print(f"Importing {ssp_filename}...")

        # Read the desired information from the color files
        color_table = []
        color3_table = np.genfromtxt(color3_filename).transpose()
        color4_table = np.genfromtxt(color4_filename).transpose()
        color_table.append(color4_table[6])  # Mstar
        color_table.append(color4_table[7])  # Mgas
        color_table.append(10 ** color3_table[5])  # NLy

        color_table = np.array(color_table)

        ssp_time, ssp_wave, ssp_lumin = read_bc03_ssp(ssp_filename)

        # Regrid the SSP data to the evenly spaced time grid. In doing so we
        # assume 10 bursts every 0.1 Myr over a period of 1 Myr in order to
        # capture short evolutionary phases.
        # The time grid starts after 0.1 Myr, so we assume the value is the same
        # as the first actual time step.
        fill_value = (color_table[:, 0], color_table[:, -1])
        color_table = interpolate.interp1d(
            ssp_time,
            color_table,
            fill_value=fill_value,
            bounds_error=False,
            assume_sorted=True,
        )(fine_time_grid)
        color_table = np.mean(color_table.reshape(3, -1, 10), axis=-1)

        # We have to do the interpolation-averaging in several blocks as it is
        # a bit RAM intensive
        ssp_lumin_interp = np.empty((ssp_wave.size, time_grid.size))
        for i in range(0, ssp_wave.size, 100):
            fill_value = (ssp_lumin[i : i + 100, 0], ssp_lumin[i : i + 100, -1])
            ssp_interp = interpolate.interp1d(
                ssp_time,
                ssp_lumin[i : i + 100, :],
                fill_value=fill_value,
                bounds_error=False,
                assume_sorted=True,
            )(fine_time_grid)
            ssp_interp = ssp_interp.reshape(ssp_interp.shape[0], -1, 10)
            ssp_lumin_interp[i : i + 100, :] = np.mean(ssp_interp, axis=-1)

        # To avoid the creation of waves when interpolating, we refine the grid
        # beyond 10 μm following a log scale in wavelength. The interpolation
        # is also done in log space as the spectrum is power-law-like
        ssp_wave_resamp = np.around(np.logspace(np.log10(10000), np.log10(160000), 50))
        argmin = np.argmin(10000.0 - ssp_wave > 0) - 1
        ssp_lumin_resamp = 10.0 ** interpolate.interp1d(
            np.log10(ssp_wave[argmin:]),
            np.log10(ssp_lumin_interp[argmin:, :]),
            assume_sorted=True,
            axis=0,
        )(np.log10(ssp_wave_resamp))

        ssp_wave = np.hstack([ssp_wave[: argmin + 1], ssp_wave_resamp])
        ssp_lumin = np.vstack([ssp_lumin_interp[: argmin + 1, :], ssp_lumin_resamp])

        db.add(
            {"imf": imf, "Z": metallicity[key]},
            {"t": time_grid, "wl": ssp_wave, "info": color_table, "spec": ssp_lumin},
        )
    db.close()


def build_dale2014():
    path = Path(__file__).parent / "dale2014"
    db = SimpleDatabase("dale2014", writable=True)

    # Getting the alpha grid for the templates
    d14cal = np.genfromtxt(path / "dhcal.dat")
    alpha_grid = d14cal[:, 1]

    # Getting the lambda grid for the templates and convert from microns to nm.
    first_template = np.genfromtxt(path / "spectra.0.00AGN.dat")
    wave = first_template[:, 0] * 1e3

    # Getting the stellar emission and interpolate it at the same wavelength
    # grid
    stell_emission_file = np.genfromtxt(path / "stellar_SED_age13Gyr_tau10Gyr.spec")
    # A -> to nm
    wave_stell = stell_emission_file[:, 0] * 0.1
    # W/A -> W/nm
    stell_emission = stell_emission_file[:, 1] * 10
    stell_emission_interp = np.interp(wave, wave_stell, stell_emission)

    # The models are in nuFnu and contain stellar emission.
    # We convert this to W/nm and remove the stellar emission.

    # Emission from dust heated by SB
    fraction = 0.0
    filename = path / "spectra.0.00AGN.dat"
    print(f"Importing {filename}...")
    with filename.open() as datafile:
        data = "".join(datafile.readlines())

    for al in range(1, len(alpha_grid) + 1, 1):
        lumin_with_stell = np.genfromtxt(io.BytesIO(data.encode()), usecols=(al))
        lumin_with_stell = pow(10, lumin_with_stell) / wave
        constant = lumin_with_stell[7] / stell_emission_interp[7]
        lumin = lumin_with_stell - stell_emission_interp * constant
        lumin[lumin < 0] = 0
        lumin[wave < 2e3] = 0
        norm = np.trapz(lumin, x=wave)
        lumin /= norm

        db.add(
            {"fracAGN": float(fraction), "alpha": float(alpha_grid[al - 1])},
            {"wl": wave, "spec": lumin},
        )

    # Emission from dust heated by AGN - Quasar template
    filename = path / "shi_agn.regridded.extended.dat"
    print(f"Importing {filename}...")

    wave, lumin_quasar = np.genfromtxt(filename, unpack=True)
    wave *= 1e3
    lumin_quasar = 10**lumin_quasar / wave
    norm = np.trapz(lumin_quasar, x=wave)
    lumin_quasar /= norm

    db.add({"fracAGN": 1.0, "alpha": 0.0}, {"wl": wave, "spec": lumin_quasar})

    db.close()


def build_dl2007():
    path = Path(__file__).parent / "dl2007"
    db = SimpleDatabase("dl2007", writable=True)

    qpah = {
        "00": 0.47,
        "10": 1.12,
        "20": 1.77,
        "30": 2.50,
        "40": 3.19,
        "50": 3.90,
        "60": 4.58,
    }

    umaximum = ["1e3", "1e4", "1e5", "1e6"]
    uminimum = [
        "0.10",
        "0.15",
        "0.20",
        "0.30",
        "0.40",
        "0.50",
        "0.70",
        "0.80",
        "1.00",
        "1.20",
        "1.50",
        "2.00",
        "2.50",
        "3.00",
        "4.00",
        "5.00",
        "7.00",
        "8.00",
        "10.0",
        "12.0",
        "15.0",
        "20.0",
        "25.0",
    ]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {
        "00": 0.0100,
        "10": 0.0100,
        "20": 0.0101,
        "30": 0.0102,
        "40": 0.0102,
        "50": 0.0103,
        "60": 0.0104,
    }

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    filename = path / "U1e3" / "U1e3_1e3_MW3.1_00.txt"
    with filename.open() as datafile:
        data = "".join(datafile.readlines()[-1001:])

    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # For some reason wavelengths are decreasing in the model files
    wave = wave[::-1]
    # We convert wavelengths from μm to nm
    wave *= 1000.0

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4.0 * np.pi * 1e-30 / (cst.m_p + cst.m_e) * cst.c / (wave * wave) * 1e9

    for model in sorted(qpah.keys()):
        for umin in uminimum:
            filename = path / f"U{umin}" / f"U{umin}_{umin}_MW3.1_{model}.txt"
            print(f"Importing {filename}...")
            with filename.open() as datafile:
                data = "".join(datafile.readlines()[-1001:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            # For some reason fluxes are decreasing in the model files
            lumin = lumin[::-1]
            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv / MdMH[model]

            db.add(
                {"qpah": float(qpah[model]), "umin": float(umin), "umax": float(umin)},
                {"wl": wave, "spec": lumin},
            )
            for umax in umaximum:
                filename = path / f"U{umin}" / f"U{umin}_{umax}_MW3.1_{model}.txt"
                print(f"Importing {filename}...")
                with filename.open() as datafile:
                    data = "".join(datafile.readlines()[-1001:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                # For some reason fluxes are decreasing in the model files
                lumin = lumin[::-1]

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv / MdMH[model]

                db.add(
                    {
                        "qpah": float(qpah[model]),
                        "umin": float(umin),
                        "umax": float(umax),
                    },
                    {"wl": wave, "spec": lumin},
                )
    db.close()


def build_dl2014():
    path = Path(__file__).parent / "dl2014"
    db = SimpleDatabase("dl2014", writable=True)

    qpah = {
        "000": 0.47,
        "010": 1.12,
        "020": 1.77,
        "030": 2.50,
        "040": 3.19,
        "050": 3.90,
        "060": 4.58,
        "070": 5.26,
        "080": 5.95,
        "090": 6.63,
        "100": 7.32,
    }

    uminimum = [
        "0.100",
        "0.120",
        "0.150",
        "0.170",
        "0.200",
        "0.250",
        "0.300",
        "0.350",
        "0.400",
        "0.500",
        "0.600",
        "0.700",
        "0.800",
        "1.000",
        "1.200",
        "1.500",
        "1.700",
        "2.000",
        "2.500",
        "3.000",
        "3.500",
        "4.000",
        "5.000",
        "6.000",
        "7.000",
        "8.000",
        "10.00",
        "12.00",
        "15.00",
        "17.00",
        "20.00",
        "25.00",
        "30.00",
        "35.00",
        "40.00",
        "50.00",
    ]

    alpha = [
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
        "1.8",
        "1.9",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "2.8",
        "2.9",
        "3.0",
    ]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {
        "000": 0.0100,
        "010": 0.0100,
        "020": 0.0101,
        "030": 0.0102,
        "040": 0.0102,
        "050": 0.0103,
        "060": 0.0104,
        "070": 0.0105,
        "080": 0.0106,
        "090": 0.0107,
        "100": 0.0108,
    }

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    filename = path / "U0.100_0.100_MW3.1_000" / "spec_1.0.dat"
    with filename.open() as datafile:
        data = "".join(datafile.readlines()[-1001:])
    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # For some reason wavelengths are decreasing in the model files
    wave = wave[::-1]
    # We convert wavelengths from μm to nm
    wave *= 1000.0

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4.0 * np.pi * 1e-30 / (cst.m_p + cst.m_e) * cst.c / (wave * wave) * 1e9

    for model in sorted(qpah.keys()):
        for umin in uminimum:
            filename = path / f"U{umin}_{umin}_MW3.1_{model}" / "spec_1.0.dat"
            print(f"Importing {filename}...")
            with filename.open() as datafile:
                data = "".join(datafile.readlines()[-1001:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            # For some reason fluxes are decreasing in the model files
            lumin = lumin[::-1]

            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv / MdMH[model]

            db.add(
                {
                    "qpah": float(qpah[model]),
                    "umin": float(umin),
                    "umax": float(umin),
                    "alpha": 1.0,
                },
                {"wl": wave, "spec": lumin},
            )

            for al in alpha:
                filename = path / f"U{umin}_1e7_MW3.1_{model}" / f"spec_{al}.dat"
                print(f"Importing {filename}...")
                with filename.open() as datafile:
                    data = "".join(datafile.readlines()[-1001:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                # For some reason fluxes are decreasing in the model files
                lumin = lumin[::-1]

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv / MdMH[model]

                db.add(
                    {
                        "qpah": float(qpah[model]),
                        "umin": float(umin),
                        "umax": 1e7,
                        "alpha": float(al),
                    },
                    {"wl": wave, "spec": lumin},
                )
    db.close()


def build_fritz2006():
    path = Path(__file__).parent / "fritz2006"
    db = SimpleDatabase("fritz2006", writable=True)

    # Parameters of Fritz+2006
    psy = [
        "0.001",
        "10.100",
        "20.100",
        "30.100",
        "40.100",
        "50.100",
        "60.100",
        "70.100",
        "80.100",
        "89.990",
    ]  # Viewing angle in degrees
    opening_angle = [20, 40, 60]  # Theta = 2*(90 - opening_angle)
    gamma = ["0.0", "2.0", "4.0", "6.0"]
    beta = ["-1.00", "-0.75", "-0.50", "-0.25", "0.00"]
    tau = ["0.1", "0.3", "0.6", "1.0", "2.0", "3.0", "6.0", "10.0"]
    r_ratio = [10, 30, 60, 100, 150]

    # Read and convert the wavelength
    filename = path / "ct20al0.0be-1.00ta0.1rm10.tot"
    with filename.open() as datafile:
        data = "".join(datafile.readlines()[-178:])
    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    wave *= 1e3
    # Number of wavelengths: 178; Number of comments lines: 28
    nskip = 28
    blocksize = 178

    iter_params = (
        (oa, gam, be, ta, rm)
        for oa in opening_angle
        for gam in gamma
        for be in beta
        for ta in tau
        for rm in r_ratio
    )

    for params in iter_params:
        filename = path / "ct{}al{}be{}ta{}rm{}.tot".format(*params)
        print(f"Importing {filename}...")
        try:
            with filename.open() as datafile:
                data = datafile.readlines()
        except IOError:
            continue

        for n in range(len(psy)):
            block = data[
                nskip
                + blocksize * n
                + 4 * (n + 1)
                - 1 : nskip
                + blocksize * (n + 1)
                + 4 * (n + 1)
                - 1
            ]
            dust, scatt, disk = np.genfromtxt(
                io.BytesIO("".join(block).encode()), usecols=(2, 3, 4), unpack=True
            )
            # Remove NaN
            dust = np.nan_to_num(dust)
            scatt = np.nan_to_num(scatt)
            disk = np.nan_to_num(disk)
            # Merge scatter into disk
            disk += scatt
            # Conversion from erg/s/microns to W/nm
            dust *= 1e-4
            disk *= 1e-4
            # Normalization of the lumin_therm to 1W
            norm = np.trapz(dust, x=wave)
            dust /= norm
            disk /= norm

            db.add(
                {
                    "r_ratio": float(params[4]),
                    "tau": float(params[3]),
                    "beta": float(params[2]),
                    "gamma": float(params[1]),
                    "opening_angle": float(params[0]),
                    "psy": float(psy[n]),
                },
                {"norm": norm, "wl": wave, "disk": disk, "dust": dust},
            )
    db.close()


def build_skirtor2016():
    path = Path(__file__).parent / "skirtor2016"
    db = SimpleDatabase("skirtor2016", writable=True)

    params = [f.stem.split("_")[:-1] for f in path.glob("*")]

    # Parameters of SKIRTOR 2016
    t = list({param[0][1:] for param in params})
    p = list({param[1][1:] for param in params})
    q = list({param[2][1:] for param in params})
    oa = list({param[3][2:] for param in params})
    R = list({param[4][1:] for param in params})
    Mcl = list({param[5][3:] for param in params})
    i = list({param[6][1:] for param in params})

    iter_params = (
        (p1, p2, p3, p4, p5, p6, p7)
        for p1 in t
        for p2 in p
        for p3 in q
        for p4 in oa
        for p5 in R
        for p6 in Mcl
        for p7 in i
    )

    for params in iter_params:
        filename = path / "t{}_p{}_q{}_oa{}_R{}_Mcl{}_i{}_sed.dat".format(*params)
        print(f"Importing {filename}...")

        wl, disk, scatt, dust = np.genfromtxt(
            filename, unpack=True, usecols=(0, 2, 3, 4)
        )
        wl *= 1e3
        disk += scatt
        disk /= wl
        dust /= wl

        # Extrapolate the model to 10 mm
        wl_ext = np.array([2e6, 4e6, 8e6, 1e7])
        disk_ext = np.zeros(len(wl_ext)) + 1e-99
        if dust[-1] == 0:
            dust_ext = np.zeros(len(wl_ext)) + 1e-99
        else:
            dust_ext = 10 ** (
                np.log10(dust[-1])
                + np.log10(wl_ext / wl[-1])
                * np.log10(dust[-2] / dust[-1])
                / np.log10(wl[-2] / wl[-1])
            )
        wl = np.append(wl, wl_ext)
        disk[-1] = 1e-99
        disk = np.append(disk, disk_ext)
        dust = np.append(dust, dust_ext)

        # Interpolate to a denser grid
        with SimpleDatabase("nebular_continuum") as db1:
            nebular = db1.get(Z=0.019, logU=-2.0, ne=100.0)
        wl_den = nebular.wl[np.where((nebular.wl >= 3e4) & (nebular.wl <= 1e7))]
        idx = np.where(wl > 1e4)
        disk_den = 10 ** np.interp(
            np.log10(wl_den), np.log10(wl[idx]), np.log10(disk[idx])
        )
        dust_den = 10 ** np.interp(
            np.log10(wl_den), np.log10(wl[idx]), np.log10(dust[idx])
        )
        idx = np.where(wl < 3e4)
        wl = np.append(wl[idx], wl_den)
        disk = np.append(disk[idx], disk_den)
        dust = np.append(dust[idx], dust_den)

        # Normalization of the lumin_therm to 1W
        norm = np.trapz(dust, x=wl)
        disk /= norm
        dust /= norm

        db.add(
            {
                "t": int(params[0]),
                "pl": float(params[1]),
                "q": float(params[2]),
                "oa": int(params[3]),
                "R": int(params[4]),
                "Mcl": float(params[5]),
                "i": int(params[6]),
            },
            {"norm": norm, "wl": wl, "disk": disk, "dust": dust},
        )
    db.close()


def build_nebular():
    path = Path(__file__).parent / "nebular"

    filename = path / "lines.dat"
    print(f"Importing {filename}...")
    lines = np.genfromtxt(filename)

    tmp = Table.read(path / "line_wavelengths.dat", format="ascii")
    wave_lines = tmp["col1"].data
    name_lines = tmp["col2"].data

    # Build the parameters
    metallicities = np.unique(lines[:, 1])
    logUs = np.around(np.arange(-4.0, -0.9, 0.1), 1)
    nes = np.array([10.0, 100.0, 1000.0])

    filename = path / "continuum.dat"
    print(f"Importing {filename}...")
    cont = np.genfromtxt(filename)

    # Convert wavelength from Å to nm
    wave_lines *= 0.1
    wave_cont = cont[:1600, 0] * 0.1

    # Compute the wavelength grid to resample the models so as to eliminate
    # non-physical waves and compute the models faster by avoiding resampling
    # them at run time.
    with SimpleDatabase("bc03") as db:
        wave_stellar = db.get(imf="salp", Z=0.02).wl
    with SimpleDatabase("dl2014") as db:
        wave_dust = db.get(qpah=0.47, umin=1.0, umax=1.0, alpha=1.0).wl
    wave_cont_interp = np.unique(
        np.hstack([wave_cont, wave_stellar, wave_dust, np.logspace(7.0, 9.0, 501)])
    )

    # Keep only the fluxes
    lines = lines[:, 2:]
    cont = cont[:, 1:]

    # Reshape the arrays so they are easier to handle
    cont = np.reshape(cont, (metallicities.size, wave_cont.size, logUs.size, nes.size))
    lines = np.reshape(
        lines, (wave_lines.size, metallicities.size, logUs.size, nes.size)
    )

    # Move the wavelength to the last position to ease later computations
    # 0: metallicity, 1: log U, 2: ne, 3: wavelength
    cont = np.moveaxis(cont, 1, -1)
    lines = np.moveaxis(lines, (0, 1, 2, 3), (3, 0, 1, 2))

    # Convert lines to a linear scale
    lines = 10.0**lines

    # Convert continuum to W/nm
    cont *= 1e-7 * cst.c * 1e9 / wave_cont**2

    # Import lines
    db = SimpleDatabase("nebular_lines", writable=True)
    for idxZ, metallicity in enumerate(metallicities):
        for idxU, logU in enumerate(logUs):
            for ne, spectrum in zip(nes, lines[idxZ, idxU, :, :]):
                db.add(
                    {"Z": float(metallicity), "logU": float(logU), "ne": float(ne)},
                    {"name": name_lines, "wl": wave_lines, "spec": spectrum},
                )
    db.close()

    # Import continuum
    db = SimpleDatabase("nebular_continuum", writable=True)
    spectra = 10 ** interpolate.interp1d(np.log10(wave_cont), np.log10(cont), axis=-1)(
        np.log10(wave_cont_interp)
    )
    spectra = np.nan_to_num(spectra)
    for idxZ, metallicity in enumerate(metallicities):
        for idxU, logU in enumerate(logUs):
            for ne, spectrum in zip(nes, spectra[idxZ, idxU, :, :]):
                db.add(
                    {"Z": float(metallicity), "logU": float(logU), "ne": float(ne)},
                    {"wl": wave_cont_interp, "spec": spectrum},
                )
    db.close()


def build_schreiber2016():
    path = Path(__file__).parent / "schreiber2016"
    db = SimpleDatabase("schreiber2016", writable=True)

    filename = path / "g15_pah.fits"
    print(f"Importing {filename}...")
    pah = Table.read(filename)

    filename = path / "g15_dust.fits"
    print(f"Importing {filename}...")
    dust = Table.read(filename)

    # Getting the lambda grid for the templates and convert from μm to nm.
    wave = dust["LAM"][0, 0, :].data * 1e3

    for td in np.arange(15.0, 100.0):
        # Find the closest temperature in the model list of tdust
        tsed = np.argmin(np.absolute(dust["TDUST"][0].data - td))

        # The models are in νFν.  We convert this to W/nm.
        lumin_dust = dust["SED"][0, tsed, :].data / wave
        lumin_pah = pah["SED"][0, tsed, :].data / wave

        db.add({"type": 0, "tdust": float(td)}, {"wl": wave, "spec": lumin_dust})
        db.add({"type": 1, "tdust": float(td)}, {"wl": wave, "spec": lumin_pah})

    db.close()


def build_themis():
    path = Path(__file__).parent / "themis"
    db = SimpleDatabase("themis", writable=True)

    # Mass fraction of hydrocarbon solids i.e., a-C(:H) smaller than 1.5 nm,
    # also known as HAC
    qhac = {
        "000": 0.02,
        "010": 0.06,
        "020": 0.10,
        "030": 0.14,
        "040": 0.17,
        "050": 0.20,
        "060": 0.24,
        "070": 0.28,
        "080": 0.32,
        "090": 0.36,
        "100": 0.40,
    }

    uminimum = [
        "0.100",
        "0.120",
        "0.150",
        "0.170",
        "0.200",
        "0.250",
        "0.300",
        "0.350",
        "0.400",
        "0.500",
        "0.600",
        "0.700",
        "0.800",
        "1.000",
        "1.200",
        "1.500",
        "1.700",
        "2.000",
        "2.500",
        "3.000",
        "3.500",
        "4.000",
        "5.000",
        "6.000",
        "7.000",
        "8.000",
        "10.00",
        "12.00",
        "15.00",
        "17.00",
        "20.00",
        "25.00",
        "30.00",
        "35.00",
        "40.00",
        "50.00",
        "80.00",
    ]

    alpha = [
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
        "1.8",
        "1.9",
        "2.0",
        "2.1",
        "2.2",
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
        "2.8",
        "2.9",
        "3.0",
    ]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {
        "000": 7.4e-3,
        "010": 7.4e-3,
        "020": 7.4e-3,
        "030": 7.4e-3,
        "040": 7.4e-3,
        "050": 7.4e-3,
        "060": 7.4e-3,
        "070": 7.4e-3,
        "080": 7.4e-3,
        "090": 7.4e-3,
        "100": 7.4e-3,
    }

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    filename = path / "U0.100_0.100_MW3.1_000" / "spec_1.0.dat"
    with filename.open() as datafile:
        data = "".join(datafile.readlines()[-576:])

    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))

    # We convert wavelengths from μm to nm
    wave *= 1000.0

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4.0 * np.pi * 1e-30 / (cst.m_p + cst.m_e) * cst.c / (wave * wave) * 1e9

    for model in sorted(qhac.keys()):
        for umin in uminimum:
            filename = path / f"U{umin}_{umin}_MW3.1_{model}" / "spec_1.0.dat"
            print(f"Importing {filename}...")
            with open(filename) as datafile:
                data = "".join(datafile.readlines()[-576:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))

            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv / MdMH[model]

            db.add(
                {
                    "qhac": float(qhac[model]),
                    "umin": float(umin),
                    "umax": float(umin),
                    "alpha": 1.0,
                },
                {"wl": wave, "spec": lumin},
            )
            for al in alpha:
                filename = path / f"U{umin}_1e7_MW3.1_{model}" / f"spec_{al}.dat"
                print(f"Importing {filename}...")
                with open(filename) as datafile:
                    data = "".join(datafile.readlines()[-576:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv / MdMH[model]

                db.add(
                    {
                        "qhac": float(qhac[model]),
                        "umin": float(umin),
                        "umax": 1e7,
                        "alpha": float(al),
                    },
                    {"wl": wave, "spec": lumin},
                )
    db.close()


def build_base(bc03res="lr"):
    print("#" * 78)
    print("1- Importing filters...\n")
    build_filters()
    print("\nDONE\n")
    print("#" * 78)

    print("2- Importing Maraston 2005 SSP\n")
    build_m2005()
    print("\nDONE\n")
    print("#" * 78)

    print("3- Importing Bruzual and Charlot 2003 SSP\n")
    build_bc2003(bc03res)
    print("\nDONE\n")
    print("#" * 78)

    print("4- Importing Draine and Li (2007) models\n")
    build_dl2007()
    print("\nDONE\n")
    print("#" * 78)

    print("5- Importing the updated Draine and Li (2007) models\n")
    build_dl2014()
    print("\nDONE\n")
    print("#" * 78)

    print("6- Importing Jones et al (2017) models)\n")
    build_themis()
    print("\nDONE\n")
    print("#" * 78)

    print("7- Importing Dale et al (2014) templates\n")
    build_dale2014()
    print("\nDONE\n")
    print("#" * 78)

    print("8- Importing Schreiber et al (2016) models\n")
    build_schreiber2016()
    print("\nDONE\n")
    print("#" * 78)

    print("9- Importing nebular lines and continuum\n")
    build_nebular()
    print("\nDONE\n")
    print("#" * 78)

    print("10- Importing Fritz et al. (2006) models\n")
    build_fritz2006()
    print("\nDONE\n")
    print("#" * 78)

    print("11- Importing SKIRTOR 2016 models\n")
    build_skirtor2016()
    print("\nDONE\n")
    print("#" * 78)


if __name__ == "__main__":
    build_base()
