from astropy.table import Table, Column
import numpy as np


def save_sed_to_fits(sed, prefix, norm=1.0):
    """
    Save a SED object to fits files

    Parameters
    ----------
    sed: a pcigale.sed.SED object
        The SED to save
    prefix: string
        Prefix of the fits file containing the path and the id of the model
    norm: float
        Normalisation factor of the SED

    """
    info = {}
    for name in sed.info:
        if name in sed.mass_proportional_info:
            info[name] = str(norm * sed.info[name])
        else:
            info[name] = str(sed.info[name])

    table = Table(meta=info)
    table['wavelength'] = Column(sed.wavelength_grid, unit="nm")
    table['Fnu'] = Column(norm * sed.fnu, unit="mJy")
    table['L_lambda_total'] = Column(norm * sed.luminosity, unit="W/nm")
    for name in sed.luminosities:
        table[name] = Column(norm * sed.luminosities[name], unit="W/nm")
    table.write(f"{prefix}_best_model.fits")

    if sed.sfh is not None:
        table = Table(meta=info)
        table["time"] = Column(np.arange(sed.sfh.size), unit="Myr")
        table["SFR"] = Column(norm * sed.sfh, unit="Msun/yr")
        table.write(f"{prefix}_SFH.fits")
