from itertools import repeat

from astropy.table import Table
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pkg_resources
from scipy.constants import c
from pcigale.data import SimpleDatabase as Database
from utils.io import read_table
import matplotlib.gridspec as gridspec
from utils.counter import Counter

# Name of the file containing the best models information
BEST_RESULTS = "results.fits"
MOCK_RESULTS = "results_mock.fits"

AVAILABLE_SERIES = [
    'stellar_attenuated',
    'stellar_unattenuated',
    'nebular',
    'dust',
    'agn',
    'radio',
    'model'
]


def pool_initializer(counter):
    """Initializer of the pool of processes to share variables between workers.
    Parameters
    ----------
    :param counter: Counter class object for the number of models plotted
    """
    global gbl_counter

    gbl_counter = counter


def sed(config, sed_type, nologo, xrange, yrange, series, format, outdir):
    """Plot the best SED with associated observed and modelled fluxes.
    """
    obs = read_table(outdir.parent / config.configuration['data_file'])
    mod = Table.read(outdir / BEST_RESULTS)

    with Database("filters") as db:
        filters = {name: db.get(name=name)
                   for name in config.configuration['bands']
                   if not (name.endswith('_err') or name.startswith('line'))}

    if nologo is True:
        logo = False
    else:
        logo = plt.imread(pkg_resources.resource_filename(__name__,
                                                          "../resources/CIGALE.png"))

    counter = Counter(len(obs))
    with mp.Pool(processes=config.configuration['cores'],
                 initializer=pool_initializer, initargs=(counter,)) as pool:
        pool.starmap(_sed_worker, zip(obs, mod, repeat(filters),
                                      repeat(sed_type), repeat(logo),
                                      repeat(xrange), repeat(yrange),
                                      repeat(series), repeat(format),
                                      repeat(outdir)))
        pool.close()
        pool.join()


def _sed_worker(obs, mod, filters, sed_type, logo, xrange, yrange, series,
                format, outdir):
    """Plot the best SED with the associated fluxes in bands

    Parameters
    ----------
    obs: Table row
        Data from the input file regarding one object.
    mod: Table row
        Data from the best model of one object.
    filters: dict
        The observed fluxes in each filter.
    sed_type: string
        Type of SED to plot. It can either be "mJy" (flux in mJy and observed
        frame) or "lum" (luminosity in W and rest frame)
    logo: numpy.array | boolean
        The readed logo image data. Has shape
        (M, N) for grayscale images.
        (M, N, 3) for RGB images.
        (M, N, 4) for RGBA images.
        Do not add the logo when set to False.
    xrange: tuple(float|boolean, float|boolean)
    yrange: tuple(float|boolean, float|boolean)
    series: list
    format: string
        One of png, pdf, ps, eps or svg.
    outdir: Path
        Path to outdir

    """
    np.seterr(invalid='ignore')
    gbl_counter.inc()

    id_best_model_file = outdir / f"{obs['id']}_best_model.fits"
    if id_best_model_file.is_file():
        sed = Table.read(id_best_model_file)

        filters_wl = np.array([filt.pivot
                               for filt in filters.values()]) * 1e-3
        wavelength_spec = sed['wavelength'] * 1e-3
        obs_fluxes = np.array([obs[filt] for filt in filters.keys()])
        obs_fluxes_err = np.array([obs[filt + '_err']
                                   for filt in filters.keys()])
        mod_fluxes = np.array([mod["best." + filt]
                               if "best." + filt in mod.colnames else np.nan
                               for filt in filters.keys()])
        if obs['redshift'] >= 0:
            z = float(obs['redshift'])
        else:  # Redshift mode
            z = float(mod['best.universe.redshift'])
        zp1 = 1. + z
        surf = 4. * np.pi * mod['best.universe.luminosity_distance'] ** 2

        xmin = 0.9 * np.min(filters_wl) if xrange[0] is False else xrange[0]
        xmax = 1.1 * np.max(filters_wl) if xrange[1] is False else xrange[1]

        if sed_type == 'lum':
            k_corr_SED = 1e-29 * surf * c / (filters_wl * 1e-6)
            obs_fluxes *= k_corr_SED
            obs_fluxes_err *= k_corr_SED
            mod_fluxes *= k_corr_SED

            for cname in sed.colnames[1:]:
                sed[cname] *= wavelength_spec * 1e3

            filters_wl /= zp1
            wavelength_spec /= zp1
            xmin /= zp1
            xmax /= zp1
        elif sed_type == 'mJy':
            k_corr_SED = 1.

            fact = 1e29 * 1e-3 * wavelength_spec**2 / c / surf
            for cname in sed.colnames[1:]:
                sed[cname] *= fact
        else:
            print("Unknown plot type")

        wsed = np.where((wavelength_spec > xmin) & (wavelength_spec < xmax))
        figure = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        if (sed.columns[1][wsed] > 0.).any():
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])

            # Stellar emission
            if ('stellar_attenuated' in series
                    and 'stellar.young' in sed.columns):
                spectrum = (sed['stellar.young'][wsed] +
                            sed['stellar.old'][wsed])

                if 'nebular.absoroption_young' in sed.columns:
                    spectrum += sed['nebular.absortion_young'][wsed]
                    spectrum += sed['nebular.absorption_old'][wsed]

                if 'attenuation.stellar.young' in sed.columns:
                    spectrum += sed['attenuation.stellar.young'][wsed]
                    spectrum += sed['attenuation.stellar.old'][wsed]

                ax1.loglog(wavelength_spec[wsed], spectrum,
                           label="Stellar attenuated", color='gold',
                           marker=None, nonpositive='clip', linestyle='-',
                           linewidth=1.0)

            if ('stellar_unattenuated' in series
                    and 'stellar.young' in sed.columns):
                ax1.loglog(wavelength_spec[wsed],
                           (sed['stellar.old'][wsed] +
                            sed['stellar.young'][wsed]),
                           label="Stellar unattenuated",
                           color='xkcd:deep sky blue', marker=None,
                           nonpositive='clip', linestyle='--', linewidth=1.0)

            # Nebular emission
            if 'nebular' in series and 'nebular.lines_young' in sed.columns:
                spectrum = (sed['nebular.lines_young'][wsed] +
                            sed['nebular.lines_old'][wsed] +
                            sed['nebular.continuum_young'][wsed] +
                            sed['nebular.continuum_old'][wsed])

                if 'attenuation.nebular.lines_young' in sed.columns:
                    spectrum += sed['attenuation.nebular.lines_young'][wsed]
                    spectrum += sed['attenuation.nebular.lines_old'][wsed]
                    spectrum += sed['attenuation.nebular.continuum_young'][wsed]
                    spectrum += sed['attenuation.nebular.continuum_old'][wsed]

                ax1.loglog(wavelength_spec[wsed], spectrum,
                           label="Nebular emission", color='xkcd:true green',
                           marker=None, nonpositive='clip', linewidth=1.0)

            # Dust emission Draine & Li
            if 'dust' in series and 'dust.Umin_Umin' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           (sed['dust.Umin_Umin'][wsed] +
                            sed['dust.Umin_Umax'][wsed]),
                           label="Dust emission", color='xkcd:bright red',
                           marker=None, nonpositive='clip', linestyle='-',
                           linewidth=1.0)

            # Dust emission Dale
            if 'dust' in series and 'dust' in sed.columns:
                ax1.loglog(wavelength_spec[wsed], sed['dust'][wsed],
                           label="Dust emission", color='xkcd:bright red',
                           marker=None, nonpositive='clip', linestyle='-',
                           linewidth=1.0)

            # AGN emission Fritz
            if 'agn' in series and 'agn.fritz2006_therm' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           (sed['agn.fritz2006_therm'][wsed] +
                            sed['agn.fritz2006_scatt'][wsed] +
                            sed['agn.fritz2006_agn'][wsed]),
                           label="AGN emission", color='xkcd:apricot',
                           marker=None, nonpositive='clip', linestyle='-',
                           linewidth=1.0)

            # AGN emission SKIRTOR
            if 'agn' in series and 'agn.SKIRTOR2016_dust' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           (sed['agn.SKIRTOR2016_dust'][wsed] +
                            sed['agn.SKIRTOR2016_disk'][wsed]),
                           label="AGN emission", color='xkcd:apricot',
                           marker=None, nonpositive='clip', linestyle='-',
                           linewidth=1.0)

            # Radio emission
            if 'radio' in series and 'radio.sf_nonthermal' in sed.columns:
                ax1.loglog(wavelength_spec[wsed],
                           sed['radio.sf_nonthermal'][wsed],
                           label="Radio SF nonthermal", color='brown',
                           marker=None, nonpositive='clip', linestyle='-',
                           linewidth=1.0)

            if 'model' in series:
                ax1.loglog(wavelength_spec[wsed], sed['L_lambda_total'][wsed],
                           label="Model spectrum", color='k',
                           nonpositive='clip', linestyle='-', linewidth=1.5)

            ax1.set_autoscale_on(False)
            s = np.argsort(filters_wl)
            filters_wl = filters_wl[s]
            mod_fluxes = mod_fluxes[s]
            obs_fluxes = obs_fluxes[s]
            obs_fluxes_err = obs_fluxes_err[s]
            ax1.scatter(filters_wl, mod_fluxes, marker='o',
                        color='xkcd:strawberry', s=8, zorder=3,
                        label="Model fluxes")
            mask_ok = np.logical_and(obs_fluxes > 0., obs_fluxes_err > 0.)
            ax1.errorbar(filters_wl[mask_ok], obs_fluxes[mask_ok],
                         yerr=obs_fluxes_err[mask_ok], ls='', marker='o',
                         label='Observed fluxes', markerfacecolor='None',
                         markersize=5, markeredgecolor='xkcd:pastel purple',
                         color='xkcd:light indigo', capsize=0., zorder=3, lw=1)
            mask_uplim = np.logical_and(np.logical_and(obs_fluxes > 0.,
                                                       obs_fluxes_err < 0.),
                                        obs_fluxes_err > -9990. * k_corr_SED)
            if not mask_uplim.any() == False:
                ax1.errorbar(filters_wl[mask_uplim], obs_fluxes[mask_uplim],
                             yerr=obs_fluxes_err[mask_uplim], ls='', marker='v',
                             label='Observed upper limits',
                             markerfacecolor='None', markersize=6,
                             markeredgecolor='g', capsize=0.)
            mask_noerr = np.logical_and(obs_fluxes > 0.,
                                        obs_fluxes_err < -9990. * k_corr_SED)
            if not mask_noerr.any() == False:
                ax1.errorbar(filters_wl[mask_noerr], obs_fluxes[mask_noerr],
                             ls='', marker='p', markerfacecolor='None',
                             markersize=5, markeredgecolor='r',
                             label='Observed fluxes, no errors', capsize=0.)
            mask = np.where(obs_fluxes > 0.)
            ax2.errorbar(filters_wl[mask],
                         (obs_fluxes[mask] - mod_fluxes[mask]) / obs_fluxes[mask],
                         yerr=obs_fluxes_err[mask] / obs_fluxes[mask],
                         marker='_', label="(Obs-Mod)/Obs", color='k',
                         capsize=0., ls='None', lw=1)
            ax2.plot([xmin, xmax], [0., 0.], ls='--', color='k')
            ax2.set_xscale('log')
            ax2.minorticks_on()

            ax1.tick_params(direction='in', axis='both', which='both', top=True,
                            left=True, right=True, bottom=False)
            ax2.tick_params(direction='in', axis='both', which='both',
                            right=True)

            figure.subplots_adjust(hspace=0., wspace=0.)

            ax1.set_xlim(xmin, xmax)

            if yrange[0] is not False:
                ymin = yrange[0]
            else:
                if not mask_uplim.any() == False:
                    ymin = min(min(np.min(obs_fluxes[mask_ok]),
                                   np.min(obs_fluxes[mask_uplim])),
                               min(np.min(mod_fluxes[mask_ok]),
                                   np.min(mod_fluxes[mask_uplim])))
                elif not mask_ok.any() == False:
                    ymin = min(np.min(obs_fluxes[mask_ok]),
                               np.min(mod_fluxes[mask_ok]))
                else:  # No valid flux (e.g., fitting only properties)
                    ymin = ax1.get_ylim()[0]
                ymin *= 1e-1

            if yrange[1] is not False:
                ymax = yrange[1]
            else:
                if not mask_uplim.any() == False:
                    ymax = max(max(np.max(obs_fluxes[mask_ok]),
                                   np.max(obs_fluxes[mask_uplim])),
                               max(np.max(mod_fluxes[mask_ok]),
                                   np.max(mod_fluxes[mask_uplim])))
                elif not mask_ok.any() == False:
                    ymax = max(np.max(obs_fluxes[mask_ok]),
                               np.max(mod_fluxes[mask_ok]))
                else:  # No valid flux (e.g., fitting only properties)
                    ymax = ax1.get_ylim()[1]
                ymax *= 1e1

            xmin = xmin if xmin < xmax else xmax - 1e1
            ymin = ymin if ymin < ymax else ymax - 1e1

            ax1.set_ylim(ymin, ymax)
            ax2.set_xlim(xmin, xmax)
            ax2.set_ylim(-1.0, 1.0)
            if sed_type == 'lum':
                ax2.set_xlabel(r"Rest-frame wavelength [$\mu$m]")
                ax1.set_ylabel("Luminosity [W]")
            else:
                ax2.set_xlabel(r"Observed $\lambda$ ($\mu$m)")
                ax1.set_ylabel(r"S$_\nu$ (mJy)")
            ax2.set_ylabel("Relative\nresidual")
            ax1.legend(fontsize=6, loc='best', frameon=False)
            ax2.legend(fontsize=6, loc='best', frameon=False)
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax1.get_yticklabels()[1], visible=False)
            figure.suptitle(f"Best model for {obs['id']}\n (z={z:.3}, "
                            f"reduced χ²={mod['best.reduced_chi_square']:.2})")
            if logo is not False:
                # Multiplying the dpi by 2 is a hack so the figure is small
                # and not too pixelated
                figwidth = figure.get_figwidth() * figure.dpi * 2.
                figure.figimage(logo, figwidth - logo.shape[0], 0,
                                origin='upper', zorder=0, alpha=1)

            figure.savefig(outdir / f"{obs['id']}_best_model.{format}",
                           dpi=figure.dpi * 2.)
            plt.close(figure)
        else:
            print(f"No valid best SED found for {obs['id']}. No plot created.")
    else:
        print(f"No SED found for {obs['id']}. No plot created.")
