import argparse
import multiprocessing as mp
import sys

from astropy.table import Table, Column
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cst

from pcigale.data import SimpleDatabase as Database


def list_filters():
    """Print the list of filters in the pcigale database.
    """
    with Database("filters") as base:
        filters = {name: base.get(name=name) for name in
                   base.parameters["name"]}

    name = Column(data=[filters[f].name for f in filters], name='Name')
    description = Column(data=[filters[f].desc for f in filters],
                         name='Description')
    wl = Column(data=[filters[f].pivot for f in filters],
                name='Pivot Wavelength', unit=u.nm, format='%d')
    samples = Column(data=[filters[f].wl.size for f in filters],
                     name="Points")

    t = Table()
    t.add_columns([name, description, wl, samples])
    t.sort(['Pivot Wavelength'])
    t.pprint(max_lines=-1, max_width=-1)


def add_filters(fnames):
    """Add filters to the pcigale database.
    """
    db = Database("filters", writable=True)

    for fname in fnames:
        with open(fname, 'r') as f:
            name = f.readline().strip('# \n\t')
            type_ = f.readline().strip('# \n\t')
            desc = f.readline().strip('# \n\t')
        wl, tr = np.genfromtxt(fname, unpack=True)

        # We convert the wavelength from Å to nm.
        wl *= 0.1

        # We convert to energy if needed
        if type_ == 'photon':
            tr *= wl
        elif type_ != 'energy':
            raise ValueError("Filter transmission type can only be "
                             "'energy' or 'photon'.")

        print(f"Importing {name}... ({wl.size} points)")

        # We normalise the filter and compute the pivot wavelength. If the
        # filter is a pseudo-filter used to compute line fluxes, it should not
        # be normalised.
        if not name.startswith('PSEUDO'):
            pivot = np.sqrt(np.trapz(tr, wl) / np.trapz(tr / wl**2, wl))

            # The factor 10²⁰ is so that we get the fluxes directly in mJy when
            # we integrate with the wavelength in units of nm and the spectrum
            # in units of W/m²/nm.
            tr *= 1e20 / (cst.c * np.trapz(tr / wl**2, wl))
        else:
            pivot = np.mean(wl[tr > 0.])

        db.add({"name": name},
               {"wl": wl, "tr": tr, "pivot": pivot, "desc": desc})
    db.close()


def worker_plot(fname):
    """Worker to plot filter transmission curves in parallel

    Parameters
    ----------
    fname: string
        Name of the filter to be plotted
    """
    with Database("filters") as db:
        _filter = db.get(name=fname)

    if _filter.pivot >= 1e3 and _filter.pivot < 1e6:
        _filter.wl *= 1e-3
        unit = "μm"
    elif _filter.pivot >= 1e6 and _filter.pivot < 1e7:
        _filter.wl *= 1e-6
        unit = "mm"
    elif _filter.pivot >= 1e7:
        _filter.wl *= 1e-7
        unit = "cm"
    else:
        unit = "nm"

    _filter.tr *= 1. / np.max(_filter.tr)

    plt.clf()
    plt.plot(_filter.wl, _filter.tr, color='k')
    plt.xlim(_filter.wl[0], _filter.wl[-1])
    plt.minorticks_on()
    plt.xlabel(f'Wavelength [{unit}]')
    plt.ylabel('Relative transmission')
    plt.title(f"{fname} filter")
    plt.tight_layout()
    plt.savefig(f"{fname}.pdf")


def plot_filters(fnames):
    """Plot the filters provided as parameters. If not filter is given, then
    plot all the filters.
    """
    if len(fnames) == 0:
        with Database("filters") as db:
            fnames = db.parameters["name"]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(worker_plot, fnames)


def main():

    if sys.version_info[:2] >= (3, 4):
        mp.set_start_method('spawn')
    else:
        print("Could not set the multiprocessing start method to spawn. If "
              "you encounter a deadlock, please upgrade to Python≥3.4.")

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(help="List of commands")

    list_parser = subparsers.add_parser('list', help=list_filters.__doc__)
    list_parser.set_defaults(parser='list')

    add_parser = subparsers.add_parser('add', help=add_filters.__doc__)
    add_parser.add_argument('names', nargs='+', help="List of file names")
    add_parser.set_defaults(parser='add')

    plot_parser = subparsers.add_parser('plot', help=plot_filters.__doc__)
    plot_parser.add_argument('names', nargs='*', help="List of filter names")
    plot_parser.set_defaults(parser='plot')

    if len(sys.argv) == 1:
        parser.print_usage()
    else:
        args = parser.parse_args()
        if args.parser == 'list':
            list_filters()
        elif args.parser == 'add':
            add_filters(args.names)
        elif args.parser == 'plot':
            plot_filters(args.names)
