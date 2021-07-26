import os
# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import multiprocessing as mp
from pathlib import Path

from pcigale.session.configuration import Configuration
from .plot_types.chi2 import chi2 as chi2_action
from .plot_types.pdf import pdf as pdf_action
from .plot_types.sed import sed as sed_action, AVAILABLE_SERIES
from .plot_types.mock import mock as mock_action

__version__ = "0.2-alpha"


def parser_range(range_str):
    """
    Auxiliary parser for plot X and Y ranges
    :param range_str: a string like [<min>]:[<max>]
    :return:
    """
    rmin, rmax = range_str.split(':')
    try:
        rmin = float(rmin) if rmin else False
        rmax = float(rmax) if rmax else False
    except ValueError:
        msg = '{} has not the format [<min>]:[<max>], where ' \
              'min and max are either float or empty'.format(range_str)
        raise argparse.ArgumentTypeError(msg)
    return rmin, rmax


def main():
    if sys.version_info[:2] < (3, 8):
        raise Exception(f"Python {sys.version_info[0]}.{sys.version_info[1]} is"
                        f" unsupported. Please upgrade to Python 3.8 or later.")

    # We set the sub processes start method to spawn because it solves
    # deadlocks when a library cannot handle being used on two sides of a
    # forked process. This happens on modern Macs with the Accelerate library
    # for instance. On Linux we should be pretty safe with a fork, which allows
    # to start processes much more rapidly.
    if sys.platform.startswith('linux'):
        mp.set_start_method('fork')
    else:
        mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--conf-file', dest='config_file',
                        help="Alternative configuration file to use.")

    subparsers = parser.add_subparsers(help="List of commands")

    fmtstr = 'format of the output files, for instance pdf or png.'
    pdf_parser = subparsers.add_parser('pdf', help=pdf_action.__doc__)
    pdf_parser.add_argument('--format', default='pdf', help=fmtstr)
    pdf_parser.add_argument('--outdir', default='out')
    pdf_parser.set_defaults(parser='pdf')

    chi2_parser = subparsers.add_parser('chi2', help=chi2_action.__doc__)
    chi2_parser.add_argument('--format', default='pdf', help=fmtstr)
    chi2_parser.add_argument('--outdir', default='out')
    chi2_parser.set_defaults(parser='chi2')

    sed_parser = subparsers.add_parser('sed', help=sed_action.__doc__)
    sed_parser.add_argument('--type', default='mJy',
                            help='type of plot. Options are mJy (observed '
                                 'frame in flux) and lum (rest-frame in '
                                 'lumunosity).')
    sed_parser.add_argument('--nologo', action='store_true')
    sed_parser.add_argument('--format', default='pdf', help=fmtstr)
    sed_parser.add_argument('--outdir', default='out')
    sed_parser.add_argument('--xrange', default=':', type=parser_range,
                            help='Wavelength range [<min>]:[<max>] in μm.')
    sed_parser.add_argument('--yrange', default=':', type=parser_range,
                            help='y-axis range [<min>]:[<max>].')
    sed_parser.add_argument('--series', nargs='*',
                            help='components to plot. Options are: ' +
                                 ', '.join(AVAILABLE_SERIES) + '.')
    sed_parser.add_argument('--seriesdisabled',
                            help='components not to plot. Options are: ' +
                                 ', '.join(AVAILABLE_SERIES) + '.',
                            action='store_true')
    sed_parser.set_defaults(parser='sed')

    mock_parser = subparsers.add_parser('mock', help=mock_action.__doc__)
    mock_parser.add_argument('--nologo', action='store_true')
    mock_parser.add_argument('--format', default='pdf', help=fmtstr)
    mock_parser.add_argument('--outdir', default='out')
    mock_parser.set_defaults(parser='mock')

    args = parser.parse_args()
    outdir = Path(args.outdir)

    if args.config_file:
        config = Configuration(Path(args.config_file))
    else:
        config = Configuration(outdir / 'pcigale.ini')

    if len(sys.argv) == 1:
        parser.print_usage()
    else:
        if args.parser == 'chi2':
            chi2_action(config, args.format, outdir)
        elif args.parser == 'pdf':
            pdf_action(config, args.format, outdir)
        elif args.parser == 'sed':
            if not args.series:
                series = AVAILABLE_SERIES
            else:
                if args.seriesdisabled:
                    series = [series for series in AVAILABLE_SERIES
                              if series not in args.series]
                else:
                    series = args.series
            sed_action(config, args.type, args.nologo, args.xrange, args.yrange,
                       series, args.format, outdir)
        elif args.parser == 'mock':
            mock_action(config, args.nologo, outdir)
