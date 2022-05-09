import os

# Set environment variables to disable multithreading as users will probably
# want to set the number of cores to the max of their computer.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import datetime as dt
import multiprocessing as mp
from pathlib import Path
import sys
import time

from .session.configuration import Configuration
from .analysis_modules import get_module
from pcigale.utils.info import Info
from pcigale.utils.console import console, INFO

from pcigale.version import __version__


def init(config):
    """Create a blank configuration file."""
    config.create_blank_conf()
    console.print(
        f"{INFO} The initial configuration file was created. Please complete "
        "it with the data file name and the pcigale modules to use."
    )


def genconf(config):
    """Generate the full configuration."""
    config.generate_conf()
    console.print(
        f"{INFO} The configuration file has been updated. Please complete the "
        "various module parameters and the data file columns to use in the "
        "analysis."
    )

    # Pass config rather than configuration as the file cannot be auto-filled.
    info = Info(config.config)
    info.print_tables()


def check(config):
    """Check the configuration."""
    configuration = config.configuration

    if configuration:
        info = Info(config.configuration)
        info.print_tables()


def run(config):
    """Run the analysis."""
    configuration = config.configuration

    if configuration:
        info = Info(config.configuration)
        info.print_tables()
        analysis_module = get_module(configuration["analysis_method"])

        start = dt.datetime.now()
        console.print(f"{INFO} Start: {start.isoformat('/', 'seconds')}")
        start = time.monotonic()  # Simpler time for run duration

        analysis_module.process(configuration)

        end = dt.datetime.now()
        console.print(f"{INFO} End: {end.isoformat('/', 'seconds')}")
        end = time.monotonic()

        delta = dt.timedelta(seconds=int(end - start))
        console.print(f"{INFO} Total duration: {delta}")


def main():
    Info.print_panel()
    if sys.version_info[:2] < (3, 8):
        raise Exception(
            f"Python {sys.version_info[0]}.{sys.version_info[1]} is "
            "unsupported. Please upgrade to Python 3.8 or later."
        )

    # We set the sub processes start method to spawn because it solves
    # deadlocks when a library cannot handle being used on two sides of a
    # forked process. This happens on modern Macs with the Accelerate library
    # for instance. On Linux we should be pretty safe with a fork, which allows
    # to start processes much more rapidly.
    if sys.platform.startswith("linux"):
        mp.set_start_method("fork")
    else:
        mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--conf-file",
        dest="config_file",
        help="Alternative configuration file to use.",
    )

    subparsers = parser.add_subparsers(help="List of commands")

    init_parser = subparsers.add_parser("init", help=init.__doc__)
    init_parser.set_defaults(parser="init")

    genconf_parser = subparsers.add_parser("genconf", help=genconf.__doc__)
    genconf_parser.set_defaults(parser="genconf")

    check_parser = subparsers.add_parser("check", help=check.__doc__)
    check_parser.set_defaults(parser="check")

    run_parser = subparsers.add_parser("run", help=run.__doc__)
    run_parser.set_defaults(parser="run")

    if len(sys.argv) == 1:
        parser.print_usage()
    else:
        args = parser.parse_args()

        if args.config_file:
            config = Configuration(Path(args.config_file))
        else:
            config = Configuration()

        if args.parser == "init":
            init(config)
        elif args.parser == "genconf":
            genconf(config)
        elif args.parser == "check":
            check(config)
        elif args.parser == "run":
            run(config)
