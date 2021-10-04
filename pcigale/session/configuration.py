import multiprocessing as mp
from pathlib import Path
import sys
from textwrap import wrap

import configobj
from glob import glob  # To allow the use of glob() in "eval..."
import numpy as np
import validate

from ..managers.parameters import ParametersManager
from ..data import SimpleDatabase as Database
from utils.io import read_table
from .. import sed_modules
from .. import analysis_modules
from ..warehouse import SedWarehouse
from . import validation
from pcigale.sed_modules.nebular import default_lines
from pcigale.utils.console import console, INFO, WARNING, ERROR

class Configuration:
    """This class manages the configuration of pcigale.
    """

    def __init__(self, filename=Path("pcigale.ini")):
        """Initialise a pcigale configuration.

        Parameters
        ----------
        filename: Path
            Name of the configuration file (pcigale.conf by default).

        """
        # We should never be in the case when there is a pcigale.ini but no
        # pcigale.ini.spec. While this seems to work when doing the pcigale
        # genconf, it actually generates an incorrect pcigale.ini.spec. The only
        # clean solution is to rebuild both files.
        if filename.is_file() and not filename.with_suffix('.ini.spec').is_file():
            raise Exception("The pcigale.ini.spec file appears to be missing. "
                            "Please delete the pcigale.ini file and regenrate "
                            "it with 'pcigale init' and then 'pcigale genconf' "
                            "after having filled the initial pcigale.ini "
                            "template.")
        else:
            self.spec = configobj.ConfigObj(filename.with_suffix('.ini.spec').name,
                                            write_empty_values=True,
                                            indent_type='  ',
                                            encoding='UTF8',
                                            list_values=False,
                                            _inspec=True)
            self.config = configobj.ConfigObj(filename.name,
                                              write_empty_values=True,
                                              indent_type='  ',
                                              encoding='UTF8',
                                              configspec=self.spec)

        # We validate the configuration so that the variables are converted to
        # the expected that. We do not handle errors at the point but only when
        # we actually return the configuration file from the property() method.
        self.config.validate(validate.Validator(validation.functions))

        self.pcigaleini_exists = filename.is_file()

    def create_blank_conf(self):
        """Create the initial configuration file

        Write the initial pcigale configuration file where the user can state
        which data file to use, which modules to use for the SED creation, as
        well as the method selected for statistical analysis.

        """

        self.config['data_file'] = ""
        self.config.comments['data_file'] = wrap(
            "File containing the input data. The columns are 'id' (name of the"
            " object), 'redshift' (if 0 the distance is assumed to be 10 pc), "
            "'distance' (Mpc, optional, if present it will be used in lieu "
            "of the distance computed from the redshift), the filter names for "
            "the fluxes, and the filter names with the '_err' suffix for the "
            "uncertainties. The fluxes and the uncertainties must be in mJy "
            "for broadband data and in W/m² for emission lines. This file is "
            "optional to generate the configuration file, in particular for "
            "the savefluxes module.")
        self.spec['data_file'] = "string()"

        self.config['parameters_file'] = ""
        self.config.comments['parameters_file'] = [""] + wrap(
            "Optional file containing the list of physical parameters. Each "
            "column must be in the form module_name.parameter_name, with each "
            "line being a different model. The columns must be in the order "
            "the modules will be called. The redshift column must be the last "
            "one. Finally, if this parameter is not empty, cigale will not "
            "interpret the configuration parameters given in pcigale.ini. "
            "They will be given only for information. Note that this module "
            "should only be used in conjonction with the savefluxes module. "
            "Using it with the pdf_analysis module will yield incorrect "
            "results.")
        self.spec['parameters_file'] = "string()"

        self.config['sed_modules'] = []
        self.config.comments['sed_modules'] = ([""] +
            ["Available modules to compute the models. The order must be kept."
             ] +
            ["SFH:"] +
            ["* sfh2exp (double exponential)"] +
            ["* sfhdelayed (delayed SFH with optional exponential burst)"] +
            ["* sfhdelayedbq (delayed SFH with optional constant burst/quench)"
             ] +
            ["* sfhfromfile (arbitrary SFH read from an input file)"] +
            ["* sfhperiodic (periodic SFH, exponential, rectangle or delayed"
             ")"] +
            ["SSP:"] +
            ["* bc03 (Bruzual and Charlot 2003)"] +
            ["* m2005 (Maraston 2005; note that it cannot be combined with the "
             "nebular module)"] +
            ["Nebular emission:"] +
            ["* nebular (continuum and line nebular emission)"] +
            ["Dust attenuation:"] +
            ["* dustatt_modified_CF00 (modified Charlot & Fall 2000 "
             "attenuation law)"] +
            ["* dustatt_modified_starburst (modified Calzetti 2000 attenuaton "
             "law)"] +
            ["Dust emission:"] +
            ["* casey2012 (Casey 2012 dust emission models)"] +
            ["* dale2014 (Dale et al. 2014 dust emission templates)"] +
            ["* dl2007 (Draine & Li 2007 dust emission models)"] +
            ["* dl2014 (Draine et al. 2014 update of the previous models)"] +
            ["* themis (Themis dust emission models from Jones et al. 2017)"] +
            ["AGN:"] +
            ["* skirtor2016 (AGN models from Stalevski et al. 2012, 2016)"] +
            ["* fritz2006 (AGN models from Fritz et al. 2006)"] +
            ["X-ray:"] +
            ["* xray (from AGN and galaxies; skirtor2016/fritz2006 is needed for AGN)"] +
            ["Radio:"] +
            ["* radio (galaxy synchrotron emission and AGN; skirtor2016/fritz2006 "
             "is needed for AGN)"] +
            ["Restframe parameters:"] +
            ["* restframe_parameters (UV slope (β), IRX, D4000, EW, etc.)"] +
            ["Redshift+IGM:"] +
            ["* redshifting (mandatory, also includes the IGM from Meiksin "
             "2006)"]
        )
        self.spec['sed_modules'] = "cigale_string_list()"

        self.config['analysis_method'] = ""
        self.config.comments['analysis_method'] = [""] + wrap(
            "Method used for statistical analysis. Available methods: "
            "pdf_analysis, savefluxes.")
        self.spec['analysis_method'] = "string()"

        self.config['cores'] = ""
        self.config.comments['cores'] = [""] + wrap(
            f"Number of CPU cores available. This computer has "
            f"{mp.cpu_count()} cores.")
        self.spec['cores'] = "integer(min=1)"

        self.config.write()
        self.spec.write()

    def generate_conf(self):
        """Generate the full configuration file

        Reads the user entries in the initial configuration file and add the
        configuration options of all selected modules as well as the filter
        selection based on the filters identified in the data table file.

        """
        if self.pcigaleini_exists is False:
            raise Exception("pcigale.ini could not be found.")

        # Getting the list of the filters available in pcigale database
        with Database("filters") as db:
            filter_list = db.parameters["name"]
        filter_list += [f'line.{line}' for line in default_lines]

        if self.config['data_file'] != '':
            obs_table = read_table(self.config['data_file'])

            # Check that the the file was correctly read and that the id and
            # redshift columns are present in the input file
            if 'col1' in obs_table.columns:
                raise Exception("The input could not be read properly. Verify "
                                "its format and that it does not have two "
                                "columns with the same name.")
            if 'id' not in obs_table.columns:
                raise Exception("Column id not present in input file")
            if 'redshift' not in obs_table.columns:
                raise Exception("Column redshift not present in input file")

            # Finding the known filters in the data table
            bands = []
            for band in obs_table.columns:
                filter_name = band[:-4] if band.endswith('_err') else band
                if filter_name in filter_list:
                    bands.append(band)

            # Check that we don't have an band error without the associated
            # band
            for band in bands:
                if band.endswith('_err') and (band[:-4] not in bands):
                    raise Exception(f"The observation table as a {band} column "
                                    f"but no {band[:-4]} column.")

            self.config['bands'] = bands
        else:
            self.config['bands'] = ''
        self.config.comments['bands'] = [""] + wrap("Bands to consider. To "
            "consider uncertainties too, the name of the band must be "
            "indicated with the _err suffix. For instance: FUV, FUV_err.")
        self.spec['bands'] = "cigale_string_list()"

        self.config['properties'] = ''
        self.config.comments['properties'] = [""] + wrap("Properties to be "
            "considered. All properties are to be given in the rest frame "
            "rather than the observed frame. This is the case for instance "
            "the equivalent widths and for luminosity densities.")
        self.spec['properties'] = "cigale_string_list()"

        # SED creation modules configurations. For each module, we generate
        # the configuration section from its parameter list.
        self.config['sed_modules_params'] = {}
        self.config.comments['sed_modules_params'] = ["", ""] + wrap(
            "Configuration of the SED creation modules.")
        self.spec['sed_modules_params'] = {}

        for module_name in self.config['sed_modules']:
            self.config['sed_modules_params'][module_name] = {}
            self.spec['sed_modules_params'][module_name] = {}
            sub_config = self.config['sed_modules_params'][module_name]
            sub_spec = self.spec['sed_modules_params'][module_name]

            for name, (typ, description, default) in \
                    sed_modules.get_module(
                        module_name,
                        blank=True).parameter_list.items():
                if default is None:
                    default = ''
                sub_config[name] = default
                sub_config.comments[name] = wrap(description)
                sub_spec[name] = typ
            self.config['sed_modules_params'].comments[module_name] = [
                sed_modules.get_module(module_name, blank=True).comments]

        self.check_modules()

        # Configuration for the analysis method
        self.config['analysis_params'] = {}
        self.config.comments['analysis_params'] = ["", ""] + wrap(
            "Configuration of the statistical analysis method.")
        self.spec['analysis_params'] = {}

        module_name = self.config['analysis_method']
        for name, (typ, desc, default) in \
                analysis_modules.get_module(module_name).parameter_list.items():
            if default is None:
                default = ''
            self.config['analysis_params'][name] = default
            self.config['analysis_params'].comments[name] = wrap(desc)
            self.spec['analysis_params'][name] = typ

        if 'pdf_analysis' == module_name:
            bands = [band for band in self.config['bands']
                     if not band.endswith('_err')]
            self.config['analysis_params']['bands'] = bands

        self.config.write()
        self.spec.write()

    @property
    def configuration(self):
        """Returns a dictionary for the session configuration if it is valid.
        Otherwise, print the erroneous keys.

        Returns
        -------
        configuration: dictionary
            Dictionary containing the information provided in pcigale.ini.
        """
        if self.pcigaleini_exists is False:
            raise Exception("pcigale.ini could not be found.")

        self.complete_redshifts()
        self.check_and_complete_analysed_parameters()

        vdt = validate.Validator(validation.functions)
        validity = self.config.validate(vdt, preserve_errors=True)

        if validity is not True:
            console.print(f"{ERROR} The following issues have been found in "
                          "pcigale.ini:")
            for module, param, message in configobj.flatten_errors(self.config,
                                                                   validity):
                if len(module) > 0:
                    console.print(f"{ERROR} Module [b]{'/'.join(module)}[/b], "
                                  f"parameter [b]{param}[/b]: {message}")
                else:
                    console.print(f"{ERROR} Parameter [b]{param}[/b]:"
                                  f"{message}")
            console.print(f"{INFO} Run the same command after having fixed "
                          "pcigale.ini.")

            return None

        return self.config.copy()

    def check_modules(self):
        """Make a basic check to ensure that some required modules are present.
        Otherwise we emit a warning so the user knows their list of modules is
        suspicious. We do not emit an exception as they may be using an
        unofficial module that is not in our list
        """

        modules = {'SFH': ['sfh2exp', 'sfhdelayed', 'sfhdelayedbq',
                           'sfhfromfile', 'sfhperiodic'],
                   'SSP': ['bc03', 'm2005'],
                   'nebular': ['nebular'],
                   'dust attenuation': ['dustatt_calzleit', 'dustatt_powerlaw',
                                        'dustatt_2powerlaws',
                                        'dustatt_modified_CF00',
                                        'dustatt_modified_starburst'],
                   'dust emission': ['casey2012', 'dale2014', 'dl2007',
                                     'dl2014', 'themis'],
                   'AGN': ['fritz2006', 'skirtor2016'],
                   'X-ray': ['xray'],
                   'radio': ['radio'],
                   'restframe_parameters': ['restframe_parameters'],
                   'redshift': ['redshifting']
                   }

        comments = {'SFH': f"{WARNING} No SFH module selected. Is it on "
                           "purpose?",
                    'SSP': f"{WARNING} No SSP module selected. Is it on "
                           "purpose?",
                    'nebular': f"{INFO} No nebular module selected. Without it "
                               "the Lyman continuum is left untouched.",
                    'dust attenuation': f"{INFO} No dust attenuation module "
                                        "selected.",
                    'dust emission': f"{INFO} No dust emission module "
                                     "selected.",
                    'AGN': f"{INFO} No AGN module selected.",
                    'X-ray': f"{INFO} No X-ray module selected.",
                    'radio': f"{INFO} No radio module selected.",
                    'restframe_parameters': f"{INFO} No restframe parameters "
                                            "module selected.",
                    'redshift': f"{ERROR} No redshifting module found."}

        for module in modules:
            if all([user_module not in modules[module] for user_module in
                    self.config['sed_modules']]):
                console.print(f"{comments[module]} Options are: [b]"
                              f"{'[/b], [b]'.join(modules[module])}[/b].")

    def complete_redshifts(self):
        """Complete the configuration when the redshifts are missing from the
        configuration file and must be extracted from the input flux file.
        """

        z_mod = self.config['sed_modules_params']['redshifting']['redshift']
        if type(z_mod) is str and not z_mod:
            if self.config['data_file']:
                obs_table = read_table(self.config['data_file'])
                if 'redshift_decimals' in self.config['analysis_params']:
                    decimals = self.config['analysis_params']['redshift_decimals']
                    if decimals < 0:
                        z = list(np.unique(obs_table['redshift']))
                    else:
                        z = list(np.unique(np.around(obs_table['redshift'],
                                                     decimals=decimals)))
                else:
                    z = list(np.unique(obs_table['redshift']))
                self.config['sed_modules_params']['redshifting']['redshift'] = z
            elif self.config['parameters_file']:
                # The entry will be ignored anyway. Just pass a dummy list
                self.config['sed_modules_params']['redshifting']['redshift'] = []
            else:
                raise Exception("No flux file and no redshift indicated. "
                                "The spectra cannot be computed. Aborting.")

    def check_and_complete_analysed_parameters(self):
        """Check that the variables to be analysed are indeed computed and if "
        "no variable is given, complete the configuration with all the "
        "variables extracted from a dummy run."""
        params = ParametersManager(self.config.dict())
        sed = SedWarehouse().get_sed(params.modules, params.from_index(0))
        info = list(sed.info.keys())

        if len(self.config['analysis_params']['variables']) > 0:
            nolog = [k if not k.endswith('_log') else k[:-4]
                     for k in self.config['analysis_params']['variables']]
            diff = set(nolog) - set(info)
            if len(diff) > 0:
                raise Exception(f"{', '.join(diff)} unknown. "
                                f"Available variables are: {', '.join(info)}.")
        else:
            info.sort()
            self.config['analysis_params']['variables'] = info
