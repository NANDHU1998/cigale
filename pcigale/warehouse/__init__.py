from ..sed import SED
from .. import sed_modules
from .sedcache import SedCache


class SedWarehouse:
    """Create, cache and store SED

    This object is responsible for creating SED and storing them in a memory
    cache or a database.
    """

    def __init__(self, nocache=None):
        """Instantiate an SED warehouse

        Parameters
        ----------
        nocache: list
            SED module or list of the SED modules that are not to be cached,
        trading CPU for memory.

        """
        if nocache is None:
            self.nocache = []
        elif isinstance(nocache, list) is True:
            self.nocache = nocache
        elif isinstance(nocache, str) is True:
            self.nocache = [nocache]
        else:
            raise TypeError("The nocache argument must be a list or an str.")

        self.sed_cache = SedCache()
        self.module_cache = {}

    def get_module_cached(self, name, **kwargs):
        """Get the SED module using the internal cache.

        Parameters
        ----------
        name: string
            Module name.

        The other keyword parameters are the module parameters.

        Returns
        -------
        a pcigale.sed_modules.Module instance

        """
        if name in self.nocache:
            module = sed_modules.get_module(name, **kwargs)
        else:
            # Use the name of the module and the values of the parameters as a
            # key to the cache. This works because the parameters are stored in
            # ordered dictionaries.
            module_key = (name, tuple(kwargs.values()))
            if module_key in self.module_cache:
                module = self.module_cache[module_key]
            else:
                module = sed_modules.get_module(name, **kwargs)
                self.module_cache[module_key] = module

        return module

    def get_sed(self, module_list, parameter_list):
        """Get the SED corresponding to the module and parameter lists

        If the SED was cached, get it from the cache. If it is not, create it
        and add it the the cache. The method is recursive to permit caching
        partial SED.

        Parameters
        ----------
        module_list: iterable
            List of module names in the order they have to be used to
            create the SED.
        parameter_list: iterable
            List of the parameter dictionaries corresponding to each
            module of the module_list list.

        Returns
        -------
        sed: pcigale.sed
            The SED made from the given modules with the given parameters.

        """
        module_list = list(module_list)
        parameter_list = list(parameter_list)

        # Use the values of the parameters of all the modules as the key for
        # the cache. This works because the parameters are stored in ordered
        # dictionaries.
        key = tuple(tuple(par.values()) for par in parameter_list)

        sed = self.sed_cache[key]
        if sed is None:
            mod = self.get_module_cached(
                module_list.pop(), **parameter_list.pop()
            )

            if len(module_list) == 0:
                sed = SED()
            else:
                sed = self.get_sed(module_list, parameter_list).copy()

            mod.process(sed)
            self.sed_cache[key] = sed

        return sed
