"""
Schreiber (2016) IR models module
=====================================

This module implements the Schreiber et al. (2016) infra-red models.

"""

import numpy as np
from pcigale.data import SimpleDatabase as Database
from . import SedModule


class Schreiber2016(SedModule):
    """Schreiber et al. (2016) templates IR re-emission module

    Given an amount of attenuation (e.g. resulting from the action of a dust
    attenuation module) this module normalises the Schreiber et al. (2016)
    template corresponding to a given α to this amount of energy and add it
    to the SED.

    """

    parameter_list = {
        'tdust': (
            'cigale_list(options=15. & 16. & 17. & 18. & 19. & 20. & 21. & '
            '22. & 23. & 24. & 25. & 26. & 27. & 28. & 29. & 30. & 31. & '
            '32. & 33. & 34. & 35. & 36. & 37. & 38. & 39. & 40. & 41. & '
            '42. & 43. & 44. & 45. & 46. & 47. & 48. & 49. & 50. & 51. & '
            '52. & 53. & 54. & 55. & 56. & 57. & 58. & 59. & 60.)',
            "Dust temperature. "
            "Between 15 and 60K, with 1K step.",
            20.
        ),
        'fpah': (
            'cigale_list(minvalue=0., maxvalue=1.)',
            "Mass fraction of PAH. "
            "Between 0 and 1.",
            0.05
        )
    }

    def _init_code(self):
        """Get the model out of the database"""

        self.tdust = float(self.parameters["tdust"])
        self.fpah = float(self.parameters["fpah"])
        with Database("schreiber2016") as db:
            self.model_dust = db.get(type=0, tdust=self.tdust)
            self.model_pah = db.get(type=1, tdust=self.tdust)

        # The models in memory are in W/nm/kg. At the same time we
        # need to normalize them to 1 W here to easily scale them from the
        # power absorbed in the UV-optical. If we want to retrieve the dust
        # mass at a later point, we have to save their "emissivity" per unit
        # mass in W kg¯¹, The gamma parameter does not affect the fact that it
        # is for 1 kg because it represents a mass fraction of each component.

        self.emissivity = np.trapz((1. - self.fpah) * self.model_dust.spec +
                                   self.fpah * self.model_pah.spec,
                                   x=self.model_dust.wl)

        # We want to be able to display the respective contributions of both
        # components, therefore we keep they separately.
        self.model_dust.spec *= (1. - self.fpah) / self.emissivity
        self.model_pah.spec *= self.fpah / self.emissivity

    def process(self, sed):
        """Add the IR re-emission contributions

        Parameters
        ----------
        sed: pcigale.sed.SED object
        parameters: dictionary containing the parameters

        """
        if 'dust.luminosity' not in sed.info.keys():
            sed.add_info('dust.luminosity', 1., True, unit='W')
        luminosity = sed.info['dust.luminosity']

        sed.add_module(self.name, self.parameters)
        sed.add_info('dust.tdust', self.tdust, unit='K')
        sed.add_info('dust.fpah', self.fpah)
        # To compute the dust mass we simply divide the luminosity by the
        # emissivity and then by the expected MH/Mdust as the emissivity was
        # computed for 1 kg of H. Note that we take 100 here but it should vary
        # with the exact model. Fix that later. Maybe directly in the database.
        sed.add_info('dust.mass', luminosity / self.emissivity, True, unit='kg')

        sed.add_contribution('dust.dust_continuum', self.model_dust.wl,
                             luminosity * self.model_dust.spec)
        sed.add_contribution('dust.pah', self.model_pah.wl,
                             luminosity * self.model_pah.spec)


# SedModule to be returned by get_module
Module = Schreiber2016
