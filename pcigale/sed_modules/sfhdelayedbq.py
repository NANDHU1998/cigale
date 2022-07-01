"""
Delayed tau model for star formation history with an optional burst/quench
==========================================================================

This module implements a star formation history (SFH) described as a delayed
rise of the SFR up to a maximum, followed by an exponential decrease. Optionally
a quenching or a bursting episode can be added. It is described in more detail
in Ciesla et al. (2017).

"""

import numpy as np

from . import SedModule

__category__ = "SFH"


class SFHDelayedBQ(SedModule):
    """Delayed tau model for Star Formation History with an optional burst or
    quench.

    This module sets the SED star formation history (SFH) proportional to time,
    with a declining exponential parametrised with a time-scale τ. Optionally
    a burst/quench can be added. In that case the SFR of that episode is
    constant and parametrised as a ratio of the SFR before the beginning of the
    episode. See Ciesla et al. (2017).

    """

    parameter_list = {
        "tau_main": (
            "cigale_list()",
            "e-folding time of the main stellar population model in Myr.",
            2000.
        ),
        "age_main": (
            "cigale_list(dtype=int, minvalue=0.)",
            "Age of the main stellar population in the galaxy in Myr. The "
            "precision is 1 Myr.",
            5000
        ),
        "age_bq": (
            "cigale_list(dtype=int)",
            "Age of the burst/quench episode. The precision is 1 Myr.",
            500.
        ),
        "r_sfr": (
            "cigale_list(minvalue=0.)",
            "Ratio of the SFR after/before age_bq.",
            0.1
        ),
        "sfr_A": (
            "cigale_list(minvalue=0.)",
            "Multiplicative factor controlling the SFR if normalise is False. "
            "For instance without any burst/quench: SFR(t)=sfr_A×t×exp(-t/τ)/τ²",
            1.
        ),
        "normalise": (
            "boolean()",
            "Normalise the SFH to produce one solar mass.",
            True
        )
    }

    def _init_code(self):
        self.tau_main = float(self.parameters["tau_main"])
        self.age_main = int(self.parameters["age_main"])
        self.age_bq = int(self.parameters["age_bq"])
        self.r_sfr = float(self.parameters["r_sfr"])
        sfr_A = float(self.parameters["sfr_A"])
        if isinstance(self.parameters["normalise"], str):
            normalise = self.parameters["normalise"].lower() == 'true'
        else:
            normalise = bool(self.parameters["normalise"])

        # Delayed SFH
        t = np.arange(self.age_main)
        self.sfr = t * np.exp(-t / self.tau_main) / self.tau_main**2

        # Add the burst/quench
        t_bq = self.age_main - self.age_bq
        self.sfr[t >= t_bq] = self.r_sfr * self.sfr[t_bq - 1]

        # Compute the galaxy mass and normalise the SFH to 1 solar mass
        # produced if asked to.
        self.sfr_integrated = np.sum(self.sfr) * 1e6
        if normalise:
            self.sfr /= self.sfr_integrated
            self.sfr_integrated = 1.
        else:
            self.sfr *= sfr_A
            self.sfr_integrated *= sfr_A

    def process(self, sed):
        """
        Parameters
        ----------
        sed : pcigale.sed.SED object

        """

        sed.add_module(self.name, self.parameters)

        # Add the sfh and the output parameters to the SED.
        sed.sfh = self.sfr
        sed.add_info("sfh.integrated", self.sfr_integrated, True,
                     unit='solMass')
        sed.add_info("sfh.age_main", self.age_main, unit='Myr')
        sed.add_info("sfh.tau_main", self.tau_main, unit='Myr')
        sed.add_info("sfh.age_bq", self.age_bq, unit='Myr')
        sed.add_info("sfh.r_sfr", self.r_sfr)


# CreationModule to be returned by get_module
Module = SFHDelayedBQ
