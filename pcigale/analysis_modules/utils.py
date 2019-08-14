# -*- coding: utf-8 -*-
# Copyright (C) 2014 Médéric Boquien
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""
Various utility functions for pcigale analysis modules
"""

import multiprocessing as mp
import time


class Counter:
    """Class to count the number of models computers/objects analysed. It has
    two internal counters. One is internal to the process and is incremented at
    each iteration. The other one is global is is only incremented
    periodically. The fundamental reason is that a lock is needed to increment
    the global value. When using many cores this can strongly degrade the
    performance. Similarly printing the number of iterations achieved and at
    what speed is only done periodically. For practical reasons the printing
    frequency has to be a multiple of the incrementation frequency of the
    global counter.
    """

    def __init__(self, nmodels, freq_inc=1, freq_print=1):
        if freq_print % freq_inc != 0:
            raise ValueError("The printing frequency must be a multiple of "
                             "the increment frequency.")
        self.nmodels = nmodels
        self.freq_inc = freq_inc
        self.freq_print = freq_print
        self.global_counter = mp.Value('i', 0)
        self.proc_counter = 0
        self.t0 = time.time()

    def inc(self):
        self.proc_counter += 1
        if self.proc_counter % self.freq_inc == 0:
            with self.global_counter.get_lock():
                self.global_counter.value += self.freq_inc
                n = self.global_counter.value
            if n % self.freq_print == 0:
                self.pprint(n)

    def pprint(self, n):
        dt = time.time() - self.t0
        print(f"{n}/{self.nmodels} performed in {dt:.1f} seconds ({n/dt:.1f}/s)",
              end="\n" if n == self.nmodels else "\r")
