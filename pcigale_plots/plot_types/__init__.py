from abc import ABC, abstractmethod
import multiprocessing as mp

from pcigale.utils.counter import Counter


class Plotter(ABC):
    @abstractmethod
    def initializer(counter):
        pass

    @abstractmethod
    def worker(self):
        pass

    def _parallel_job(self, items, counter):
        if self.configuration["cores"] == 1:  # Do not create a new process
            self.initializer(counter)
            for item in items:
                self.worker(*item)
        else:  # run in parallel
            # Temporarily remove the counter sub-process that updates the
            # progress bar as it cannot be pickled when creating the parallel
            # processes when using the "spawn" starting method.
            progress = counter.progress
            counter.progress = None

            with mp.Pool(
                processes=self.configuration["cores"],
                initializer=self.initializer,
                initargs=(counter,),
            ) as pool:
                pool.starmap(self.worker, items, 1)

            # After the parallel processes have exited, it can be restored
            counter.progress = progress
