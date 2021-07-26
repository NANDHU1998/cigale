"""
Multiprocessing counter class with utility functions for periodic updating and
printing.
"""

import multiprocessing as mp
from queue import Empty
from time import sleep

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Text,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)

from pcigale.utils.console import console


class SpeedColumn(ProgressColumn):
    """Renders speed."""

    def render(self, task: "Task") -> Text:
        """Show speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            speed = 0.0
        return Text(f"{speed:.1f}/s", style="progress.data.speed")


class Counter:
    """Class to count the number of models computers/objects analysed. It has
    two internal counters. One is internal to the process and is incremented at
    each iteration. The other one is global and is only incremented
    periodically. The fundamental reason is that a lock is needed to increment
    the global value. When using many cores this can strongly degrade the
    performance. The user-visible progress bar is updated every 0.1 s based on
    the value of the global counter. This is done automatically in a dedicated
    process so that the process in charge of the computation only have to update
    the counter and nothing more.
    """

    def __init__(self, nmodels, freq_inc=1, text=""):
        self.nmodels = nmodels
        self.freq_inc = freq_inc
        self.text = text
        self.message = mp.Queue()
        self.global_counter = mp.Value("i", 0)
        self.local_counter = 0
        self.progress = mp.Process(target=self.update, daemon=True)
        self.progress.start()

    def inc(self):
        self.local_counter += 1
        if self.local_counter % self.freq_inc == 0:
            with self.global_counter.get_lock():
                self.global_counter.value += self.freq_inc

    def update(self):
        with Progress(
            SpinnerColumn(spinner_name="moon", finished_text=":thumbs_up:"),
            TextColumn("{task.description} {task.completed}/{task.total}"),
            BarColumn(None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}% "),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            SpeedColumn(),
            expand=True,
            console=console,
        ) as progress:
            task = progress.add_task(self.text, total=self.nmodels)

            while not progress.finished:
                progress.update(task, completed=self.global_counter.value)
                # The refresh rate of the progress bar is 10 Hz. There is no
                # need to update it more frequently as this would needlessly
                # take some CPU time.
                sleep(0.1)

        # We display the messages only at the end. The reason is that otherwise
        # it would block the progress bar if they are too frequent. We also
        # display the messages after the progress bar, otherwise it is much
        # slower.
        while True:
            try:
                console.print(self.message.get_nowait())
            except Empty:
                break
