"""
Various utility function for pcigale
"""
from functools import lru_cache

from astropy.table import Table
from astropy.io.ascii.core import InconsistentTableError

@lru_cache
def read_table(file_):
    """Read a data table using astropy

    This function first tries to automatically read the table with astropy,
    if that fails, it tries with the ascii format.

    All the integer columns, except the "id" column, are converted to float.

    Parameters
    ----------
    file_: string
        Name of the file to read.

    Return
    ------
    an astropy.table.Table object

    Raise
    -----
    An error is raised when the table can not be parsed.

    """
    try:
        table = Table.read(file_)
    except Exception:  # astropy should raise a specific exception
        try:
            table = Table.read(file_, format="ascii", delimiter='\s')
        except InconsistentTableError:
            raise Exception(f"The file {file_} can not be parsed as a data "
                            "table.")

    # Convert all the integers to floats.
    return Table([
        col.astype(float) if col.name != 'id' and col.dtype == int
        else col
        for col in table.columns.values()
    ])
