"""This is the database where we store some data used by pcigale."""

from pathlib import Path
import pickle
import traceback

import pkg_resources


class SimpleDatabaseEntry:
    """Entry in SimpleDatabase object."""

    def __init__(self, primarykeys, data):
        """Create a dynamically-constructed object. The primary keys and the
        data are passed through two dictionaries. Each key of each dictionary
        is then transformed into an attribute to which the correspond value is
        assigned.

        Parameters
        ----------
        primarykeys: dict
            Dictionary containing the primary keys (e.g., metallicity, etc.)
        data: dict
            Dictionary containing the data (e.g., wavelength, spectrum, etc.)

        """
        for k, v in {**primarykeys, **data}.items():
            setattr(self, k, v)


class SimpleDatabase:
    """Simple database that can contain any data. It is entirely dynamic and
    does not require the database format to be declared. It is created
    on-the-fly when importing the data. The mechanism is that the primary keys
    and the data are passed through two dictionaries. These dictionaries are
    transformed into a SimpleDatabaseEntry where each key corresponds to an
    attribute. This allows to eliminate much of the boilerplate code that is
    needed for an SqlAlchemy database. Each SimpleDatabaseEntry object is saved
    as a pickle file in a directory of the name of the database. So that it is
    straightforward to retrieve the pickle file corresponding to a given set of
    primary keys, the name contains the values of the primary keys. While this
    is very fast, it requires the use to always make queries using the same data
    types for each key (though different keys can have different types). Overall
    a SimpleDatabase is much easier to handle than an SqlAlchemy database.
    """

    def __init__(self, name, writable=False):
        """Prepare the database. Each database is stored in a directory of the
        same name and each entry is a pickle file. We store a specific pickle
        file named parameters.pickle, which is a dictionary that contains the
        values taken by each parameter as a list.

        Parameters
        ----------
        name: str
            Name of the database
        writable: bool
            Flag whether the database should be open as read-only or in write
            mode
        """
        self.name = name
        self.writable = writable
        self.path = Path(pkg_resources.resource_filename(__name__, name))

        if writable is True and self.path.is_dir() is False:
            # Everything looks fine, so we create the database and save a stub
            # of the parameters dictionary.
            self.path.mkdir()

            self.parameters = {}
            with open(self.path / "parameters.pickle", "wb") as f:
                pickle.dump(self.parameters, f)

        # We load the parameters dictionary. If this fails it is likely that
        # something went wrong and it needs to be rebuilt.
        try:
            with open(self.path / "parameters.pickle", "rb") as f:
                self.parameters = pickle.load(f)
        except Exception:
            raise Exception(
                f"The database {self.name} appears corrupted. "
                f"Erase {self.path} and rebuild it."
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)

        self.close()

    def close(self):
        """Close the database and save the parameters dictionary if the database
        was writable.
        """
        if self.writable is True:
            # Eliminate duplicated parameter values and we save the dictionary.
            for k, v in self.parameters.items():
                self.parameters[k] = list(set(v))

            with open(self.path / "parameters.pickle", "wb") as f:
                pickle.dump(self.parameters, f)

    def add(self, primarykeys, data):
        """Add an entry to the database. The primary keys and the data are used
        to instantiate a SimpleDatabaseEntry object, which is then saved as a
        pickle file. The name of the file is constructed from the names and
        values of the primary keys.

        Parameters
        ----------
        primarykeys: dict
            Dictionary containing the primary keys (e.g., metallicity, etc.)
        data: dict
            Dictionary containing the data (e.g., wavelength, spectrum, etc.)
        """
        if self.writable is False:
            raise Exception(f"The database {self.name} is read-only.")

        entry = SimpleDatabaseEntry(primarykeys, data)
        basename = "_".join(f"{k}={v}" for k, v in sorted(primarykeys.items()))

        with open(self.path / Path(f"{basename}.pickle"), "wb") as f:
            pickle.dump(entry, f)

        if len(self.parameters) == 0:  # Create the initial lists
            for k, v in primarykeys.items():
                self.parameters[k] = [v]
        else:
            for k, v in primarykeys.items():
                self.parameters[k].append(v)

    def get(self, **primarykeys):
        """Get an entry from the database. This is done by loading a pickle file
        whose name is constructed from the names values of the primary keys. It
        is important that for each key the same type is used for adding and
        getting an entry.

        Parameters
        ----------
        primarykeys: keyword argument
            Primary key names and values

        Returns
        -------
        entry: SimpleDatabaseEntry
            Object containing the primary keys (e.g., metallicity, etc.) and the
            data (e.g., wavelength, spectrum, etc.).
        """
        basename = "_".join(f"{k}={v}" for k, v in sorted(primarykeys.items()))

        try:
            with open(self.path / Path(f"{basename}.pickle"), "rb") as f:
                entry = pickle.load(f)
        except Exception:
            raise Exception(
                f"Cannot read model {primarykeys}. Either the parameters were "
                "parameters were passed incorrectly or the database has not "
                "been built correctly."
            )

        return entry
