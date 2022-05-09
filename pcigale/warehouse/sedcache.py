class SedCache:
    """Minimal class to cache partially computed SED. The central idea is that
    at any given time we only to store one SED for any given set of modules.
    If the SED in the cache does not correspond to the right parameters, then
    it means we have moved to a new set of parameters for this set of modules
    and the cached SED can be discarded. To avoid lengthy computations we just
    cache the SED in a dictionary with the number of modules as the key. The
    parameters are also save with the SED in a tuple to allow
    """

    def __init__(self):
        self.dict = {}

    def __setitem__(self, key, value):
        self.dict[len(key)] = (key, value)

    def __getitem__(self, key):
        try:
            val = self.dict[len(key)]
            if val[0] == key:
                return val[1]
            else:
                return None
        except KeyError:
            return None
