import numpy as np

from .utils import weighted_param


def functions(mode, n, a1, a2):
    it = np.arange(n)

    if mode == "min":
        return a1 * (a2 / a1) ** (it / n)
    if mode == "exp":
        return a1 * np.exp(a2 * it / n)
    if mode == "expsquare":
        return a1 * np.exp(a2 * (it / n) ** 2.0)
    if mode == "linear":
        return (a1 - a2) * (1.0 - it / n) + a2
    if mode == "inverse":
        return a1 / it
    if mode == "root":
        return a1 ** (it / n)

    raise ValueError(f"Function {mode} unknown.")


class Map:
    def __init__(self, shape, metric, topology, periodic, seed=0):
        self.shape = shape
        self.topology = topology
        self.metric = metric
        self.periodic = periodic
        self.seed = seed
        rng = np.random.default_rng(seed)

        # The weights will be created when initiating the training
        self.weights = {}

        # In order to speed up the computations, we keep the square of the
        # distances from the neuron at location (0, …, 0)/ This allows to obtain
        # the distance between one neuron and all the other neurons by simply
        # roll'ing the array along each dimension.
        self.dist2 = self._dist2()

    def _dist2(self):
        if self.periodic is True:
            x = [
                (np.linspace(0, n - 1, n, dtype=float) + n // 2) % n - n // 2
                for n in self.shape
            ]
        elif self.periodic is False:
            x = [np.arange(-n, n, dtype=float) for n in self.shape]
        else:
            raise ValueError(
                f"The periodic parameter must be True or False. It is {self.periodic}."
            )

        if self.topology == "rectangular":
            xy = np.meshgrid(*x, indexing="ij")
        elif self.topology == "hexagonal":
            if (n := len(self.shape)) > 2:
                raise ValueError(f"Hexagonal maps can only be 2D, not {n}D.")
            x[1] *= 0.5 * np.sqrt(3.0)
            xy = np.meshgrid(*x, indexing="ij")
            xy[0][:, 1::2] += 0.5
        else:
            raise ValueError(f"Unknown topology {self.topology}.")

        return np.sum(np.array(xy) ** 2.0, axis=0)

    def bmu(self, x):
        dist2 = np.sum([(self.weights[k] - x[k]) ** 2.0 for k in x], axis=0)

        return np.unravel_index(np.argmin(dist2), dist2.shape)

    def likelihood(self, x):
        fl = x.flux
        err = x.flux_err

        _ = list(x.flux.keys())[0]
        num = np.zeros_like(self.weights[_])
        denom = np.zeros_like(self.weights[_])

        for band, flux in fl.items():
            # Multiplications are faster than divisions, so we directly use the
            # inverse error
            inv_err2 = 1.0 / err[band] ** 2.0
            model = self.weights[band]
            num += model * (flux * inv_err2)
            denom += model ** 2.0 * inv_err2

        alpha = num / denom
        chi2 = np.sum([((self.weights[k] * alpha - fl[k]) / err[k]) ** 2.0 for k in fl], axis=0)

        return np.exp(-0.5 * chi2), alpha

    def h(self, x, sigma2):
        bmu = self.bmu(x)
        if self.periodic is True:
            dist2 = np.roll(self.dist2, bmu, axis=np.arange(len(bmu)))
        else:
            slices = tuple(
                slice(s - bmu[i], s * 2 - bmu[i])
                for i, s in enumerate(self.shape)
            )
            dist2 = self.dist2[(*slices, None)]

        return np.exp(dist2 * (-0.5 / sigma2)).reshape(self.shape)


class SOM:
    def __init__(self):
        pass

    def _init_func(self):
        self.alpha = functions(self.learn_mode, self.n, self.a1, self.a2)
        self.sigma2 = (
            functions(
                self.neighborhood_mode, self.n, np.max(self.map.shape) / 2, 1
            )
            ** 2.0
        )

    def _init_map(self, X, mode="data"):
        Xsize = list(X.values())[0].size

        if mode == "random":
            self.map.weights = {k: self.rng.random(self.shape) for k in X}
        elif mode == "data":
            indices = self.rng.integers(0, Xsize, self.shape)
            self.map.weights = {k: X[k][indices] for k in X}
        else:
            raise ValueError(
                f"Invalid init mode: {self.init_}."
            )

    def train(self, X):
        self._init_map(X)
        Xsize = list(X.values())[0].size

        w = self.map.weights
        alpha = self.alpha
        sigma2 = self.sigma2

        idx = self.rng.integers(0, Xsize, self.n)

        for it in range(self.n):
            # Pick a data point at random, compute the neighborhood function of
            # the weights map for the BMU corresponding to that random data
            # point and pdate the weights map.
            x = {k: X[k][idx[it]] for k in X}
            scale = alpha[it] * self.map.h(x, sigma2[it])

            for k in X:
                w[k] += scale * (x[k] - w[k])


class UnsupervisedSOM(SOM):
    def __init__(
        self,
        shape=(10, 10),
        *,
        metric="euclidean",
        topology="rectangular",
        periodic=False,
        init="random",
        n=1000,
        a1=0.5,
        a2=0.05,
        neighborhood_mode="linear",
        learn_mode="min",
    ):
        self.shape = shape
        self.metric = metric
        self.topology = topology
        self.periodic = periodic
        self.init = init
        self.n = n
        self.a1 = a1
        self.a2 = a2
        self.neighborhood_mode = neighborhood_mode
        self.learn_mode = learn_mode

        self.map = Map(shape, metric, topology, periodic)

        self._init_func()
        self.rng = np.random.default_rng(self.map.seed)


class SupervisedSOM(SOM):
    def __init__(
        self,
        usom,
        *,
        metric="euclidean",
        topology="rectangular",
        periodic=False,
        init="random",
        n=1000,
        a1=0.5,
        a2=0.05,
        neighborhood_mode="linear",
        learn_mode="min",
    ):
        self.usom = usom
        self.shape = self.usom.shape
        self.init = init
        self.n = n
        self.a1 = a1
        self.a2 = a2
        self.neighborhood_mode = neighborhood_mode
        self.learn_mode = learn_mode

        self.map = Map(
            self.usom.shape,
            self.usom.metric,
            self.usom.topology,
            self.usom.periodic
        )

        self._init_func()
        self.rng = np.random.default_rng(self.map.seed)

    def train(self, X, Y):
        self._init_map(Y)
        Ysize = list(Y.values())[0].size

        w = self.map.weights
        alpha = self.alpha
        sigma2 = self.sigma2

        for it in range(self.n):
            # Pick a data point at random, compute the neighborhood function of
            # the weights map for the BMU corresponding to that random data
            # point and pdate the weights map.
            i = self.rng.integers(0, Ysize)
            scale = alpha[it] * self.usom.map.h({k: X[k][i] for k in X}, sigma2[it])
            for k in Y:
                w[k] += scale * (Y[k][i] - w[k])

    def best(self, x):
        pos = self.usom.map.bmu(x.flux)

        if len(pos) == 0:
            results = {k: np.nan for k in self.map.weights}
            results |= {k: np.nan for k in self.usom.map.weights}
        else:
            results = {k: self.map.weights[k][pos] for k in self.map.weights}
            results |= {k: self.usom.map.weights[k][pos] for k in self.usom.map.weights}

        return results

    def bayes(self, x):
        results = {}
        errors = {}
        if len(x.flux) <= 1:
            results |= {k: np.nan for k in self.map.weights | self.usom.map.weights}
            errors |= {k: np.nan for k in self.map.weights | self.usom.map.weights}
        else:
            likelihood, alpha = self.usom.map.likelihood(x)
            likelihood *= 1. / np.sum(likelihood)
            likelihood = likelihood.ravel()
            for k in self.map.weights:
                results[k], errors[k] = weighted_param((self.map.weights[k] * alpha).ravel(), likelihood)

            for k in self.usom.map.weights:
                results[k], errors[k] = weighted_param((self.usom.map.weights[k] * alpha).ravel(), likelihood)


        results = {}
        errors = {}
        if len(x.flux) <= 1:
            results |= {k: np.nan for k in self.map.weights | self.usom.map.weights}
            errors |= {k: np.nan for k in self.map.weights | self.usom.map.weights}
        else:
            likelihood, alpha = self.usom.map.likelihood(x)
            likelihood *= 1. / np.sum(likelihood)
            likelihood = likelihood.ravel()
            for k in self.map.weights:
                results[k], errors[k] = weighted_param((self.map.weights[k] * alpha).ravel(), likelihood)

            for k in self.usom.map.weights:
                results[k], errors[k] = weighted_param((self.usom.map.weights[k] * alpha).ravel(), likelihood)




        return results, errors
