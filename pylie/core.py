import numpy as np


class _LieIntegrator:
    """Base class on which all other integrators are built.
    The only methods implemented are __init__, as well as naive implementations
    of cot and csc defined by simply inverting the corresonding numpy functions.
    """

    def __init__(self, f, y0, h, T, verbose=True, solve_immediately=True):
        if not isinstance(y0, np.ndarray):
            raise TypeError("y0 must be a one-dimensional numpy array")

        if y0.ndim != 1:
            raise ValueError("y0 must be a one-dimensional numpy array")
        self.f = f
        self.T = T
        self.steps = int(T / h)
        self.t, self.h = np.linspace(0, T, self.steps, endpoint=True, retstep=True)

        # Create an array to hold the solution and initialize it with the given initial
        # value
        self.y = np.zeros((y0.size, self.steps))
        self.y[:, 0] = y0

        if solve_immediately:
            self.solve()
            if verbose:
                print("Problem was solved successfully")

    def solve(self):
        for i in range(1, self.steps):
            self.y[:, i] = self._single_step(self.y[:, i - 1])

    def _single_step(self, y_n):
        raise NotImplementedError

    def _hat(self, u):
        raise NotImplementedError

    def _exp(self, u):
        raise NotImplementedError

    def _cot(self, x):
        return 1 / np.tan(x)

    def _csc(self, x):
        return 1 / np.sin(x)

    def _dexpinv(self, x, y):
        raise NotImplementedError

    def _coadjoint_action(self, x, y):
        raise NotImplementedError