from .core import _LieIntegrator
import numpy as np


class _SO3Integrator(_LieIntegrator):
    def _hat(self, u):
        print(u)
        u1, u2, u3 = u
        return np.array([[0, -u3, u2], [u3, 0, -u1], [-u2, u1, 0]])

    def _exp(self, u):
        alpha = np.linalg.norm(u)
        u_hat = self._hat(u)
        return (
            np.eye(3)
            + (np.sin(alpha) / alpha) * u_hat
            + ((1 - np.cos(alpha)) / (alpha ** 2)) * u_hat @ u_hat
        )

    def _dexpinv(self, u):
        alpha = np.linalg.norm(u)
        u_hat = self._hat(u)
        return (
            np.eye(3)
            - 0.5 * u_hat
            + ((1 - 0.5 * alpha * self._cot(0.5 * alpha)) / (alpha ** 2))
            * u_hat
            @ u_hat
        )

    def _coadjoint_action(self, g, u):
        return g @ u