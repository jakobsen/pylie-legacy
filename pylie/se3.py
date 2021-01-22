from .core import _LieIntegrator
import numpy as np


class _SE3Integrator(_LieIntegrator):
    def _hat(self, u):
        u1, u2, u3 = u
        return np.array([[0, -u3, u2], [u3, 0, -u1], [-u2, u1, 0]])

    def _so3_exp(self, u):
        alpha = np.linalg.norm(u)
        u_hat = self._hat(u)
        return (
            np.eye(3)
            + (np.sin(alpha) / alpha) * u_hat
            + ((1 - np.cos(alpha)) / (alpha ** 2)) * u_hat @ u_hat
        )

    def _exp(self, u):
        u, v = np.split(u, 2)
        u_exp = self._so3_exp(u)
        u_hat = self._hat(u)
        alpha = np.linalg.norm(u)
        v_exp = (
            np.eye(3)
            + (1 - np.cos(alpha)) / (alpha ** 2) * u_hat
            + (alpha - np.sin(alpha)) / (alpha ** 3) * u_hat @ u_hat
        ) @ v
        return (u_exp, v_exp)

    def _dexpinv_helper_1(self, z):
        return (1 - 0.5 * z * self._cot(0.5 * z)) / (z ** 2)

    def _dexpinv_helper_2(self, z, rho):
        return (
            0.25
            * rho
            * ((z * self._csc(0.5 * z)) ** 2 + 2 * z * self._cot(0.5 * z) - 8)
            / (z ** 4)
        )

    def _dexpinv(self, x, y):
        """Returns the result of dexp^(-1)_(x) (y).
        Both x and y are elements of se(3), represented as vectors of length 6"""
        A, a = np.split(x, 2)
        B, b = np.split(y, 2)
        alpha = np.linalg.norm(A, ord=2)
        rho = np.inner(A, a)
        c1 = (
            B
            - 0.5 * np.cross(A, B)
            + self._dexpinv_helper_1(alpha) * np.cross(A, np.cross(A, B))
        )
        c2 = (
            b
            - 0.5 * (np.cross(a, B) + np.cross(A, b))
            + self._dexpinv_helper_2(alpha, rho) * np.cross(A, np.cross(A, B))
            + self._dexpinv_helper_1(alpha)
            * (
                np.cross(a, np.cross(A, B))
                + np.cross(A, np.cross(a, B))
                + np.cross(A, np.cross(A, b))
            )
        )
        return np.hstack((c1, c2))

    def _coadjoint_action(self, g, u, y):
        mu, beta = np.split(y, 2)
        z1 = g.T @ (mu - np.cross(u, beta))
        z2 = g.T @ beta
        return np.hstack((z1, z2))