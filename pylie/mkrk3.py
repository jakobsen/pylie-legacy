from .se3 import _SE3Integrator
from .so3 import _SO3Integrator
import typing


class MKRK3_SO3(_SO3Integrator):
    def _single_step(self, y_n):
        k1 = self.f(y_n)

        u2 = (1 / 3) * self.h * k1
        k2 = self._coadjoint_action(
            self._dexpinv(u2), self.f(self._coadjoint_action(self._exp(u2), y_n))
        )

        u3 = (2 / 3) * self.h * k2
        k3 = self._coadjoint_action(
            self._dexpinv(u3), self.f(self._coadjoint_action(self._exp(u3), y_n))
        )

        v = (self.h / 4) * (k1 + 3 * k3)
        return self._coadjoint_action(self._exp(v), y_n)


class MKRK3_SE3(_SE3Integrator):
    def _single_step(self, y_n, h):
        k1 = self.f(y_n)

        u2 = 1 / 3 * h * k1
        k2 = self._dexpinv(u2, self.f(self._coadjoint_action(*self._exp(u2), y_n)))

        u3 = 2 / 3 * h * k2
        k3 = self._dexpinv(u3, self.f(self._coadjoint_action(*self._exp(u3), y_n)))

        v = (h / 4) * (k1 + 3 * k3)
        return self._coadjoint_action(*self._exp(v), y_n)


def mkrk3(f, y0, h, T, group: str) -> typing.Union[MKRK3_SO3, MKRK3_SE3]:
    group = group.upper()
    if group not in ["SO3", "SE3"]:
        raise ValueError(
            "The supplied group must be either SO3 or SE3 (case insensitive)"
        )
    if group == "SO3":
        return MKRK3_SO3(f, y0, h, T)
    if group == "SE3":
        return MKRK3_SE3(f, y0, h, T)