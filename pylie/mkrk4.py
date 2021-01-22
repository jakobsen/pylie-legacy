from .se3 import _SE3Integrator
from .so3 import _SO3Integrator
import typing


class MKRK4_SO3(_SO3Integrator):
    def _single_step(self, y_n):
        k1 = self.f(y_n)

        u2 = 0.5 * self.h * k1
        k2 = self._coadjoint_action(
            self._dexpinv(u2), self.f(self._coadjoint_action(self._exp(u2), y_n))
        )

        u3 = 0.5 * self.h * k2
        k3 = self._coadjoint_action(
            self._dexpinv(u3), self.f(self._coadjoint_action(self._exp(u3), y_n))
        )

        u4 = self.h * k3
        k4 = self._coadjoint_action(
            self._dexpinv(u4), self.f(self._coadjoint_action(self._exp(u4), y_n))
        )

        v = (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self._coadjoint_action(self._exp(v), y_n)


class MKRK4_SE3(_SE3Integrator):
    def _single_step(self, y_n):
        k1 = self.f(y_n)

        u2 = 0.5 * self.h * k1
        k2 = self._dexpinv(u2, self.f(self._coadjoint_action(*self._exp(u2), y_n)))

        u3 = 0.5 * self.h * k2
        k3 = self._dexpinv(u3, self.f(self._coadjoint_action(*self._exp(u3), y_n)))

        u4 = self.h * k3
        k4 = self._dexpinv(u4, self.f(self._coadjoint_action(*self._exp(u4), y_n)))

        v = (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self._coadjoint_action(*self._exp(v), y_n)


def mkrk4(f, y0, h, T, group: str) -> typing.Union[MKRK4_SO3, MKRK4_SE3]:
    group = group.upper()
    if group not in ["SO3", "SE3"]:
        raise ValueError(
            "The supplied group must be either SO3 or SE3 (case insensitive)"
        )
    if group == "SO3":
        return MKRK4_SO3(f, y0, h, T)
    if group == "SE3":
        return MKRK4_SE3(f, y0, h, T)