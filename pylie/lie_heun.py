from .so3 import _SO3Integrator
from .se3 import _SE3Integrator
import typing


class LieHeun_SO3(_SO3Integrator):
    def _single_step(self, y_n):
        k1 = self.f(y_n)

        u2 = self.h * k1
        k2 = self.f(self._coadjoint_action(self._exp(u2), y_n))

        v = 0.5 * self.h * (k1 + k2)
        return self._coadjoint_action(self._exp(v), y_n)


class LieHeun_SE3(_SE3Integrator):
    def _single_step(self, y_n):
        k1 = self.f(y_n)

        u2 = self.h * k1
        k2 = self.f(self._coadjoint_action(*self._exp(u2), y_n))

        v = 0.5 * self.h * (k1 + k2)
        return self._coadjoint_action(*self._exp(v), y_n)


def lie_heun(f, y0, h, T, group: str) -> typing.Union[LieHeun_SO3, LieHeun_SE3]:
    group = group.upper()
    if group not in ["SO3", "SE3"]:
        raise ValueError(
            "The supplied group must be either SO3 or SE3 (case insensitive)"
        )
    if group == "SO3":
        return LieHeun_SO3(f, y0, h, T)
    if group == "SE3":
        return LieHeun_SE3(f, y0, h, T)