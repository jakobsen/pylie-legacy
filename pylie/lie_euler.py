from .so3 import _SO3Integrator
from .se3 import _SE3Integrator
import typing


class LieEuer_SO3(_SO3Integrator):
    def _single_step(self, y_n):
        return self._coadjoint_action(self._exp(self.h * self.f(y_n)), y_n)


class LieEuler_SE3(_SE3Integrator):
    def _single_step(self, y_n):
        return self._coadjoint_action(*self._exp(self.h * self.f(y_n)), y_n)


def lie_euler(f, y0, h, T, group: str) -> typing.Union[LieEuer_SO3, LieEuler_SE3]:
    group = group.upper()
    if group not in ["SO3", "SE3"]:
        raise ValueError(
            "The supplied group must be either SO3 or SE3 (case insensitive)"
        )
    if group == "SO3":
        return LieEuer_SO3(f, y0, h, T)
    if group == "SE3":
        return LieEuler_SE3(f, y0, h, T)