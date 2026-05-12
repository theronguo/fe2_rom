from abc import ABC, abstractmethod
from typing import Callable, Any
import ufl


class MaterialModel(ABC):
    @abstractmethod
    def strain_energy(self, F) -> Any:
        """Return UFL scalar strain energy density W(F). F must be a ufl.variable."""
        ...

    def first_pk_stress(self, F) -> Any:
        """First Piola-Kirchhoff stress P = dW/dF. Requires F to be a ufl.variable."""
        return ufl.diff(self.strain_energy(F), F)
    
    def tangent_moduli(self, F) -> Any:
        """Tangent moduli A = dP/dF. Requires F to be a ufl.variable."""
        P = self.first_pk_stress(F)
        return ufl.diff(P, F)


class NeoHookean(MaterialModel):
    def __init__(self, mu: float, lmbda: float):
        self.mu = mu
        self.lmbda = lmbda

    def strain_energy(self, F):
        mu, lmbda = self.mu, self.lmbda
        J = ufl.det(F)
        C = F.T * F
        return (mu / 2) * (ufl.tr(C) - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J)) ** 2


class LambdaMaterial(MaterialModel):
    """Adapter so a plain callable W(F) can be used as a MaterialModel."""

    def __init__(self, fn: Callable):
        self._fn = fn

    def strain_energy(self, F):
        return self._fn(F)
