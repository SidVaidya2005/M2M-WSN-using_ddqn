"""
Battery model for wireless sensor nodes.

Simulates degradation based on depth of discharge (DoD) cycles and calendar fade.
"""

import numpy as np
from typing import Tuple


class BatteryModel:
    """
    Battery model with State of Charge (SoC) and State of Health (SoH).

    SoC: available energy (0..E_max).
    SoH: health percentage (0..1). Degrades with deep discharge cycles and calendar aging.
    charging: True when the node is connected to an external charger.

    Degradation model:
    - Cycle degradation: proportional to (Depth of Discharge)^alpha * k_cycle
    - Calendar fade: small constant decay per timestep (discharge or charge)
    """

    def __init__(
        self,
        E_max: float = 100.0,
        soh_init: float = 1.0,
        k_cycle: float = 1e-4,
        alpha: float = 1.2,
        calendar_decay: float = 1e-6,
    ):
        """Initialize battery.

        Args:
            E_max: Maximum energy capacity
            soh_init: Initial state of health (0..1)
            k_cycle: Cycle degradation rate constant
            alpha: DoD exponent for cycle degradation model
            calendar_decay: Calendar fade rate per timestep
        """
        self.E_max = E_max
        self.soc = E_max
        self.soh = soh_init
        self.k_cycle = k_cycle
        self.alpha = alpha
        self.calendar_decay = calendar_decay
        self.prev_soc = self.soc
        self.charging: bool = False

    def discharge(self, energy_draw: float) -> None:
        """Discharge battery and apply degradation.

        Args:
            energy_draw: Energy to draw (>= 0)
        """
        self.prev_soc = self.soc
        self.soc = max(0.0, self.soc - energy_draw)

        dod = abs(self.prev_soc - self.soc) / self.E_max
        if dod > 0:
            self.soh -= self.k_cycle * (dod ** self.alpha)

        self.soh -= self.calendar_decay
        self.soh = max(0.0, min(1.0, self.soh))

    def charge(self, rate: float) -> None:
        """Recover SoC while on charger and apply calendar aging.

        Args:
            rate: Fraction of E_max recovered per step (e.g. 0.05)
        """
        self.prev_soc = self.soc
        self.soc = min(self.E_max, self.soc + rate * self.E_max)
        # Calendar decay still applies while charging (heat / electrolyte aging)
        self.soh -= self.calendar_decay
        self.soh = max(0.0, min(1.0, self.soh))

    @property
    def is_charging(self) -> bool:
        """True if this node is currently connected to a charger."""
        return self.charging

    def needs_charge(self, threshold: float) -> bool:
        """True when normalised SoC falls below *threshold*.

        Args:
            threshold: Fraction of E_max (e.g. 0.2 for 20 %)
        """
        return (self.soc / self.E_max) < threshold

    def is_dead(
        self, soc_threshold: float = 0.01, soh_threshold: float = 0.05
    ) -> bool:
        """Check if battery is considered dead.

        Args:
            soc_threshold: Dead if SoC <= this fraction of E_max
            soh_threshold: Dead if SoH <= this value

        Returns:
            True if battery is dead, False otherwise
        """
        return (self.soc <= soc_threshold) or (self.soh <= soh_threshold)

    def reset_to_health(self, soh: float = 1.0) -> None:
        """Reset battery to full charge and specified health.

        Args:
            soh: New state of health value (0..1)
        """
        self.soc = self.E_max
        self.soh = soh
        self.prev_soc = self.soc
        self.charging = False

    def __repr__(self) -> str:
        return (
            f"Battery(SoC={self.soc:.1f}/{self.E_max}, "
            f"SoH={self.soh:.3f}, charging={self.charging})"
        )
