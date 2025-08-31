"""
ess/battery.py - Physical battery model with round-trip efficiency
"""

import numpy as np
from typing import Tuple


class Battery:
    """Minimal battery model with fixed round-trip efficiency."""
    
    def __init__(self,
                 capacity_kwh: float = 10,
                 soc_init: float = 0.5,
                 max_charge_kw: float = 3,
                 max_discharge_kw: float = 3,
                 efficiency: float = 0.9,
                 soc_min: float = 0.2,
                 soc_max: float = 0.8):
        
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.efficiency = efficiency
        self.soc_min = soc_min
        self.soc_max = soc_max
        
        # Split round-trip efficiency equally between charge/discharge
        self.efficiency_charge = np.sqrt(efficiency)
        self.efficiency_discharge = np.sqrt(efficiency)
        
        # State
        self.soc = soc_init  # State of charge (0-1)
        self.soc_kwh = soc_init * capacity_kwh
        
        # Tracking for metrics
        self.total_charged_kwh = 0
        self.total_discharged_kwh = 0
        self.cycles = 0
        
    def get_available_capacity(self) -> Tuple[float, float]:
        """Returns (available_to_charge_kwh, available_to_discharge_kwh)"""
        available_to_charge = (self.soc_max * self.capacity_kwh) - self.soc_kwh
        available_to_discharge = self.soc_kwh - (self.soc_min * self.capacity_kwh)
        return max(0, available_to_charge), max(0, available_to_discharge)
    
    def get_max_power(self, duration_hours: float = 0.25) -> Tuple[float, float]:
        """
        Get maximum charge/discharge power considering SOC limits.
        Returns (max_charge_kw, max_discharge_kw) for the given duration.
        """
        available_to_charge, available_to_discharge = self.get_available_capacity()
        
        # Power limited by energy available
        max_charge_by_energy = available_to_charge / (duration_hours * self.efficiency_charge)
        max_discharge_by_energy = available_to_discharge * self.efficiency_discharge / duration_hours
        
        # Apply power limits
        max_charge = min(self.max_charge_kw, max_charge_by_energy)
        max_discharge = min(self.max_discharge_kw, max_discharge_by_energy)
        
        return max_charge, max_discharge
    
    def charge(self, power_kw: float, duration_hours: float = 0.25) -> float:
        """
        Charge the battery. Returns actual energy charged from grid (kWh).
        duration_hours: default 0.25 for 15-minute intervals
        """
        if power_kw <= 0:
            return 0
            
        # Get maximum allowed charge power
        max_charge, _ = self.get_max_power(duration_hours)
        power_kw = min(power_kw, max_charge)
        
        # Energy from grid
        energy_from_grid = power_kw * duration_hours
        
        # Energy stored in battery (after losses)
        energy_to_battery = energy_from_grid * self.efficiency_charge
        
        # Update state
        self.soc_kwh += energy_to_battery
        self.soc = self.soc_kwh / self.capacity_kwh
        
        # Track metrics
        self.total_charged_kwh += energy_from_grid
        self.cycles += energy_to_battery / (2 * self.capacity_kwh)  # Partial cycle
        
        return energy_from_grid
    
    def discharge(self, power_kw: float, duration_hours: float = 0.25) -> float:
        """
        Discharge the battery. Returns actual energy delivered to load (kWh).
        duration_hours: default 0.25 for 15-minute intervals
        """
        if power_kw <= 0:
            return 0
            
        # Get maximum allowed discharge power
        _, max_discharge = self.get_max_power(duration_hours)
        power_kw = min(power_kw, max_discharge)
        
        # Energy to deliver
        energy_to_load = power_kw * duration_hours
        
        # Energy from battery (before losses)
        energy_from_battery = energy_to_load / self.efficiency_discharge
        
        # Update state
        self.soc_kwh -= energy_from_battery
        self.soc = self.soc_kwh / self.capacity_kwh
        
        # Track metrics
        self.total_discharged_kwh += energy_to_load
        self.cycles += energy_from_battery / (2 * self.capacity_kwh)  # Partial cycle
        
        return energy_to_load
    
    def reset(self, soc_init: float = None):
        """Reset battery to initial state"""
        if soc_init is None:
            soc_init = 0.5
        self.soc = soc_init
        self.soc_kwh = soc_init * self.capacity_kwh
        self.total_charged_kwh = 0
        self.total_discharged_kwh = 0
        self.cycles = 0
    
    def get_state(self) -> dict:
        """Get current battery state as dict"""
        return {
            'soc': self.soc,
            'soc_kwh': self.soc_kwh,
            'soc_pct': self.soc * 100,
            'available_to_charge_kwh': self.get_available_capacity()[0],
            'available_to_discharge_kwh': self.get_available_capacity()[1],
            'total_charged_kwh': self.total_charged_kwh,
            'total_discharged_kwh': self.total_discharged_kwh,
            'cycles': self.cycles
        }