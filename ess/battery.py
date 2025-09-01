"""
ess/battery.py - Physical battery model with round-trip efficiency (CORRECTED VERSION)
"""

import numpy as np
from typing import Tuple


class Battery:
    """
    Improved battery model with proper efficiency handling and enhanced constraints.
    """
    
    def __init__(self,
                 capacity_kwh: float = 10,
                 soc_init: float = 0.5,
                 max_charge_kw: float = 3,
                 max_discharge_kw: float = 3,
                 efficiency: float = 0.9,
                 soc_min: float = 0.2,
                 soc_max: float = 0.8,
                 degradation_cost_eur_kwh: float = 0.0):
        """
        Parameters
        ----------
        capacity_kwh : float
            Battery capacity in kWh
        soc_init : float
            Initial state of charge (0-1)
        max_charge_kw : float
            Maximum charging power
        max_discharge_kw : float
            Maximum discharging power
        efficiency : float
            Round-trip efficiency (0-1)
        soc_min : float
            Minimum allowed SOC (0-1)
        soc_max : float
            Maximum allowed SOC (0-1)
        degradation_cost_eur_kwh : float
            Cost per kWh cycled for degradation tracking
        """
        
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.degradation_cost_eur_kwh = degradation_cost_eur_kwh
        
        # Efficiency handling - apply round-trip efficiency correctly
        # Option 1: Split efficiency equally between charge and discharge
        self.efficiency_charge = np.sqrt(efficiency)  # √0.9 ≈ 0.949
        self.efficiency_discharge = np.sqrt(efficiency)  # √0.9 ≈ 0.949
        
        # Option 2: Alternative - apply full efficiency to one direction
        # self.efficiency_charge = efficiency  # 0.9
        # self.efficiency_discharge = 1.0  # 1.0
        
        # Or apply to discharge only:
        # self.efficiency_charge = 1.0
        # self.efficiency_discharge = efficiency
        
        # Validate inputs
        if not 0 < efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        if not 0 <= soc_min < soc_max <= 1:
            raise ValueError("Invalid SOC limits")
        if soc_init < soc_min or soc_init > soc_max:
            raise ValueError("Initial SOC outside allowed range")
        
        # State variables
        self.soc = soc_init  # State of charge (0-1)
        self.soc_kwh = soc_init * capacity_kwh  # Energy content in kWh
        
        # Tracking variables
        self.total_charged_kwh = 0  # Total energy charged from grid
        self.total_discharged_kwh = 0  # Total energy discharged to load
        self.cycles = 0  # Equivalent full cycles
        self.total_degradation_cost = 0  # Cumulative degradation cost
        
        # Operational constraints
        self.min_energy_kwh = soc_min * capacity_kwh
        self.max_energy_kwh = soc_max * capacity_kwh
        
    def get_available_capacity(self) -> Tuple[float, float]:
        """
        Returns available capacity for charging and discharging.
        
        Returns
        -------
        Tuple[float, float]
            (available_to_charge_kwh, available_to_discharge_kwh)
        """
        available_to_charge = self.max_energy_kwh - self.soc_kwh
        available_to_discharge = self.soc_kwh - self.min_energy_kwh
        return max(0, available_to_charge), max(0, available_to_discharge)
    
    def get_max_power(self, duration_hours: float = 0.25) -> Tuple[float, float]:
        """
        Get maximum charge/discharge power considering SOC and energy limits.
        
        Parameters
        ----------
        duration_hours : float
            Duration of the power delivery (default: 0.25h = 15min)
        
        Returns
        -------
        Tuple[float, float]
            (max_charge_kw, max_discharge_kw) for the given duration
        """
        available_to_charge, available_to_discharge = self.get_available_capacity()
        
        # Power limited by available energy and efficiency
        max_charge_by_energy = available_to_charge / (duration_hours * self.efficiency_charge)
        max_discharge_by_energy = (available_to_discharge * self.efficiency_discharge) / duration_hours
        
        # Apply power rating limits
        max_charge = min(self.max_charge_kw, max_charge_by_energy)
        max_discharge = min(self.max_discharge_kw, max_discharge_by_energy)
        
        return max(0, max_charge), max(0, max_discharge)
    
    def charge(self, power_kw: float, duration_hours: float = 0.25) -> float:
        """
        Charge the battery.
        
        Parameters
        ----------
        power_kw : float
            Charging power from grid (kW)
        duration_hours : float
            Duration of charging (hours)
        
        Returns
        -------
        float
            Actual energy drawn from grid (kWh)
        """
        if power_kw <= 0:
            return 0
        
        # Apply constraints
        max_charge_kw, _ = self.get_max_power(duration_hours)
        actual_power = min(power_kw, max_charge_kw)
        
        if actual_power <= 0:
            return 0
        
        # Energy from grid
        energy_from_grid = actual_power * duration_hours
        
        # Energy stored in battery (after charging losses)
        energy_to_battery = energy_from_grid * self.efficiency_charge
        
        # Update battery state
        self.soc_kwh += energy_to_battery
        self.soc = self.soc_kwh / self.capacity_kwh
        
        # Update tracking metrics
        self.total_charged_kwh += energy_from_grid
        cycle_increment = energy_to_battery / (2 * self.capacity_kwh)
        self.cycles += cycle_increment
        self.total_degradation_cost += energy_to_battery * self.degradation_cost_eur_kwh
        
        # Ensure we stay within bounds (numerical precision issues)
        self.soc_kwh = min(self.soc_kwh, self.max_energy_kwh)
        self.soc = self.soc_kwh / self.capacity_kwh
        
        return energy_from_grid
    
    def discharge(self, power_kw: float, duration_hours: float = 0.25) -> float:
        """
        Discharge the battery.
        
        Parameters
        ----------
        power_kw : float
            Discharging power to load (kW)
        duration_hours : float
            Duration of discharging (hours)
        
        Returns
        -------
        float
            Actual energy delivered to load (kWh)
        """
        if power_kw <= 0:
            return 0
        
        # Apply constraints
        _, max_discharge_kw = self.get_max_power(duration_hours)
        actual_power = min(power_kw, max_discharge_kw)
        
        if actual_power <= 0:
            return 0
        
        # Energy to deliver to load
        energy_to_load = actual_power * duration_hours
        
        # Energy required from battery (before discharge losses)
        energy_from_battery = energy_to_load / self.efficiency_discharge
        
        # Update battery state
        self.soc_kwh -= energy_from_battery
        self.soc = self.soc_kwh / self.capacity_kwh
        
        # Update tracking metrics
        self.total_discharged_kwh += energy_to_load
        cycle_increment = energy_from_battery / (2 * self.capacity_kwh)
        self.cycles += cycle_increment
        self.total_degradation_cost += energy_from_battery * self.degradation_cost_eur_kwh
        
        # Ensure we stay within bounds
        self.soc_kwh = max(self.soc_kwh, self.min_energy_kwh)
        self.soc = self.soc_kwh / self.capacity_kwh
        
        return energy_to_load
    
    def can_charge(self, energy_kwh: float) -> bool:
        """Check if battery can accept the specified energy."""
        available_capacity, _ = self.get_available_capacity()
        return energy_kwh <= available_capacity * self.efficiency_charge
    
    def can_discharge(self, energy_kwh: float) -> bool:
        """Check if battery can deliver the specified energy."""
        _, available_discharge = self.get_available_capacity()
        return energy_kwh <= available_discharge * self.efficiency_discharge
    
    def reset(self, soc_init: float = None):
        """Reset battery to initial or specified state."""
        if soc_init is None:
            soc_init = 0.5
        
        if soc_init < self.soc_min or soc_init > self.soc_max:
            raise ValueError(f"Reset SOC {soc_init} outside allowed range [{self.soc_min}, {self.soc_max}]")
        
        self.soc = soc_init
        self.soc_kwh = soc_init * self.capacity_kwh
        self.total_charged_kwh = 0
        self.total_discharged_kwh = 0
        self.cycles = 0
        self.total_degradation_cost = 0
    
    def get_state(self) -> dict:
        """Get comprehensive battery state information."""
        available_to_charge, available_to_discharge = self.get_available_capacity()
        max_charge_15min, max_discharge_15min = self.get_max_power(0.25)
        
        return {
            'soc': self.soc,
            'soc_pct': self.soc * 100,
            'soc_kwh': self.soc_kwh,
            'capacity_kwh': self.capacity_kwh,
            'available_to_charge_kwh': available_to_charge,
            'available_to_discharge_kwh': available_to_discharge,
            'max_charge_power_kw': max_charge_15min,
            'max_discharge_power_kw': max_discharge_15min,
            'total_charged_kwh': self.total_charged_kwh,
            'total_discharged_kwh': self.total_discharged_kwh,
            'cycles': self.cycles,
            'efficiency_charge': self.efficiency_charge,
            'efficiency_discharge': self.efficiency_discharge,
            'round_trip_efficiency': self.efficiency_charge * self.efficiency_discharge,
            'degradation_cost_eur': self.total_degradation_cost,
            'energy_throughput_kwh': self.total_charged_kwh + self.total_discharged_kwh
        }
    
    def get_performance_metrics(self) -> dict:
        """Calculate performance metrics."""
        if self.total_charged_kwh > 0:
            actual_efficiency = self.total_discharged_kwh / self.total_charged_kwh
        else:
            actual_efficiency = 0
            
        return {
            'theoretical_round_trip_efficiency': self.efficiency_charge * self.efficiency_discharge,
            'actual_efficiency': actual_efficiency,
            'capacity_utilization': (self.soc_max - self.soc_min),
            'power_to_energy_ratio_charge': self.max_charge_kw / self.capacity_kwh,
            'power_to_energy_ratio_discharge': self.max_discharge_kw / self.capacity_kwh,
            'equivalent_full_cycles': self.cycles,
            'average_cycle_depth': self.cycles / max(1, (self.total_charged_kwh / self.capacity_kwh)),
        }
    
    def __str__(self) -> str:
        """String representation of battery state."""
        state = self.get_state()
        return (f"Battery({self.capacity_kwh}kWh): "
                f"SOC={state['soc_pct']:.1f}%, "
                f"Available: +{state['available_to_charge_kwh']:.1f}kWh/-{state['available_to_discharge_kwh']:.1f}kWh, "
                f"Cycles={self.cycles:.2f}")
    
    def __repr__(self) -> str:
        return self.__str__()