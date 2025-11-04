"""
Battery module - handles battery storage with degradation modeling
"""
import numpy as np


class Battery:
    """Battery energy storage system with degradation cost"""

    def __init__(self,
                 capacity_kwh: float,
                 max_power_kw: float,
                 efficiency_charge: float = 0.95,
                 efficiency_discharge: float = 0.95,
                 initial_soc: float = 0.5,
                 min_soc: float = 0.1,
                 max_soc: float = 0.9,
                 degradation_cost_per_kwh: float = 0.01,
                 max_charge_current_kw: float = None,
                 max_discharge_current_kw: float = None):
        """
        Args:
            capacity_kwh: Battery capacity in kWh
            max_power_kw: Maximum charge/discharge power in kW (used if specific limits not set)
            efficiency_charge: Charging efficiency (0-1)
            efficiency_discharge: Discharging efficiency (0-1)
            initial_soc: Initial state of charge (0-1)
            min_soc: Minimum allowed state of charge (0-1)
            max_soc: Maximum allowed state of charge (0-1)
            degradation_cost_per_kwh: Cost per kWh throughput (€/kWh)
            max_charge_current_kw: Maximum charging power in kW (None = use max_power_kw)
            max_discharge_current_kw: Maximum discharging power in kW (None = use max_power_kw)
        """
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.max_charge_power_kw = max_charge_current_kw if max_charge_current_kw is not None else max_power_kw
        self.max_discharge_power_kw = max_discharge_current_kw if max_discharge_current_kw is not None else max_power_kw
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.degradation_cost_per_kwh = degradation_cost_per_kwh

        # State variables
        self.soc = initial_soc  # State of charge (0-1)
        self.energy_kwh = self.soc * self.capacity_kwh

        # Tracking
        self.total_charge_kwh = 0.0
        self.total_discharge_kwh = 0.0
        self.total_degradation_cost = 0.0

    def get_energy_kwh(self) -> float:
        """Get current energy stored in battery"""
        return self.energy_kwh

    def get_soc(self) -> float:
        """Get current state of charge (0-1)"""
        return self.soc

    def get_available_charge_power(self, dt_hours: float) -> float:
        """
        Get maximum power that can be charged in next time step

        Args:
            dt_hours: Time step duration in hours

        Returns:
            Maximum charge power in kW (AC side)
        """
        # Maximum energy that can be stored (DC side)
        max_energy_increase = (self.max_soc * self.capacity_kwh) - self.energy_kwh

        # Maximum AC power needed to store this energy considering efficiency
        # AC_energy = DC_energy / efficiency
        # AC_power = AC_energy / dt = DC_energy / (efficiency * dt)
        max_power_energy = max_energy_increase / (self.efficiency_charge * dt_hours)

        # Limited by max charge power rating
        return min(self.max_charge_power_kw, max_power_energy)

    def get_available_discharge_power(self, dt_hours: float) -> float:
        """
        Get maximum power that can be discharged in next time step

        Args:
            dt_hours: Time step duration in hours

        Returns:
            Maximum discharge power in kW (AC side)
        """
        # Maximum energy that can be extracted (DC side)
        max_energy_decrease = self.energy_kwh - (self.min_soc * self.capacity_kwh)

        # Maximum AC power output from this DC energy considering efficiency
        # AC_energy = DC_energy * efficiency
        # AC_power = AC_energy / dt = DC_energy * efficiency / dt
        max_power_energy = max_energy_decrease * self.efficiency_discharge / dt_hours

        # Limited by max discharge power rating
        return min(self.max_discharge_power_kw, max_power_energy)

    def charge(self, power_kw: float, dt_hours: float) -> float:
        """
        Charge battery with given power

        Args:
            power_kw: Charging power in kW (positive)
            dt_hours: Time step duration in hours

        Returns:
            Actual power charged (may be limited by constraints)
        """
        # Limit to available charge power
        max_power = self.get_available_charge_power(dt_hours)
        actual_power = min(power_kw, max_power)

        # Update energy (accounting for efficiency)
        energy_increase = actual_power * dt_hours * self.efficiency_charge
        self.energy_kwh += energy_increase
        self.soc = self.energy_kwh / self.capacity_kwh

        # Track for degradation
        self.total_charge_kwh += energy_increase
        throughput_cost = energy_increase * self.degradation_cost_per_kwh
        self.total_degradation_cost += throughput_cost

        return actual_power

    def discharge(self, power_kw: float, dt_hours: float) -> float:
        """
        Discharge battery with given power

        Args:
            power_kw: Discharging power in kW (positive)
            dt_hours: Time step duration in hours

        Returns:
            Actual power discharged (may be limited by constraints)
        """
        # Limit to available discharge power
        max_power = self.get_available_discharge_power(dt_hours)
        actual_power = min(power_kw, max_power)

        # Update energy (accounting for efficiency)
        energy_decrease = actual_power * dt_hours / self.efficiency_discharge
        self.energy_kwh -= energy_decrease
        self.soc = self.energy_kwh / self.capacity_kwh

        # Track for degradation
        self.total_discharge_kwh += energy_decrease
        throughput_cost = energy_decrease * self.degradation_cost_per_kwh
        self.total_degradation_cost += throughput_cost

        return actual_power

    def step(self, power_kw: float, dt_hours: float) -> dict:
        """
        Execute one time step with given power

        Args:
            power_kw: Power in kW (positive = charge, negative = discharge)
            dt_hours: Time step duration in hours

        Returns:
            Dictionary with step results
        """
        initial_soc = self.soc
        initial_energy = self.energy_kwh

        if power_kw > 0:
            # Charging
            actual_power = self.charge(power_kw, dt_hours)
        elif power_kw < 0:
            # Discharging
            actual_power = -self.discharge(-power_kw, dt_hours)
        else:
            actual_power = 0.0

        return {
            'power_kw': actual_power,
            'soc_initial': initial_soc,
            'soc_final': self.soc,
            'energy_initial_kwh': initial_energy,
            'energy_final_kwh': self.energy_kwh,
            'energy_change_kwh': self.energy_kwh - initial_energy
        }

    def reset(self, soc: float = 0.5):
        """Reset battery to initial state"""
        self.soc = soc
        self.energy_kwh = self.soc * self.capacity_kwh
        self.total_charge_kwh = 0.0
        self.total_discharge_kwh = 0.0
        self.total_degradation_cost = 0.0

    def get_stats(self) -> dict:
        """Get battery statistics"""
        return {
            'soc': self.soc,
            'energy_kwh': self.energy_kwh,
            'total_charge_kwh': self.total_charge_kwh,
            'total_discharge_kwh': self.total_discharge_kwh,
            'total_degradation_cost': self.total_degradation_cost,
            'efficiency_roundtrip': self.efficiency_charge * self.efficiency_discharge
        }
