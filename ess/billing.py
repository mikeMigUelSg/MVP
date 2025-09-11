"""Billing engine to compute energy costs and metrics from physical flows.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
import pandas as pd


@dataclass
class Invoice:
    """Simple invoice representation."""
    without_battery: Dict[str, float]
    with_battery: Dict[str, float]
    total_without_battery: float
    total_with_battery: float
    savings: float


class BillingEngine:
    """Compute energy costs from physical flows using tariff configuration."""

    def __init__(self, tariff_cfg: Dict, contracted_power_kva: float, daily_fixed_cost_eur: float = 0.0):
        self.tariff_cfg = tariff_cfg
        self.contracted_power_kva = contracted_power_kva
        self.daily_fixed_cost = daily_fixed_cost_eur

        # Determine VAT rate for energy based on contracted power
        if contracted_power_kva <= tariff_cfg.get('reduced_vat_power_threshold_kva', 0):
            self.energy_vat_rate = tariff_cfg.get('reduced_vat_rate', 0.0)
        else:
            self.energy_vat_rate = tariff_cfg.get('vat_rate', 0.0)

    def generate_ledger(self, flows_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Return per-interval ledger with cost components for both scenarios."""
        df = flows_df.join(prices_df[['price_omie_eur_kwh', 'tariff_energy_eur_kwh']], how='left')
        df['energy_without_kwh'] = df['house_consumption_kwh']
        df['energy_with_kwh'] = df['total_grid_import_kwh']
        k2 = self.tariff_cfg['indexed']['k2_eur_kwh']
        iec_tax = self.tariff_cfg.get('iec_tax_eur_kwh', 0.0)
        iec_vat = self.tariff_cfg.get('iec_vat_rate', self.energy_vat_rate)

        for scenario in ['without', 'with']:
            energy_col = f'energy_{scenario}_kwh'
            omie_col = f'omie_{scenario}_eur'
            tariff_col = f'tariff_{scenario}_eur'
            k2_col = f'k2_{scenario}_eur'
            vat_col = f'energy_vat_{scenario}_eur'
            iec_col = f'iec_{scenario}_eur'
            iec_vat_col = f'iec_vat_{scenario}_eur'

            df[omie_col] = df['price_omie_eur_kwh'] * df[energy_col]
            df[tariff_col] = df['tariff_energy_eur_kwh'] * df[energy_col]
            df[k2_col] = k2 * df[energy_col]
            energy_subtotal = df[omie_col] + df[tariff_col] + df[k2_col]
            df[vat_col] = energy_subtotal * self.energy_vat_rate
            df[iec_col] = iec_tax * df[energy_col]
            df[iec_vat_col] = df[iec_col] * iec_vat

        return df

    def _aggregate_components(self, ledger: pd.DataFrame, scenario: str) -> Dict[str, float]:
        comps = {
            'OMIE Market': ledger[f'omie_{scenario}_eur'].sum(),
            'Network Access (K2)': ledger[f'k2_{scenario}_eur'].sum(),
            'Time-of-Use Tariff': ledger[f'tariff_{scenario}_eur'].sum(),
            'Energy VAT': ledger[f'energy_vat_{scenario}_eur'].sum(),
            'IEC Tax': ledger[f'iec_{scenario}_eur'].sum(),
            'IEC VAT': ledger[f'iec_vat_{scenario}_eur'].sum(),
        }
        return comps

    def invoice_from_ledger(self, ledger: pd.DataFrame) -> Invoice:
        n_days = (ledger.index[-1].date() - ledger.index[0].date()).days + 1
        fixed_total = self.daily_fixed_cost * n_days
        without_comps = self._aggregate_components(ledger, 'without')
        with_comps = self._aggregate_components(ledger, 'with')
        without_comps['Fixed Costs'] = fixed_total
        with_comps['Fixed Costs'] = fixed_total
        total_without = sum(without_comps.values())
        total_with = sum(with_comps.values())
        savings = total_without - total_with
        return Invoice(without_comps, with_comps, total_without, total_with, savings)

    def metrics_from_ledger(self, ledger: pd.DataFrame) -> Dict[str, float]:
        invoice = self.invoice_from_ledger(ledger)
        savings_pct = (invoice.savings / invoice.total_without_battery * 100) if invoice.total_without_battery else 0.0
        return {
            'cost_without_battery_eur': invoice.total_without_battery,
            'cost_with_battery_eur': invoice.total_with_battery,
            'savings_eur': invoice.savings,
            'savings_pct': savings_pct,
        }
