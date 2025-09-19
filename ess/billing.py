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

    def __init__(self, tariff_cfg: Dict, daily_fixed_cost_eur: float = 0.0):
        self.tariff_cfg = tariff_cfg
        self.daily_fixed_cost = daily_fixed_cost_eur

    def generate_ledger(self, flows_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Return per-interval ledger with cost components for both scenarios."""
        df = flows_df.join(prices_df[['price_omie_eur_kwh', 'tariff_energy_eur_kwh']], how='left')
        df['energy_without_kwh'] = df['house_consumption_kwh']
        df['energy_with_kwh'] = df['total_grid_import_kwh']
        k2 = self.tariff_cfg['indexed']['k2_eur_kwh']
        iec_tax = self.tariff_cfg.get('iec_tax_eur_kwh', 0.0)
        iec_vat_rate = self.tariff_cfg.get('iec_vat_rate', self.tariff_cfg.get('vat_rate', 0.0))

        for scenario in ['without', 'with']:
            energy_col = f'energy_{scenario}_kwh'
            omie_col = f'omie_{scenario}_eur'
            tariff_col = f'tariff_{scenario}_eur'
            k2_col = f'k2_{scenario}_eur'
            iec_col = f'iec_{scenario}_eur'
            iec_vat_col = f'iec_vat_{scenario}_eur'

            #grid energy price 
            df[omie_col] = df['price_omie_eur_kwh'] * df[energy_col]
            #TAR_energy price
            df[tariff_col] = df['tariff_energy_eur_kwh'] * df[energy_col]
            #K2 price
            df[k2_col] = k2 * df[energy_col]
            #IEC tax 
            df[iec_col] = iec_tax * df[energy_col]
            #IEC VAT
            df[iec_vat_col] = df[iec_col] * iec_vat_rate

        # === Dynamic Energy VAT per interval (resets each VAT cycle) ===
        # Config
        vat_cycle_days = int(self.tariff_cfg.get('vat_cycle_days', 30))
        reduced_block_kwh = float(self.tariff_cfg.get('reduced_vat_kwh_per_30_days', 200))
        reduced_vat_rate = float(self.tariff_cfg.get('reduced_vat_rate', 0.06))
        standard_vat_rate = float(self.tariff_cfg.get('vat_rate', 0.23))

        # Anchor date for VAT cycles (start-of-day). If not provided, use the ledger's first day
        if 'vat_cycle_anchor_date' in self.tariff_cfg:
            anchor = pd.Timestamp(self.tariff_cfg['vat_cycle_anchor_date']).normalize()
        else:
            anchor = df.index[0].normalize()

        # Compute a cycle id for each row
        df['_cycle_id'] = ((df.index.normalize() - anchor).days // vat_cycle_days).astype(int)

        for scenario in ['without', 'with']:
            energy_col = f'energy_{scenario}_kwh'
            base_col = f'energy_base_{scenario}_eur'
            # Base taxable amount for energy VAT is OMIE + K2 + tariff (per interval)
            df[base_col] = df[f'omie_{scenario}_eur'] + df[f'k2_{scenario}_eur'] + df[f'tariff_{scenario}_eur']
            # Output columns
            vat_col = f'energy_vat_{scenario}_eur'
            vat_red_col = f'energy_vat_reduced_{scenario}_eur'
            vat_std_col = f'energy_vat_standard_{scenario}_eur'
            eff_rate_col = f'effective_energy_vat_rate_{scenario}'
            df[vat_col] = 0.0
            df[vat_red_col] = 0.0
            df[vat_std_col] = 0.0
            df[eff_rate_col] = 0.0

            # Iterate cycle-by-cycle to apply the reduced block dynamically
            for cid, grp in df.groupby('_cycle_id', sort=True):
                idx = grp.index
                if len(idx) == 0:
                    continue
                # Pro-rate the reduced block when the ledger covers only part of a cycle
                days_covered = int(idx.normalize().nunique())
                cycle_threshold_kwh = reduced_block_kwh * (days_covered / vat_cycle_days)
                remaining_reduced = max(0.0, cycle_threshold_kwh)

                for ts in idx:
                    e = float(df.at[ts, energy_col])
                    base = float(df.at[ts, base_col])
                    if e <= 0.0 or base <= 0.0:
                        df.at[ts, eff_rate_col] = 0.0
                        continue

                    # Split this interval's energy between reduced and standard blocks
                    reduced_kwh = min(e, max(0.0, remaining_reduced))
                    standard_kwh = max(0.0, e - reduced_kwh)
                    #--somando os dois tem que dar e

                    # Allocate the base amount proportionally
                    base_per_kwh = base / e
                    reduced_base = base_per_kwh * reduced_kwh
                    standard_base = base_per_kwh * standard_kwh

                    # Compute VAT for each portion
                    vat_reduced = reduced_base * reduced_vat_rate
                    vat_standard = standard_base * standard_vat_rate
                    df.at[ts, vat_red_col] = vat_reduced
                    df.at[ts, vat_std_col] = vat_standard
                    df.at[ts, vat_col] = vat_reduced + vat_standard
                    df.at[ts, eff_rate_col] = (df.at[ts, vat_col] / base) if base > 0 else 0.0

                    # Update remaining reduced allowance in this cycle
                    remaining_reduced -= reduced_kwh

        # Clean up helper column
        df.drop(columns=['_cycle_id'], inplace=True)

        return df

    def _aggregate_components(self, ledger: pd.DataFrame, scenario: str) -> Dict[str, float]:
        comps = {
            'OMIE Market': ledger[f'omie_{scenario}_eur'].sum(),
            'Network Access (K2)': ledger[f'k2_{scenario}_eur'].sum(),
            'Time-of-Use Tariff': ledger[f'tariff_{scenario}_eur'].sum(),
            'IEC Tax': ledger[f'iec_{scenario}_eur'].sum(),
            'IEC VAT': ledger[f'iec_vat_{scenario}_eur'].sum(),
        }
        return comps

    def invoice_from_ledger(self, ledger: pd.DataFrame) -> Invoice:
        # Total days covered (inclusive) to apply fixed daily costs
        n_days = (ledger.index[-1].date() - ledger.index[0].date()).days + 1
        fixed_total = self.daily_fixed_cost * n_days

        # Aggregate base components (excluding Energy VAT; we handle it below)
        without_comps = self._aggregate_components(ledger, 'without')
        with_comps = self._aggregate_components(ledger, 'with')
      
        energy_vat_without = float(ledger['energy_vat_without_eur'].sum())
        energy_vat_with = float(ledger['energy_vat_with_eur'].sum())

        # Insert Energy VAT component
        without_comps['Energy VAT'] = energy_vat_without
        with_comps['Energy VAT'] = energy_vat_with

        # Add fixed costs
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
