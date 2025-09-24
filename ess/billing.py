"""
ess/billing.py - REFACTORED VERSION with modular invoice components
Invoice is composed of 3 main parts:
1. Power term (fixed)
2. Electricity term (variable, excluding IEC)  
3. Taxes (DGEG, CAV, IEC)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict
import pandas as pd
from ess.tariff import get_active_tariff_type


@dataclass
class Invoice:
    """Invoice with modular cost breakdown."""
    without_battery: Dict[str, float]
    with_battery: Dict[str, float]
    total_without_battery: float
    total_with_battery: float
    savings: float
    
    # Additional breakdown
    power_term: Dict[str, float]
    electricity_term_without: Dict[str, float]
    electricity_term_with: Dict[str, float]
    taxes: Dict[str, float]

    vat_breakdown: Dict[str, Dict[str, float]]
    period_breakdown: Dict[str, Dict[str, Dict[str, float]]]


class BillingEngine:
    """Compute energy costs with modular invoice structure."""

    def __init__(self, tariff_cfg: Dict, fixed_costs: Dict):
        """
        Parameters
        ----------
        tariff_cfg : Dict
            Tariff configuration
        fixed_costs : Dict
            Dictionary with daily fixed costs breakdown
        """
        self.tariff_cfg = tariff_cfg
        self.fixed_costs = fixed_costs

    def generate_ledger(self, flows_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate per-interval ledger with modular cost components.
        
        This method now handles VAT calculation for the electricity term,
        which previously lived in older helpers.
        """
        # Determine active tariff type
        active_type = get_active_tariff_type(self.tariff_cfg)

        # Join price data with flows, only including available columns
        available_cols = [c for c in (
            'price_omie_adjusted_eur_kwh', 'k2_eur_kwh',
            'tariff_energy_eur_kwh',
            'price_final_eur_kwh', 'price_omie_eur_kwh', 'tariff_period') if c in prices_df.columns]
        df = flows_df.join(prices_df[available_cols], how='left')

        # Energy columns for both scenarios
        df['energy_without_kwh'] = df['house_consumption_kwh']
        df['energy_with_kwh'] = df['total_grid_import_kwh']

        # ========== ELECTRICITY TERM COMPONENTS (before VAT) ==========
        if active_type == 'indexed':
            # Preserve detailed components for indexed tariffs
            for scenario in ['without', 'with']:
                energy_col = f'energy_{scenario}_kwh'
                df[f'omie_adjusted_{scenario}_eur'] = df['price_omie_adjusted_eur_kwh'] * df[energy_col]
                df[f'k2_{scenario}_eur'] = df['k2_eur_kwh'] * df[energy_col]
                df[f'tariff_{scenario}_eur'] = df['tariff_energy_eur_kwh'] * df[energy_col]
                df[f'electricity_base_{scenario}_eur'] = (
                    (df['price_omie_adjusted_eur_kwh'] + df['k2_eur_kwh'] + df['tariff_energy_eur_kwh'])
                    * df[energy_col]
                )
        else:
            # Simple / Bi-horária: single energy component from final unit price
            unit_col = 'price_final_eur_kwh'
            if unit_col not in df.columns:
                raise KeyError("Non-indexed tariff requires 'price_final_eur_kwh' in price dataframe.")
            for scenario in ['without', 'with']:
                energy_col = f'energy_{scenario}_kwh'
                # Single consolidated energy base
                df[f'electricity_base_{scenario}_eur'] = df[unit_col] * df[energy_col]
                # Explicitly ensure no legacy components linger
                for col in (f'omie_adjusted_{scenario}_eur', f'k2_{scenario}_eur', f'tariff_{scenario}_eur'):
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)

        # ========== IEC TAX (separate from electricity term) ==========
        iec_tax = self.tariff_cfg.get('iec_tax_eur_kwh', 0.0)
        iec_vat_rate = self.tariff_cfg.get('iec_vat_rate', 0.23)

        for scenario in ['without', 'with']:
            energy_col = f'energy_{scenario}_kwh'
            df[f'iec_{scenario}_eur'] = iec_tax * df[energy_col]
            df[f'iec_vat_{scenario}_eur'] = df[f'iec_{scenario}_eur'] * iec_vat_rate

        # ========== DYNAMIC ENERGY VAT (with reduced block logic) ==========
        self._apply_energy_vat(df)

        return df
    
    def _apply_energy_vat(self, df: pd.DataFrame):
        """
        Apply energy VAT with a simplified proportional allocation.
        
        Rule:
        - For the whole invoice period (whatever date range the ledger covers),
          compute a single reduced-VAT allowance proportional to the number of days.
        - Allocate that reduced allowance proportionally to each interval based on its
          energy share, irrespective of chronological order.
        - VAT applies to the electricity term only (not IEC).
        """
        # VAT configuration
        vat_cycle_days = int(self.tariff_cfg.get('vat_cycle_days', 30))
        reduced_block_kwh = float(self.tariff_cfg.get('reduced_vat_kwh_per_30_days', 200))
        reduced_vat_rate = float(self.tariff_cfg.get('reduced_vat_rate', 0.06))
        standard_vat_rate = float(self.tariff_cfg.get('vat_rate', 0.23))

        # Determine how many unique days exist in the ledger timeframe
        if df.index.size == 0:
            return
        days_covered = int(pd.Index(df.index.normalize()).nunique())

        # Total reduced allowance for the whole period
        period_reduced_allowance_kwh = max(0.0, reduced_block_kwh * (days_covered / vat_cycle_days))

        for scenario in ['without', 'with']:
            energy_col = f'energy_{scenario}_kwh'
            base_col = f'electricity_base_{scenario}_eur'  # VAT applies to electricity term only

            # Output columns
            vat_col = f'energy_vat_{scenario}_eur'
            vat_red_col = f'energy_vat_reduced_{scenario}_eur'
            vat_std_col = f'energy_vat_standard_{scenario}_eur'
            eff_rate_col = f'effective_energy_vat_rate_{scenario}'

            reduced_kwh_col = f'reduced_kwh_{scenario}'
            standard_kwh_col = f'standard_kwh_{scenario}'
            electricity_base_reduced_col = f'electricity_base_reduced_{scenario}_eur'
            electricity_base_standard_col = f'electricity_base_standard_{scenario}_eur'

            # Initialize columns
            for col in [vat_col, vat_red_col, vat_std_col, eff_rate_col,
                        reduced_kwh_col, standard_kwh_col,
                        electricity_base_reduced_col, electricity_base_standard_col]:
                df[col] = 0.0

            # Guard clauses
            if energy_col not in df.columns or base_col not in df.columns:
                continue

            total_energy = float(df[energy_col].sum())
            if total_energy <= 0.0:
                continue


            # Reduced allowance capped by total energy for this scenario
            reduced_total = min(period_reduced_allowance_kwh, total_energy)

            print(reduced_total)

            # Proportional weights by energy share
            weights = (df[energy_col] / total_energy).fillna(0.0)

            # Per-interval split of kWh
            reduced_kwh_series = weights * reduced_total
            standard_kwh_series = df[energy_col] - reduced_kwh_series

            # Allocate base proportionally to kWh share within each interval
            # (i.e., base * (reduced_kwh / energy) )
            with pd.option_context('mode.use_inf_as_na', True):
                share_reduced = (reduced_kwh_series / df[energy_col]).fillna(0.0)
            
            reduced_base_series = df[base_col] * share_reduced
            standard_base_series = df[base_col] - reduced_base_series

            # Compute VAT
            vat_reduced_series = reduced_base_series * reduced_vat_rate
            vat_standard_series = standard_base_series * standard_vat_rate
            vat_total_series = vat_reduced_series + vat_standard_series

            # Write results
            df[reduced_kwh_col] = reduced_kwh_series.values
            df[standard_kwh_col] = standard_kwh_series.values
            df[electricity_base_reduced_col] = reduced_base_series.values
            df[electricity_base_standard_col] = standard_base_series.values
            df[vat_red_col] = vat_reduced_series.values
            df[vat_std_col] = vat_standard_series.values
            df[vat_col] = vat_total_series.values

            # Effective VAT rate per interval
            with pd.option_context('mode.use_inf_as_na', True):
                df[eff_rate_col] = (df[vat_col] / df[base_col]).fillna(0.0)

    def _aggregate_vat_breakdown(self, ledger: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Aggregate VAT breakdown for reduced and standard blocks, for both scenarios."""
        breakdown = {}
        for block in ['reduced', 'standard']:
            if block == 'reduced':
                kwh_without_col = 'reduced_kwh_without'
                kwh_with_col = 'reduced_kwh_with'
                base_without_col = 'electricity_base_reduced_without_eur'
                base_with_col = 'electricity_base_reduced_with_eur'
                vat_without_col = 'energy_vat_reduced_without_eur'
                vat_with_col = 'energy_vat_reduced_with_eur'
            else:
                kwh_without_col = 'standard_kwh_without'
                kwh_with_col = 'standard_kwh_with'
                base_without_col = 'electricity_base_standard_without_eur'
                base_with_col = 'electricity_base_standard_with_eur'
                vat_without_col = 'energy_vat_standard_without_eur'
                vat_with_col = 'energy_vat_standard_with_eur'

            breakdown[block] = {
                'kwh_without': float(ledger[kwh_without_col].sum()) if kwh_without_col in ledger.columns else 0.0,
                'kwh_with': float(ledger[kwh_with_col].sum()) if kwh_with_col in ledger.columns else 0.0,
                'base_without_eur': float(ledger[base_without_col].sum()) if base_without_col in ledger.columns else 0.0,
                'base_with_eur': float(ledger[base_with_col].sum()) if base_with_col in ledger.columns else 0.0,
                'vat_without_eur': float(ledger[vat_without_col].sum()) if vat_without_col in ledger.columns else 0.0,
                'vat_with_eur': float(ledger[vat_with_col].sum()) if vat_with_col in ledger.columns else 0.0,
            }
        return breakdown

    def _aggregate_period_breakdown(self, ledger: pd.DataFrame, scenario: str) -> Dict[str, Dict[str, float]]:
        """
        Aggregate period (e.g., bi-hourly) breakdown for the selected scenario.
        This uses the exact interval-by-interval VAT allocation performed earlier,
        not any proportional split.
        """
        if 'tariff_period' not in ledger.columns:
            return {}

        # Scenario-dependent column names
        energy_col = f'energy_{scenario}_kwh'
        base_col = f'electricity_base_{scenario}_eur'
        red_kwh_col = f'reduced_kwh_{scenario}'
        std_kwh_col = f'standard_kwh_{scenario}'
        vat_red_col = f'energy_vat_reduced_{scenario}_eur'
        vat_std_col = f'energy_vat_standard_{scenario}_eur'

        if any(col not in ledger.columns for col in (energy_col, base_col, red_kwh_col, std_kwh_col, vat_red_col, vat_std_col)):
            return {}

        periods = sorted(ledger['tariff_period'].dropna().unique())
        result: Dict[str, Dict[str, float]] = {}

        for p in periods:
            mask = ledger['tariff_period'] == p

            kwh = float(ledger.loc[mask, energy_col].sum())
            base_eur = float(ledger.loc[mask, base_col].sum())
            reduced_kwh = float(ledger.loc[mask, red_kwh_col].sum())
            standard_kwh = float(ledger.loc[mask, std_kwh_col].sum())
            vat_reduced_eur = float(ledger.loc[mask, vat_red_col].sum())
            vat_standard_eur = float(ledger.loc[mask, vat_std_col].sum())
            total_with_vat_eur = base_eur + vat_reduced_eur + vat_standard_eur

            # Weighted average unit price for the scenario
            if 'price_final_eur_kwh' in ledger.columns:
                energy_series = ledger.loc[mask, energy_col]
                price_series = ledger.loc[mask, 'price_final_eur_kwh']
                weighted_sum = float((energy_series * price_series).sum())
                total_energy = float(energy_series.sum())
                avg_unit_price = (weighted_sum / total_energy) if total_energy > 0 else 0.0
            else:
                avg_unit_price = 0.0

            result[p] = {
                'kwh': kwh,
                'avg_unit_price_eur_kwh': avg_unit_price,
                'base_eur': base_eur,
                'reduced_kwh': reduced_kwh,
                'standard_kwh': standard_kwh,
                'vat_reduced_eur': vat_reduced_eur,
                'vat_standard_eur': vat_standard_eur,
                'total_with_vat_eur': total_with_vat_eur,
            }

        return result
    
    def _aggregate_electricity_term(self, ledger: pd.DataFrame, scenario: str) -> Dict[str, float]:
        # If detailed components exist (indexed), return them. Otherwise, return a single consolidated key.
        omie_col = f'omie_adjusted_{scenario}_eur'
        k2_col = f'k2_{scenario}_eur'
        tar_col = f'tariff_{scenario}_eur'
        base_col = f'electricity_base_{scenario}_eur'

        if all(col in ledger.columns for col in (omie_col, k2_col, tar_col)):
            return {
                'OMIE Market (adjusted)': ledger[omie_col].sum(),
                'Network Access (K2)': ledger[k2_col].sum(),
                'Time-of-Use Tariff': ledger[tar_col].sum(),
            }
        # Non-indexed: provide a single line
        return {
            'Energy (Fixed)': ledger[base_col].sum(),
        }
    
    def _aggregate_power_term(self, n_days: int) -> Dict[str, float]:
        """Aggregate power term components (fixed costs)."""
        return {
            'K3 Daily Fee': self.fixed_costs['k3_daily'] * n_days,
            'Contracted Power': self.fixed_costs['power_daily'] * n_days,  # Includes VAT
        }
    
    def _aggregate_taxes(self, ledger: pd.DataFrame, n_days: int) -> Dict[str, float]:
        """Aggregate all taxes (IEC, CAV, DGEG)."""
        # IEC is same for both scenarios (depends on consumption)
        iec_base = ledger['iec_without_eur'].sum()  # Same as 'iec_with_eur' for house consumption
        iec_vat = ledger['iec_vat_without_eur'].sum()
        
        return {
            'IEC Tax': iec_base,
            'IEC VAT': iec_vat,
            'CAV (incl. VAT)': self.fixed_costs['cav_daily'] * n_days,
            'DGEG (incl. VAT)': self.fixed_costs['dgeg_daily'] * n_days,
        }
    
    def invoice_from_ledger(self, ledger: pd.DataFrame) -> Invoice:
        """
        Generate modular invoice from ledger.
        
        Returns an Invoice with separated:
        - Power term (fixed)
        - Electricity term (variable, excluding IEC)
        - Taxes (IEC, CAV, DGEG)
        """
        # Total days covered
        n_days = (ledger.index[-1].date() - ledger.index[0].date()).days + 1
        
        # ========== POWER TERM (Fixed) ==========
        power_term = self._aggregate_power_term(n_days)
        
        # ========== ELECTRICITY TERM (Variable) ==========
        electricity_without = self._aggregate_electricity_term(ledger, 'without') 
        electricity_with = self._aggregate_electricity_term(ledger, 'with')
        
        # Add energy VAT to electricity term
        electricity_without['Energy VAT'] = float(ledger['energy_vat_without_eur'].sum())
        electricity_with['Energy VAT'] = float(ledger['energy_vat_with_eur'].sum())
        
        # ========== TAXES ==========
        taxes = self._aggregate_taxes(ledger, n_days)

        # ========== VAT BREAKDOWN & PERIOD BREAKDOWN ==========
        vat_breakdown = self._aggregate_vat_breakdown(ledger)
        period_breakdown = {
            'without': self._aggregate_period_breakdown(ledger, 'without'),
            'with': self._aggregate_period_breakdown(ledger, 'with'),
        }

        # ========== TOTAL INVOICE ==========
        # Combine all components for backward compatibility
        without_battery = {
            **electricity_without,
            **power_term,
            **taxes
        }

        with_battery = {
            **electricity_with,
            **power_term,  # Power term is the same
            **taxes  # Most taxes are the same except IEC might vary slightly
        }

        # For with_battery scenario, recalculate IEC based on actual consumption
        with_battery['IEC Tax'] = float(ledger['iec_with_eur'].sum())
        with_battery['IEC VAT'] = float(ledger['iec_vat_with_eur'].sum())

        # Calculate totals
        total_without = sum(without_battery.values())
        total_with = sum(with_battery.values())
        savings = total_without - total_with

        return Invoice(
            without_battery=without_battery,
            with_battery=with_battery,
            total_without_battery=total_without,
            total_with_battery=total_with,
            savings=savings,
            power_term=power_term,
            electricity_term_without=electricity_without,
            electricity_term_with=electricity_with,
            taxes=taxes,
            vat_breakdown=vat_breakdown,
            period_breakdown=period_breakdown,
        )
    
    def metrics_from_ledger(self, ledger: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary metrics from ledger."""
        invoice = self.invoice_from_ledger(ledger)
        
        savings_pct = (invoice.savings / invoice.total_without_battery * 100) if invoice.total_without_battery else 0.0
        
        # Calculate subtotals for analysis
        electricity_cost_without = sum(invoice.electricity_term_without.values())
        electricity_cost_with = sum(invoice.electricity_term_with.values())
        power_cost = sum(invoice.power_term.values())
        taxes_cost = sum(invoice.taxes.values())
        
        return {
            # Main metrics
            'cost_without_battery_eur': invoice.total_without_battery,
            'cost_with_battery_eur': invoice.total_with_battery,
            'savings_eur': invoice.savings,
            'savings_pct': savings_pct,
            
            # Modular breakdown
            'electricity_cost_without_eur': electricity_cost_without,
            'electricity_cost_with_eur': electricity_cost_with,
            'electricity_savings_eur': electricity_cost_without - electricity_cost_with,
            'power_cost_eur': power_cost,
            'taxes_cost_eur': taxes_cost,
            
            # Component percentages
            'electricity_pct_of_total': (electricity_cost_without / invoice.total_without_battery * 100) if invoice.total_without_battery else 0,
            'power_pct_of_total': (power_cost / invoice.total_without_battery * 100) if invoice.total_without_battery else 0,
            'taxes_pct_of_total': (taxes_cost / invoice.total_without_battery * 100) if invoice.total_without_battery else 0,
        }