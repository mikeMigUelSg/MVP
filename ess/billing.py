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
        which was previously done in apply_indexed_tariff.
        """
        # Join price data with flows
        df = flows_df.join(prices_df[['price_omie_adjusted_eur_kwh', 'k2_eur_kwh', 
                                     'tariff_energy_eur_kwh', 'price_electricity_base_eur_kwh']], 
                          how='left')
        
        # Energy columns for both scenarios
        df['energy_without_kwh'] = df['house_consumption_kwh']
        df['energy_with_kwh'] = df['total_grid_import_kwh']
        
        # ========== ELECTRICITY TERM COMPONENTS (before VAT) ==========
        for scenario in ['without', 'with']:
            energy_col = f'energy_{scenario}_kwh'
            
            # OMIE adjusted component (OMIE * losses * k1)
            df[f'omie_adjusted_{scenario}_eur'] = df['price_omie_adjusted_eur_kwh'] * df[energy_col]
            
            # K2 network access component
            df[f'k2_{scenario}_eur'] = df['k2_eur_kwh'] * df[energy_col]
            
            # Time-of-use tariff component
            df[f'tariff_{scenario}_eur'] = df['tariff_energy_eur_kwh'] * df[energy_col]
            
            # Total electricity base (before VAT) - using the pre-calculated unit price
            # This is equivalent to: omie_adjusted + k2 + tariff (all in EUR)
            df[f'electricity_base_{scenario}_eur'] = df['price_electricity_base_eur_kwh'] * df[energy_col]
        
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
        Apply dynamic energy VAT with reduced block logic.
        VAT applies to the electricity term only (not IEC).
        """
        # VAT configuration
        vat_cycle_days = int(self.tariff_cfg.get('vat_cycle_days', 30))
        reduced_block_kwh = float(self.tariff_cfg.get('reduced_vat_kwh_per_30_days', 200))
        reduced_vat_rate = float(self.tariff_cfg.get('reduced_vat_rate', 0.06))
        standard_vat_rate = float(self.tariff_cfg.get('vat_rate', 0.23))
        
        # Anchor date for VAT cycles
        if 'vat_cycle_anchor_date' in self.tariff_cfg:
            anchor = pd.Timestamp(self.tariff_cfg['vat_cycle_anchor_date']).normalize()
        else:
            anchor = df.index[0].normalize()
        
        # Compute cycle ID for each row
        df['_cycle_id'] = ((df.index.normalize() - anchor).days // vat_cycle_days).astype(int)
        
        for scenario in ['without', 'with']:
            energy_col = f'energy_{scenario}_kwh'
            base_col = f'electricity_base_{scenario}_eur'  # VAT applies to electricity term only
            
            # Output columns
            vat_col = f'energy_vat_{scenario}_eur'
            vat_red_col = f'energy_vat_reduced_{scenario}_eur'
            vat_std_col = f'energy_vat_standard_{scenario}_eur'
            eff_rate_col = f'effective_energy_vat_rate_{scenario}'
            
            df[vat_col] = 0.0
            df[vat_red_col] = 0.0
            df[vat_std_col] = 0.0
            df[eff_rate_col] = 0.0
            
            # Apply VAT cycle by cycle
            for cid, grp in df.groupby('_cycle_id', sort=True):
                idx = grp.index
                if len(idx) == 0:
                    continue
                
                # Pro-rate the reduced block
                days_covered = int(idx.normalize().nunique())
                cycle_threshold_kwh = reduced_block_kwh * (days_covered / vat_cycle_days)
                remaining_reduced = max(0.0, cycle_threshold_kwh)
                
                for ts in idx:
                    e = float(df.at[ts, energy_col])
                    base = float(df.at[ts, base_col])
                    
                    if e <= 0.0 or base <= 0.0:
                        df.at[ts, eff_rate_col] = 0.0
                        continue
                    
                    # Split energy between reduced and standard blocks
                    reduced_kwh = min(e, max(0.0, remaining_reduced))
                    standard_kwh = max(0.0, e - reduced_kwh)
                    
                    # Allocate base amount proportionally
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
                    
                    # Update remaining reduced allowance
                    remaining_reduced -= reduced_kwh
        
        # Clean up helper column
        df.drop(columns=['_cycle_id'], inplace=True)
    
    def _aggregate_electricity_term(self, ledger: pd.DataFrame, scenario: str) -> Dict[str, float]:
        """Aggregate electricity term components (excluding IEC and VAT)."""
        return {
            'OMIE Market (adjusted)': ledger[f'omie_adjusted_{scenario}_eur'].sum(),
            'Network Access (K2)': ledger[f'k2_{scenario}_eur'].sum(),
            'Time-of-Use Tariff': ledger[f'tariff_{scenario}_eur'].sum(),
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
            taxes=taxes
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