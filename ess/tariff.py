"""Tariff helpers and processors used across the project."""

from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


def _is_summer(date: datetime) -> bool:
    """Approximate check for Portuguese summer season (Apr-Oct)."""

    return 4 <= date.month <= 10


def _minutes(t) -> int:
    return t.hour * 60 + t.minute


def _period_daily(ts: datetime, option: str, season: str) -> str:
    m = _minutes(ts.time())
    if option == "bi":
        if m >= 22 * 60 or m < 8 * 60:
            return "vazio"
        return "fora_vazio"
    if option == "tri":
        if m >= 22 * 60 or m < 8 * 60:
            return "vazio"
        if season == "winter":
            if (8 * 60 + 30) <= m < (10 * 60 + 30) or (18 * 60) <= m < (20 * 60 + 30):
                return "ponta"
            return "cheias"
        if (10 * 60 + 30) <= m < (12 * 60) or (19 * 60 + 30) <= m < (21 * 60):
            return "ponta"
        return "cheias"
    return "simples"


def _period_weekly(ts: datetime, option: str, season: str) -> str:
    m = _minutes(ts.time())
    wd = ts.weekday()  # Monday=0
    if option == "bi":
        if wd <= 4:  # weekdays
            return "vazio" if m < 7 * 60 else "fora_vazio"
        if wd == 5:  # Saturday
            if season == "winter":
                if (
                    m < 9 * 60 + 30
                    or 13 * 60 <= m < 18 * 60 + 30
                    or m >= 22 * 60
                ):
                    return "vazio"
                return "fora_vazio"
            if (
                m < 9 * 60
                or 14 * 60 <= m < 20 * 60
                or m >= 22 * 60
            ):
                return "vazio"
            return "fora_vazio"
        return "vazio"  # Sunday
    if option == "tri":
        if wd <= 4:  # weekdays
            if m < 7 * 60:
                return "vazio"
            if season == "winter":
                if 9 * 60 + 30 <= m < 12 * 60 or 18 * 60 + 30 <= m < 21 * 60:
                    return "ponta"
                return "cheias"
            if 9 * 60 + 15 <= m < 12 * 60 + 15:
                return "ponta"
            return "cheias"
        if wd == 5:  # Saturday
            if season == "winter":
                if 9 * 60 + 30 <= m < 13 * 60 or 18 * 60 + 30 <= m < 22 * 60:
                    return "cheias"
                return "vazio"
            if 9 * 60 <= m < 14 * 60 or 20 * 60 <= m < 22 * 60:
                return "cheias"
            return "vazio"
        return "vazio"  # Sunday
    return "simples"


def get_tariff_period(ts: datetime, option: str, cycle: str) -> str:
    season = "summer" if _is_summer(ts) else "winter"
    if cycle == "daily":
        return _period_daily(ts, option, season)
    return _period_weekly(ts, option, season)


class TariffProcessor:
    """Base class for all tariff processors."""

    def __init__(self, tariff_cfg: Dict):
        self.config = tariff_cfg

    def apply(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe with all mandatory tariff columns populated."""

        raise NotImplementedError

    def compute_daily_fixed_costs(
        self, contracted_power_kva: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return fixed term daily costs and metadata for logging."""

        raise NotImplementedError

    def summary_lines(self) -> List[str]:
        """Human readable summary lines for console output."""

        return []


class IndexedTariffProcessor(TariffProcessor):
    """Processor for indexed tariffs (OMIE + regulated components)."""

    def __init__(self, tariff_cfg: Dict):
        super().__init__(tariff_cfg)
        if "indexed" not in tariff_cfg:
            raise ValueError("Indexed tariff configuration requires 'indexed' section")
        self.idx_cfg = tariff_cfg["indexed"]

    def apply(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        df = prices_df.copy()

        option = self.idx_cfg.get("option", "simples")
        cycle = self.idx_cfg.get("cycle", "daily")
        k1 = float(self.idx_cfg.get("k1", 1.0))
        k2 = float(self.idx_cfg.get("k2_eur_kwh", 0.0))
        losses = float(self.idx_cfg.get("losses_pct", 0.0))
        rates_cfg = self.idx_cfg.get("tariff_energy_eur_kwh", {})

        periods: List[str] = []
        omie_adjusted: List[float] = []
        k2_list: List[float] = []
        tariffs: List[float] = []
        electricity_base: List[float] = []

        for ts, row in df.iterrows():
            period = get_tariff_period(ts, option, cycle)
            periods.append(period)

            omie_price = float(row.get("price_omie_eur_kwh", 0.0))
            omie_adj = omie_price * (1 + losses) * k1
            omie_adjusted.append(omie_adj)

            k2_list.append(k2)

            tar = 0.0
            if option == "simples":
                tar = float(rates_cfg.get("simples", 0.0))
            elif option == "bi":
                tar = float(rates_cfg.get("bi", {}).get(period, 0.0))
            elif option == "tri":
                tar = float(rates_cfg.get("tri", {}).get(period, 0.0))
            tariffs.append(tar)

            electricity_base.append(omie_adj + k2 + tar)

        df["tariff_period"] = periods
        df["price_omie_adjusted_eur_kwh"] = pd.Series(omie_adjusted, index=df.index)
        df["k2_eur_kwh"] = pd.Series(k2_list, index=df.index)
        df["tariff_energy_eur_kwh"] = pd.Series(tariffs, index=df.index)
        df["price_electricity_base_eur_kwh"] = pd.Series(
            electricity_base, index=df.index
        )
        df["price_final_eur_kwh"] = df["price_electricity_base_eur_kwh"]

        return df

    def compute_daily_fixed_costs(
        self, contracted_power_kva: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        k3_daily = float(self.idx_cfg.get("k3_eur_day", 0.0))
        power_base = float(self.idx_cfg.get("tariff_power_eur_kva_day", 0.0)) * contracted_power_kva

        threshold = float(self.config.get("fixed_power_reduced_vat_threshold_kva", 6.9))
        reduced_rate = float(self.config.get("fixed_power_reduced_vat_rate", 0.06))
        standard_rate = float(self.config.get("fixed_power_vat_rate", 0.23))

        if contracted_power_kva <= threshold:
            vat_rate = reduced_rate
            vat_type = "reduced"
        else:
            vat_rate = standard_rate
            vat_type = "standard"

        power_daily = power_base * (1 + vat_rate)

        terms = {
            "k3_daily": k3_daily,
            "power_daily": power_daily,
        }

        metadata = {
            "power_vat_rate": vat_rate,
            "vat_type": vat_type,
            "threshold_kva": threshold,
            "contracted_power_kva": contracted_power_kva,
        }

        if "k3_eur_day" in self.idx_cfg:
            metadata["k3_source"] = "configured"

        return terms, metadata

    def summary_lines(self) -> List[str]:
        lines = []
        option = self.idx_cfg.get("option", "simples")
        cycle = self.idx_cfg.get("cycle", "daily")
        lines.append(f"Option: {option}")
        lines.append(f"Cycle: {cycle}")

        lines.append(
            f"Indexed modifiers -> k1: {self.idx_cfg.get('k1', 1.0)}, "
            f"k2: €{self.idx_cfg.get('k2_eur_kwh', 0.0):.4f}/kWh, "
            f"losses: {self.idx_cfg.get('losses_pct', 0.0) * 100:.2f}%"
        )

        rates_cfg = self.idx_cfg.get("tariff_energy_eur_kwh", {})
        if option == "simples":
            if "simples" in rates_cfg:
                lines.append(
                    f"Energy add-on (simples): €{rates_cfg['simples']:.4f}/kWh"
                )
        elif option == "bi":
            bi_rates = rates_cfg.get("bi", {})
            if bi_rates:
                lines.append("Energy add-on (bi-horária):")
                for period_key, label in (
                    ("fora_vazio", "Fora de vazio"),
                    ("vazio", "Vazio"),
                ):
                    if period_key in bi_rates:
                        lines.append(
                            f"  {label}: €{bi_rates[period_key]:.4f}/kWh"
                        )
        elif option == "tri":
            tri_rates = rates_cfg.get("tri", {})
            if tri_rates:
                lines.append("Energy add-on (tri-horária):")
                for period_key in ("ponta", "cheias", "vazio"):
                    if period_key in tri_rates:
                        lines.append(
                            f"  {period_key.capitalize()}: €{tri_rates[period_key]:.4f}/kWh"
                        )

        power_term = float(
            self.idx_cfg.get("tariff_power_eur_kva_day", 0.0)
        )
        lines.append(
            f"Power term base: €{power_term:.4f}/kVA/day (VAT applied later)"
        )

        return lines


class SimpleTariffProcessor(TariffProcessor):
    """Processor for simple (flat) tariffs with pre-defined prices."""

    def __init__(self, tariff_cfg: Dict):
        super().__init__(tariff_cfg)
        section = None
        for key in ("simples", "simple"):
            if key in tariff_cfg:
                section = tariff_cfg[key]
                break
        if section is None:
            raise ValueError("Simple tariff requires 'simples' section")
        self.simple_cfg = section

    def _get_energy_price(self) -> float:
        price_cfg = self.simple_cfg.get("price_electricity_eur_kwh")
        if isinstance(price_cfg, dict):
            price = price_cfg.get("simples")
        else:
            price = price_cfg
        if price is None:
            raise ValueError("Simple tariff missing 'price_electricity_eur_kwh'")
        return float(price)

    def _get_power_term_daily(self) -> float:
        for key in (
            "power_term_daily_eur",
            "fixed_power_term_daily_eur",
            "power_term_eur_day",
            "tariff_power_eur_day",
        ):
            if key in self.simple_cfg:
                return float(self.simple_cfg[key])
        raise ValueError("Simple tariff missing daily power term value")

    def apply(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        df = prices_df.copy()
        base_price = self._get_energy_price()

        df["tariff_period"] = "simples"
        df["price_omie_adjusted_eur_kwh"] = 0.0
        df["k2_eur_kwh"] = 0.0
        df["tariff_energy_eur_kwh"] = base_price
        df["price_electricity_base_eur_kwh"] = base_price
        df["price_final_eur_kwh"] = base_price
        df["price_omie_eur_kwh"] = base_price

        return df

    def compute_daily_fixed_costs(
        self, contracted_power_kva: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        k3_daily = float(self.simple_cfg.get("k3_eur_day", 0.0))
        power_daily = self._get_power_term_daily()
        terms = {
            "k3_daily": k3_daily,
            "power_daily": power_daily,
        }
        metadata = {
            "power_term_source": "provided_final",
            "k3_source": "configured" if "k3_eur_day" in self.simple_cfg else "absent",
        }
        return terms, metadata

    def summary_lines(self) -> List[str]:
        lines = [
            f"Energy price (flat): €{self._get_energy_price():.4f}/kWh",
            f"Power term (daily, provided): €{self._get_power_term_daily():.4f}",
        ]
        if "k3_eur_day" in self.simple_cfg:
            lines.append(f"K3 daily (provided): €{self.simple_cfg['k3_eur_day']:.4f}")
        return lines


class BiHourlyTariffProcessor(TariffProcessor):
    """Processor for bi-horária tariffs with pre-defined period prices."""

    def __init__(self, tariff_cfg: Dict):
        super().__init__(tariff_cfg)
        section = None
        for key in ("bi_horaria", "bi-horaria", "bi"):  # accept common variants
            if key in tariff_cfg:
                section = tariff_cfg[key]
                break
        if section is None:
            raise ValueError("Bi-horária tariff requires 'bi_horaria' section")
        self.bi_cfg = section

    def _get_rates(self) -> Dict[str, float]:
        rates = self.bi_cfg.get("price_electricity_eur_kwh", {})
        if "bi" in rates and isinstance(rates["bi"], dict):
            rates = rates["bi"]
        if not rates:
            raise ValueError("Bi-horária tariff missing energy price mapping")
        return {k: float(v) for k, v in rates.items()}

    def _get_power_term_daily(self) -> float:
        for key in (
            "power_term_daily_eur",
            "fixed_power_term_daily_eur",
            "power_term_eur_day",
            "tariff_power_eur_day",
        ):
            if key in self.bi_cfg:
                return float(self.bi_cfg[key])
        raise ValueError("Bi-horária tariff missing daily power term value")

    def apply(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        df = prices_df.copy()
        option = self.bi_cfg.get("option", "bi")
        cycle = self.bi_cfg.get("cycle", "daily")
        rates = self._get_rates()

        periods: List[str] = []
        base_prices: List[float] = []

        default_fora_vazio = rates.get("fora_vazio")

        for ts in df.index:
            period = get_tariff_period(ts, option, cycle)
            price = rates.get(period)

            if price is None:
                # Fallback to fora_vazio for any undefined non-vazio period
                if period != "vazio" and default_fora_vazio is not None:
                    price = default_fora_vazio
                elif "default" in rates:
                    price = rates["default"]
            if price is None:
                raise ValueError(
                    f"No price configured for period '{period}' in bi-horária tariff"
                )

            periods.append(period)
            base_prices.append(float(price))

        base_series = pd.Series(base_prices, index=df.index)
        df["tariff_period"] = periods
        df["price_omie_adjusted_eur_kwh"] = 0.0
        df["k2_eur_kwh"] = 0.0
        df["tariff_energy_eur_kwh"] = base_series
        df["price_electricity_base_eur_kwh"] = base_series
        df["price_final_eur_kwh"] = base_series
        df["price_omie_eur_kwh"] = base_series

        return df

    def compute_daily_fixed_costs(
        self, contracted_power_kva: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        k3_daily = float(self.bi_cfg.get("k3_eur_day", 0.0))
        power_daily = self._get_power_term_daily()
        terms = {
            "k3_daily": k3_daily,
            "power_daily": power_daily,
        }
        metadata = {
            "power_term_source": "provided_final",
            "k3_source": "configured" if "k3_eur_day" in self.bi_cfg else "absent",
        }
        return terms, metadata

    def summary_lines(self) -> List[str]:
        rates = self._get_rates()
        lines = [f"Cycle: {self.bi_cfg.get('cycle', 'daily')}"]
        lines.append("Energy price (bi-horária):")
        for period_key, label in (
            ("fora_vazio", "Fora de vazio"),
            ("vazio", "Vazio"),
        ):
            if period_key in rates:
                lines.append(f"  {label}: €{rates[period_key]:.4f}/kWh")
        for extra in sorted(set(rates.keys()) - {"fora_vazio", "vazio"}):
            lines.append(f"  {extra}: €{rates[extra]:.4f}/kWh")

        lines.append(
            f"Power term (daily, provided): €{self._get_power_term_daily():.4f}"
        )
        if "k3_eur_day" in self.bi_cfg:
            lines.append(f"K3 daily (provided): €{self.bi_cfg['k3_eur_day']:.4f}")
        return lines


def create_tariff_processor(tariff_cfg: Dict) -> TariffProcessor:
    """Factory returning the appropriate processor for the given tariff type."""

    tariff_type = tariff_cfg.get("type", "indexed").lower()

    if tariff_type == "indexed":
        return IndexedTariffProcessor(tariff_cfg)
    if tariff_type in {"simples", "simple"}:
        return SimpleTariffProcessor(tariff_cfg)
    if tariff_type in {"bi_horaria", "bi-horaria", "bi"}:
        return BiHourlyTariffProcessor(tariff_cfg)

    raise ValueError(f"Unsupported tariff type: {tariff_type}")


def apply_tariff(prices_df: pd.DataFrame, tariff_cfg: Dict) -> pd.DataFrame:
    """Convenience wrapper that applies the configured tariff to a dataframe."""

    processor = create_tariff_processor(tariff_cfg)
    return processor.apply(prices_df)


def apply_indexed_tariff(prices_df: pd.DataFrame, tariff_cfg: dict) -> pd.DataFrame:
    """Backward compatible helper kept for existing imports."""

    processor = IndexedTariffProcessor(tariff_cfg)
    return processor.apply(prices_df)
