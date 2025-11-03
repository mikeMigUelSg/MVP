"""
Calibração offline para controlador SDP.

Implementa:
1. Ajuste de t_1/2 por MSE (minimizar erro de previsão)
2. Estimativa inicial de σ
3. Afinação por desempenho em malha-fechada
"""

import sys
import os
# Add project root to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

try:
    from .sdp_controller import ResidualModel
except ImportError:
    from src.controllers.sdp_controller import ResidualModel


class SDPCalibrator:
    """
    Calibrador de parâmetros para SDP.

    Ajusta t_1/2 e σ usando dados históricos.
    """

    def __init__(self,
                 historical_data: pd.DataFrame,
                 dt_minutes: int = 15):
        """
        Args:
            historical_data: DataFrame com colunas ['timestamp', 'P_pv', 'P_load']
            dt_minutes: Passo temporal (minutos)
        """
        self.data = historical_data.copy()
        self.dt_minutes = dt_minutes

        # Processar dados
        self._process_data()

    def _process_data(self):
        """Processar dados históricos."""
        # Garantir que timestamp é datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['timestamp']):
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        # Calcular resíduo
        self.data['R'] = self.data['P_pv'] - self.data['P_load']

        # Ordenar por timestamp
        self.data = self.data.sort_values('timestamp')

    def calibrate_t_half(self,
                        R_bar: np.ndarray,
                        R_actual: np.ndarray,
                        t_half_candidates: List[float]) -> Tuple[float, Dict]:
        """
        Calibrar t_1/2 por MSE.

        Testa diferentes valores de t_1/2 e escolhe o que minimiza
        o MSE entre previsão determinística e medições.

        Args:
            R_bar: Previsão base R̄ (array)
            R_actual: Medições reais R (array)
            t_half_candidates: Lista de candidatos para t_1/2 (minutos)

        Returns:
            (melhor_t_half, resultados) onde resultados contém MSE de cada candidato
        """
        results = {}

        for t_half in t_half_candidates:
            # Criar modelo
            model = ResidualModel(t_half, self.dt_minutes, sigma=0.0)

            # Gerar previsão determinística
            R_0 = R_actual[0]
            R_hat = model.deterministic_forecast(R_bar, R_0)

            # Calcular MSE
            mse = np.mean((R_hat - R_actual)**2)  # usa R_hat, não R_bar
            results[t_half] = mse

        # Encontrar melhor
        best_t_half = min(results, key=results.get)

        return best_t_half, results

    def estimate_sigma(self,
                      t_half: float,
                      R_bar: np.ndarray,
                      R_actual: np.ndarray) -> float:
        """
        Estimar σ inicial.

        Usa σ² ≈ MSE * (1 - κ²), onde MSE é o erro residual
        após aplicar o modelo de meia-vida.

        Args:
            t_half: Meia-vida escolhida (minutos)
            R_bar: Previsão base R̄
            R_actual: Medições reais R

        Returns:
            σ estimado (kW)
        """
        # Criar modelo
        model = ResidualModel(t_half, self.dt_minutes, sigma=0.0)

        # Gerar previsão determinística
        R_0 = R_actual[0]
        R_hat = model.deterministic_forecast(R_bar, R_0)

        # Calcular MSE
        mse = np.mean((R_bar - R_actual)**2)

        # Estimar σ
        kappa = model.kappa
        if kappa < 1.0:
            sigma_est = np.sqrt(max(mse, 0.0) * (1.0 - model.kappa**2))
        else:
            sigma_est = np.sqrt(mse)

        return sigma_est

    def calibrate_from_timeseries(self,
                                 start_time: datetime,
                                 horizon_hours: int = 9,
                                 t_half_range: Tuple[float, float, float] = (15.0, 120.0, 15.0),
                                 pv_forecaster=None,
                                 load_forecaster=None) -> Dict:
        """
        Calibrar usando uma janela temporal dos dados.

        Args:
            start_time: Início da janela
            horizon_hours: Duração da janela (horas)
            t_half_range: (min, max, step) para busca de t_1/2 (minutos)
            pv_forecaster: Forecaster de PV (para gerar R̄)
            load_forecaster: Forecaster de carga (para gerar R̄)

        Returns:
            Dict com resultados da calibração
        """
        # Extrair janela
        end_time = start_time + timedelta(hours=horizon_hours)
        mask = (self.data['timestamp'] >= start_time) & (self.data['timestamp'] < end_time)
        window_data = self.data[mask]

        if len(window_data) == 0:
            raise ValueError(f"No data in window {start_time} to {end_time}")

        # Medições reais
        R_actual = window_data['R'].values

        # Gerar previsão base R̄
        n_steps = len(R_actual)

        if pv_forecaster is not None and load_forecaster is not None:
            P_pv_bar = pv_forecaster.get_forecast(start_time, n_steps)
            P_load_bar = load_forecaster.get_forecast(start_time, n_steps)
            R_bar = P_pv_bar - P_load_bar
        else:
            # Fallback: usar persistência simples
            R_bar = np.full(n_steps, R_actual[0])

        # Candidatos para t_1/2
        t_half_min, t_half_max, t_half_step = t_half_range
        t_half_candidates = np.arange(t_half_min, t_half_max + t_half_step, t_half_step)

        # Calibrar t_1/2
        best_t_half, mse_results = self.calibrate_t_half(
            R_bar, R_actual, t_half_candidates.tolist()
        )

        # Estimar σ
        best_sigma = self.estimate_sigma(best_t_half, R_bar, R_actual)

        return {
            't_half_minutes': best_t_half,
            'sigma_kw': best_sigma,
            'mse_results': mse_results,
            'best_mse': mse_results[best_t_half]
        }

    def grid_search_closed_loop(self,
                               t_half_candidates: List[float],
                               sigma_candidates: List[float],
                               simulation_days: int = 7,
                               cost_function=None) -> Dict:
        """
        Afinação por busca em grade com simulação em malha-fechada.

        Testa combinações (t_1/2, σ) e escolhe a que minimiza custo total.

        Args:
            t_half_candidates: Lista de candidatos t_1/2 (minutos)
            sigma_candidates: Lista de candidatos σ (kW)
            simulation_days: Número de dias para simular
            cost_function: Função que simula e retorna custo total
                          Assinatura: cost_function(t_half, sigma) -> float

        Returns:
            Dict com melhores parâmetros e resultados
        """
        if cost_function is None:
            raise ValueError("cost_function must be provided for closed-loop tuning")

        results = {}

        for t_half in t_half_candidates:
            for sigma in sigma_candidates:
                print(f"  Testing t_1/2={t_half} min, σ={sigma:.2f} kW...")

                # Simular
                cost = cost_function(t_half, sigma)
                results[(t_half, sigma)] = cost

        # Encontrar melhor
        best_params = min(results, key=results.get)
        best_t_half, best_sigma = best_params

        return {
            't_half_minutes': best_t_half,
            'sigma_kw': best_sigma,
            'cost': results[best_params],
            'all_results': results
        }


def quick_calibrate(historical_data: pd.DataFrame,
                   start_time: datetime,
                   pv_forecaster=None,
                   load_forecaster=None,
                   dt_minutes: int = 15) -> Dict:
    """
    Calibração rápida usando dados históricos.

    Args:
        historical_data: DataFrame com ['timestamp', 'P_pv', 'P_load']
        start_time: Início da janela de calibração
        pv_forecaster: Forecaster de PV
        load_forecaster: Forecaster de carga
        dt_minutes: Passo temporal (minutos)

    Returns:
        Dict com parâmetros calibrados
    """
    calibrator = SDPCalibrator(historical_data, dt_minutes)

    results = calibrator.calibrate_from_timeseries(
        start_time=start_time,
        horizon_hours=24,  # 24h para calibração
        t_half_range=(15.0, 120.0, 15.0),  # 15 a 120 min, passo 15
        pv_forecaster=pv_forecaster,
        load_forecaster=load_forecaster
    )

    print("\n=== Calibração SDP ===")
    print(f"Melhor t_1/2: {results['t_half_minutes']:.1f} min")
    print(f"σ estimado: {results['sigma_kw']:.3f} kW")
    print(f"MSE: {results['best_mse']:.4f}")
    print("======================\n")

    return results


if __name__ == "__main__":
    """
    Calibração usando dados reais de carga e produção FV.
    """
    print("\n" + "="*60)
    print("CALIBRAÇÃO SDP COM DADOS REAIS")
    print("="*60 + "\n")

    # Carregar dados reais
    print("Carregando dados reais...")

    from src.components.house import House
    from src.components.solar import SolarPanel

    # Dados de carga
    load_file = "data/load/merged_consumos.xlsx"
    print(f"  Carregando carga de: {load_file}")
    house = House(load_file)

    # Dados de PV
    pv_file = "data/solar/pvdata.csv"
    print(f"  Carregando PV de: {pv_file}")
    solar = SolarPanel(capacity_kw=5.0, data_file=pv_file)

    # Preparar dados de carga
    load_data = house.consumption_data.copy()
    load_data['timestamp'] = pd.to_datetime(
        load_data.apply(
            lambda row: f"2024-{int(row['month']):02d}-{int(row['day']):02d} "
                       f"{int(row['hour']):02d}:{int(row['minute']):02d}:00",
            axis=1
        )
    )
    load_data = load_data[['timestamp', 'consumption']].rename(
        columns={'consumption': 'P_load'}
    )

    # Preparar dados de PV
    pv_data = solar.production_data.copy()
    if 'timestamp' not in pv_data.columns:
        pv_data['timestamp'] = pd.to_datetime(
            pv_data.apply(
                lambda row: f"2024-{int(row['month']):02d}-{int(row['day']):02d} "
                           f"{int(row['hour']):02d}:{int(row['minute']):02d}:00",
                axis=1
            )
        )

    # Converter produção para kW
    prod_col = 'pv_1' if 'pv_1' in pv_data.columns else 'production'
    pv_data['P_pv'] = pv_data[prod_col] / 1000.0  # W para kW
    pv_data = pv_data[['timestamp', 'P_pv']]

    # Merge dos dados
    historical_data = pd.merge(
        pv_data, load_data, on='timestamp', how='inner'
    )

    print(f"  Dados carregados: {len(historical_data)} pontos")

    # Estatísticas dos dados
    n_days = (historical_data['timestamp'].max() - historical_data['timestamp'].min()).days
    dt_minutes = 15

    print(f"  Período: {historical_data['timestamp'].min()} a {historical_data['timestamp'].max()}")
    print(f"  Total de dias: {n_days}")
    print(f"  Intervalo: {dt_minutes} minutos")

    # Criar calibrador
    print("\nCriando calibrador...")
    calibrator = SDPCalibrator(historical_data, dt_minutes=dt_minutes)

    print(f"  Resíduo médio: {calibrator.data['R'].mean():.3f} kW")
    print(f"  Resíduo std: {calibrator.data['R'].std():.3f} kW")
    print(f"  Resíduo min: {calibrator.data['R'].min():.3f} kW")
    print(f"  Resíduo max: {calibrator.data['R'].max():.3f} kW")

    # Preparar forecasters para calibração mais precisa
    print("\n=== Preparando Forecasters ===")
    from src.forecasters.profile_persistence_forecaster import ProfilePersistenceForecaster

    load_forecaster = ProfilePersistenceForecaster(n_weeks=3, dt_minutes=15)
    pv_forecaster = ProfilePersistenceForecaster(n_weeks=3, dt_minutes=15)

    load_forecaster.set_data(load_data, value_column='P_load')
    pv_forecaster.set_data(pv_data, value_column='P_pv')

    print(f"  Load forecaster: {len(load_data)} pontos")
    print(f"  PV forecaster: {len(pv_data)} pontos")

    # Calibração usando múltiplos dias para robustez
    print("\n=== Calibração Completa (com forecasters) ===")

    # Escolher janela de calibração (primeiro dia completo disponível)
    calib_start = historical_data['timestamp'].min()
    calib_start = datetime(calib_start.year, calib_start.month, calib_start.day, 0, 0, 0)

    # Avançar alguns dias para garantir que forecasters têm dados históricos
    calib_start = calib_start + timedelta(days=7)

    print(f"  Janela de calibração: {calib_start}")
    print(f"  Horizonte: 24h")
    print(f"  Testando t_1/2: 15 a 120 min (passo 15 min)")

    # Calibração completa
    results = calibrator.calibrate_from_timeseries(
        start_time=calib_start,
        horizon_hours=9,
        t_half_range=(15.0, 120.0, 15.0),  # 15 a 120 min, passo 15
        pv_forecaster=pv_forecaster,
        load_forecaster=load_forecaster
    )

    print(f"\n  Resultados MSE por t_1/2:")
    for t_half, mse in sorted(results['mse_results'].items()):
        marker = " <- MELHOR" if t_half == results['t_half_minutes'] else ""
        print(f"    t_1/2 = {t_half:5.1f} min  ->  MSE = {mse:.4f}{marker}")

    # Teste em outro dia para validação
    print("\n=== Validação em Dia Diferente ===")

    val_start = calib_start + timedelta(days=7)
    print(f"  Dia de validação: {val_start.date()}")

    val_results = calibrator.calibrate_from_timeseries(
        start_time=val_start,
        horizon_hours=24,
        t_half_range=(15.0, 120.0, 15.0),
        pv_forecaster=pv_forecaster,
        load_forecaster=load_forecaster
    )

    print(f"  t_1/2 validação: {val_results['t_half_minutes']:.1f} min")
    print(f"  σ validação: {val_results['sigma_kw']:.3f} kW")
    print(f"  MSE validação: {val_results['best_mse']:.4f}")

    # Resumo final
    print("\n" + "="*60)
    print("PARÂMETROS RECOMENDADOS PARA SDP")
    print("="*60)
    print(f"\n  t_1/2 (meia-vida):  {results['t_half_minutes']:.1f} minutos")
    print(f"  σ (ruído):          {results['sigma_kw']:.3f} kW")
    print(f"  MSE calibração:     {results['best_mse']:.4f}")
    print(f"\n  Validação:")
    print(f"    t_1/2:            {val_results['t_half_minutes']:.1f} minutos")
    print(f"    σ:                {val_results['sigma_kw']:.3f} kW")
    print(f"    MSE:              {val_results['best_mse']:.4f}")

    # Sugestão de uso
    avg_t_half = (results['t_half_minutes'] + val_results['t_half_minutes']) / 2
    avg_sigma = (results['sigma_kw'] + val_results['sigma_kw']) / 2

    print(f"\n  Valores médios sugeridos:")
    print(f"    t_1/2 = {avg_t_half:.1f} min")
    print(f"    σ = {avg_sigma:.3f} kW")

    print("\n" + "="*60)
    print("Para usar estes parâmetros:")
    print("  1. Copie os valores para config.yaml")
    print("  2. Ou use em SDPParams(t_half_minutes=..., sigma_R=...)")
    print("="*60 + "\n")
