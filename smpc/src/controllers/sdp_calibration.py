"""
Calibração offline para controlador SDP.

Protocolo de calibração para (t_1/2) e (σ):

1. Preparar dados e previsões:
   - Previsões determinísticas para PV (fornecedor comercial) e carga (persistência)
   - Persistência: média do mesmo dia da semana nas 3 semanas anteriores
   - Passo: Δt = 15 min, atualizações a cada 3h, horizonte 18h
   - Variável de controle: resíduo R̄_k = P̄_PV,k - P̄_Load,k

2. Modelar resíduo com meia-vida:
   - Modelo determinístico: R̂_{k+1} = R̄_{k+1} + κ(R̂_k - R̄_k)
   - κ = 2^(-Δt/t_{1/2}) faz (R̂ - R̄) "cair para metade" a cada t_{1/2}

3. Obter palpites iniciais (malha-aberta):
   3.1. Varrer t_{1/2} e escolher o que minimiza MSE entre modelo e previsão
   3.2. Estimar σ a partir de: σ² = MSE × (1 - κ²)
        (da variância estacionária do AR(1): Var[R] = σ²/(1-κ²))

4. Afinar por desempenho (malha-fechada, 12 dias):
   4.1. Benchmark com 12 dias (um aleatório por mês disponível)
   4.2. Grade em torno dos palpites (e.g., t_{1/2} ∈ {35,45,60,80} min; σ ∈ [0.35, 0.70] kW)
        - Para cada (t_{1/2}, σ): executar controlador estocástico e somar custo
        - Tarifas: 28 c€/kWh compra, 12.3 c€/kWh injeção
        - Avaliar também autossuficiência e curtailment
        - Configuração: Δt=15min, recalcular a cada 6h, horizonte N=36 (9h),
                       discretizações n_x=150, n_R=50 (≈4.4s por política)
   4.3. Escolher par com menor custo agregado
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
                                 horizon_hours: int = 24,
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

    def select_benchmark_days(self, n_days: int = 12, seed: int = 42) -> List[datetime]:
        """
        Selecionar dias para benchmark (um aleatório por mês disponível).

        Protocolo 4.1: Construir benchmark com 12 dias, um aleatório por mês do ano disponível.

        Args:
            n_days: Número de dias desejado (default: 12)
            seed: Semente para reprodutibilidade

        Returns:
            Lista de timestamps (início de cada dia)
        """
        np.random.seed(seed)

        # Agrupar dados por mês
        self.data['year_month'] = self.data['timestamp'].dt.to_period('M')
        months = self.data['year_month'].unique()

        selected_days = []

        for month in sorted(months)[:n_days]:  # Até n_days meses
            # Dados deste mês
            month_data = self.data[self.data['year_month'] == month]

            # Dias disponíveis neste mês
            month_data['date'] = month_data['timestamp'].dt.date
            available_days = month_data['date'].unique()

            # Selecionar um dia aleatório
            if len(available_days) > 0:
                chosen_day = np.random.choice(available_days)
                selected_days.append(pd.Timestamp(chosen_day))

        # Limpar coluna temporária
        self.data.drop(columns=['year_month'], inplace=True)

        return selected_days

    def grid_search_closed_loop(self,
                               t_half_candidates: List[float],
                               sigma_candidates: List[float],
                               benchmark_days: List[datetime] = None,
                               cost_function=None,
                               tariff_buy_eur_kwh: float = 0.28,
                               tariff_sell_eur_kwh: float = 0.123,
                               verbose: bool = True) -> Dict:
        """
        Afinação por busca em grade com simulação em malha-fechada.

        Protocolo 4.2-4.3: Grade em torno dos palpites iniciais, executar controlador
        estocástico para cada (t_1/2, σ) e escolher par com menor custo agregado.

        Testa combinações (t_1/2, σ) e escolhe a que minimiza custo total.

        Configuração recomendada (do artigo):
        - Δt = 15 min
        - Recalcular políticas a cada 6h
        - Horizonte N = 36 (9h)
        - Discretizações: n_x = 150, n_R = 50
        - Tarifas: 28 c€/kWh compra, 12.3 c€/kWh injeção

        Args:
            t_half_candidates: Lista de candidatos t_1/2 (minutos)
                              Ex: [35, 45, 60, 80] do protocolo
            sigma_candidates: Lista de candidatos σ (kW)
                             Ex: np.arange(0.35, 0.75, 0.05) do protocolo
            benchmark_days: Lista de dias para benchmark (se None, usa select_benchmark_days)
            cost_function: Função que simula e retorna métricas
                          Assinatura: cost_function(t_half, sigma, days) -> Dict
                          Deve retornar: {'cost': float, 'self_sufficiency': float, 'curtailment': float}
            tariff_buy_eur_kwh: Tarifa de compra (€/kWh)
            tariff_sell_eur_kwh: Tarifa de venda (€/kWh)
            verbose: Imprimir progresso

        Returns:
            Dict com melhores parâmetros e resultados detalhados
        """
        if cost_function is None:
            raise ValueError("cost_function must be provided for closed-loop tuning")

        # Selecionar dias de benchmark se não fornecidos
        if benchmark_days is None:
            benchmark_days = self.select_benchmark_days(n_days=12)
            if verbose:
                print(f"\nDias selecionados para benchmark ({len(benchmark_days)}):")
                for i, day in enumerate(benchmark_days, 1):
                    print(f"  {i:2d}. {day.date()}")

        results = {}
        detailed_results = {}

        total_combinations = len(t_half_candidates) * len(sigma_candidates)
        current = 0

        if verbose:
            print(f"\nTestando {total_combinations} combinações de parâmetros...")
            print(f"Tarifas: compra={tariff_buy_eur_kwh:.3f} €/kWh, venda={tariff_sell_eur_kwh:.3f} €/kWh")

        for t_half in t_half_candidates:
            for sigma in sigma_candidates:
                current += 1
                if verbose:
                    print(f"\n[{current}/{total_combinations}] t_1/2={t_half:.0f} min, σ={sigma:.2f} kW")

                # Simular nos dias de benchmark
                metrics = cost_function(t_half, sigma, benchmark_days)

                # Custo total é o critério principal
                total_cost = metrics.get('cost', float('inf'))
                results[(t_half, sigma)] = total_cost
                detailed_results[(t_half, sigma)] = metrics

                if verbose:
                    print(f"    Custo total: {total_cost:.2f} €")
                    if 'self_sufficiency' in metrics:
                        print(f"    Autossuficiência: {metrics['self_sufficiency']:.1f}%")
                    if 'curtailment' in metrics:
                        print(f"    Curtailment: {metrics['curtailment']:.2f} kWh")

        # Encontrar melhor combinação
        best_params = min(results, key=results.get)
        best_t_half, best_sigma = best_params

        if verbose:
            print("\n" + "="*60)
            print("MELHOR COMBINAÇÃO (menor custo)")
            print("="*60)
            print(f"  t_1/2: {best_t_half:.0f} min")
            print(f"  σ: {best_sigma:.2f} kW")
            print(f"  Custo: {results[best_params]:.2f} €")
            if best_params in detailed_results:
                best_metrics = detailed_results[best_params]
                if 'self_sufficiency' in best_metrics:
                    print(f"  Autossuficiência: {best_metrics['self_sufficiency']:.1f}%")
                if 'curtailment' in best_metrics:
                    print(f"  Curtailment: {best_metrics['curtailment']:.2f} kWh")

        return {
            't_half_minutes': best_t_half,
            'sigma_kw': best_sigma,
            'cost': results[best_params],
            'metrics': detailed_results[best_params],
            'all_results': results,
            'all_detailed_results': detailed_results,
            'benchmark_days': benchmark_days
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


def full_calibration_protocol(historical_data: pd.DataFrame,
                              pv_forecaster=None,
                              load_forecaster=None,
                              cost_function=None,
                              dt_minutes: int = 15,
                              horizon_hours: int = 18,
                              initial_date: datetime = None) -> Dict:
    """
    Implementação completa do protocolo de calibração.

    Executa os passos 3 e 4 do protocolo:
    - Passo 3: Calibração malha-aberta (sweep t_1/2, estimativa σ)
    - Passo 4: Afinação malha-fechada em 12 dias benchmark

    Args:
        historical_data: DataFrame com ['timestamp', 'P_pv', 'P_load']
        pv_forecaster: Forecaster de PV (persistência recomendada)
        load_forecaster: Forecaster de carga (persistência recomendada)
        cost_function: Função para simulação em malha-fechada
                      Assinatura: cost_function(t_half, sigma, days) -> Dict
        dt_minutes: Passo temporal (minutos, default: 15)
        horizon_hours: Horizonte para calibração inicial (horas, default: 18)
        initial_date: Data inicial para calibração (se None, usa primeira data disponível)

    Returns:
        Dict com resultados completos:
        - 'open_loop': resultados da calibração malha-aberta
        - 'closed_loop': resultados da calibração malha-fechada
        - 'recommended': parâmetros finais recomendados
    """
    print("\n" + "="*80)
    print("PROTOCOLO COMPLETO DE CALIBRAÇÃO SDP")
    print("="*80)

    calibrator = SDPCalibrator(historical_data, dt_minutes=dt_minutes)

    # ==================== PASSO 3: CALIBRAÇÃO MALHA-ABERTA ====================
    print("\n" + "-"*80)
    print("PASSO 3: CALIBRAÇÃO MALHA-ABERTA (Palpites Iniciais)")
    print("-"*80)

    if initial_date is None:
        initial_date = historical_data['timestamp'].min()

    print(f"\n3.1. Varredura de t_1/2 para minimizar MSE")
    print(f"     Data inicial: {initial_date}")
    print(f"     Horizonte: {horizon_hours}h")
    print(f"     Candidatos t_1/2: 15 a 120 min (passo 15)")

    open_loop_results = calibrator.calibrate_from_timeseries(
        start_time=initial_date,
        horizon_hours=horizon_hours,
        t_half_range=(15.0, 120.0, 15.0),
        pv_forecaster=pv_forecaster,
        load_forecaster=load_forecaster
    )

    print(f"\n3.2. Estimativa de σ a partir de MSE e κ")
    print(f"     σ² = MSE × (1 - κ²)")

    print(f"\nResultados malha-aberta:")
    print(f"  t_1/2 inicial: {open_loop_results['t_half_minutes']:.1f} min")
    print(f"  σ inicial:     {open_loop_results['sigma_kw']:.3f} kW")
    print(f"  MSE:           {open_loop_results['best_mse']:.4f}")

    # Se não houver cost_function, retornar só resultados malha-aberta
    if cost_function is None:
        print("\nAviso: cost_function não fornecida, pulando calibração malha-fechada")
        return {
            'open_loop': open_loop_results,
            'recommended': {
                't_half_minutes': open_loop_results['t_half_minutes'],
                'sigma_kw': open_loop_results['sigma_kw']
            }
        }

    # ==================== PASSO 4: CALIBRAÇÃO MALHA-FECHADA ====================
    print("\n" + "-"*80)
    print("PASSO 4: CALIBRAÇÃO MALHA-FECHADA (Afinação por Desempenho)")
    print("-"*80)

    print("\n4.1. Seleção de benchmark (12 dias, um aleatório por mês)")
    benchmark_days = calibrator.select_benchmark_days(n_days=12, seed=42)

    # Definir grade em torno dos palpites iniciais
    t_half_init = open_loop_results['t_half_minutes']
    sigma_init = open_loop_results['sigma_kw']

    # Grade sugerida no protocolo: t_1/2 ∈ {35, 45, 60, 80}, σ ∈ [0.35, 0.70]
    # Adaptar para os valores encontrados
    t_half_candidates = [
        max(15.0, t_half_init - 15),
        t_half_init,
        t_half_init + 15,
        t_half_init + 30
    ]

    sigma_candidates = np.arange(
        max(0.1, sigma_init - 0.15),
        sigma_init + 0.20,
        0.05
    ).tolist()

    print(f"\n4.2. Busca em grade:")
    print(f"     t_1/2 candidatos: {[f'{t:.0f}' for t in t_half_candidates]} min")
    print(f"     σ candidatos: {[f'{s:.2f}' for s in sigma_candidates]} kW")
    print(f"\n     Configuração:")
    print(f"       - Δt = {dt_minutes} min")
    print(f"       - Recalcular políticas a cada 6h")
    print(f"       - Horizonte N = 36 (9h)")
    print(f"       - Discretizações: n_x = 150, n_R = 50")
    print(f"       - Tarifas: 28 c€/kWh compra, 12.3 c€/kWh injeção")

    closed_loop_results = calibrator.grid_search_closed_loop(
        t_half_candidates=t_half_candidates,
        sigma_candidates=sigma_candidates,
        benchmark_days=benchmark_days,
        cost_function=cost_function,
        tariff_buy_eur_kwh=0.28,
        tariff_sell_eur_kwh=0.123,
        verbose=True
    )

    print("\n4.3. Melhor par (menor custo agregado):")
    print(f"     t_1/2 final: {closed_loop_results['t_half_minutes']:.0f} min")
    print(f"     σ final:     {closed_loop_results['sigma_kw']:.2f} kW")

    # ==================== RESUMO FINAL ====================
    print("\n" + "="*80)
    print("RESUMO DA CALIBRAÇÃO")
    print("="*80)
    print("\nMalha-aberta (palpites iniciais):")
    print(f"  t_1/2 = {open_loop_results['t_half_minutes']:.1f} min")
    print(f"  σ     = {open_loop_results['sigma_kw']:.3f} kW")
    print("\nMalha-fechada (afinados por desempenho):")
    print(f"  t_1/2 = {closed_loop_results['t_half_minutes']:.0f} min")
    print(f"  σ     = {closed_loop_results['sigma_kw']:.2f} kW")
    print(f"  Custo = {closed_loop_results['cost']:.2f} €")

    if 'self_sufficiency' in closed_loop_results['metrics']:
        print(f"  Autossuficiência = {closed_loop_results['metrics']['self_sufficiency']:.1f}%")
    if 'curtailment' in closed_loop_results['metrics']:
        print(f"  Curtailment      = {closed_loop_results['metrics']['curtailment']:.2f} kWh")

    print("\n" + "="*80)
    print("PARÂMETROS RECOMENDADOS (malha-fechada):")
    print(f"  t_1/2 = {closed_loop_results['t_half_minutes']:.0f} min")
    print(f"  σ     = {closed_loop_results['sigma_kw']:.2f} kW")
    print("="*80 + "\n")

    return {
        'open_loop': open_loop_results,
        'closed_loop': closed_loop_results,
        'recommended': {
            't_half_minutes': closed_loop_results['t_half_minutes'],
            'sigma_kw': closed_loop_results['sigma_kw']
        }
    }


if __name__ == "__main__":
    """
    Calibração usando dados reais de carga e produção FV.

    Uso:
        python -m src.controllers.sdp_calibration
        python -m src.controllers.sdp_calibration --start-date 2024-01-15
        python -m src.controllers.sdp_calibration --full-protocol
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibração de parâmetros SDP (t_1/2 e σ).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Calibração malha-aberta apenas:
  python -m src.controllers.sdp_calibration

  # Protocolo completo (malha-aberta + malha-fechada):
  python -m src.controllers.sdp_calibration --full-protocol

  # Especificar data inicial:
  python -m src.controllers.sdp_calibration --start-date 2024-01-15 --full-protocol
        """
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Data de início para a calibração (formato: YYYY-MM-DD)"
    )
    parser.add_argument(
        "--full-protocol",
        action="store_true",
        help="Executar protocolo completo (malha-aberta + malha-fechada com 12 dias benchmark)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=18,
        help="Horizonte de previsão em horas (default: 18h, conforme protocolo)"
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("CALIBRAÇÃO SDP COM DADOS REAIS")
    print("="*80 + "\n")

    # Carregar configuração
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Carregar dados reais
    print("Carregando dados reais...")

    from src.components.house import House
    from src.components.solar import SolarPanel

    # Dados de carga
    load_file = config['house']['data_file']
    print(f"  Carregando carga de: {load_file}")
    house = House(load_file)

    # Dados de PV
    solar_config = config['solar']
    pv_file = solar_config['data_file']
    scale_factor = solar_config.get('scale_factor', 1.0)
    print(f"  Carregando PV de: {pv_file}")
    print(f"  Scale factor FV: {scale_factor} ({scale_factor*100:.0f}%)")
    solar = SolarPanel(capacity_kw=solar_config['capacity_kw'],
                      data_file=pv_file,
                      scale_factor=scale_factor)

    # Preparar dados de carga
    load_data = house.consumption_data.copy()
    load_data.reset_index(inplace=True)
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
    pv_data.reset_index(inplace=True)
    if 'timestamp' not in pv_data.columns:
        pv_data['timestamp'] = pd.to_datetime(
            pv_data.apply(
                lambda row: f"2024-{int(row['month']):02d}-{int(row['day']):02d} "
                           f"{int(row['hour']):02d}:{int(row['minute']):02d}:00",
                axis=1
            )
        )

    # Converter produção para kW e aplicar scale_factor
    prod_col = 'pv_1' if 'pv_1' in pv_data.columns else 'production'
    pv_data['P_pv'] = pv_data[prod_col] / 1000.0 * solar.scale_factor  # W para kW com scale
    print(f"  DEBUG - Exemplo produção PV: antes={pv_data[prod_col].iloc[1000]:.1f}W, depois={pv_data['P_pv'].iloc[1000]:.3f}kW (scale={solar.scale_factor})")
    pv_data = pv_data[['timestamp', 'P_pv']]

    # Merge dos dados
    historical_data = pd.merge(
        pv_data, load_data, on='timestamp', how='inner'
    )

    print(f"  Dados carregados: {len(historical_data)} pontos")
    print(f"  DEBUG - Stats PV: mean={historical_data['P_pv'].mean():.3f}kW, max={historical_data['P_pv'].max():.3f}kW")
    print(f"  DEBUG - Stats Load: mean={historical_data['P_load'].mean():.3f}kW, max={historical_data['P_load'].max():.3f}kW")

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

    # Escolher janela de calibração
    if args.start_date:
        try:
            calib_start = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(f"Formato de data inválido: {args.start_date}. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        calib_start = historical_data['timestamp'].min().replace(hour=0, minute=0, second=0, microsecond=0)

    # ========== ESCOLHER MODO: PROTOCOLO COMPLETO OU APENAS MALHA-ABERTA ==========
    if args.full_protocol:
        print("\n" + "="*80)
        print("MODO: PROTOCOLO COMPLETO")
        print("="*80)
        print("\nAVISO: Protocolo completo requer função de custo para malha-fechada.")
        print("       Por ora, executando apenas PASSO 3 (malha-aberta).")
        print("       Para malha-fechada, implemente cost_function em seu código.\n")

        # Usar full_calibration_protocol sem cost_function (só malha-aberta)
        results = full_calibration_protocol(
            historical_data=historical_data,
            pv_forecaster=pv_forecaster,
            load_forecaster=load_forecaster,
            cost_function=None,  # TODO: Implementar simulação em malha-fechada
            dt_minutes=dt_minutes,
            horizon_hours=args.horizon,
            initial_date=calib_start
        )

        # Extrair parâmetros recomendados
        final_t_half = results['recommended']['t_half_minutes']
        final_sigma = results['recommended']['sigma_kw']

    else:
        # ========== CALIBRAÇÃO MALHA-ABERTA (MODO PADRÃO) ==========
        print("\n" + "="*80)
        print("MODO: CALIBRAÇÃO MALHA-ABERTA (Passo 3 do protocolo)")
        print("="*80)
        print("\nUse --full-protocol para executar Passo 3 + Passo 4 (12 dias benchmark)")

        print(f"\n  Janela de calibração: {calib_start}")
        print(f"  Horizonte: {args.horizon}h")
        print(f"  Testando t_1/2: 15 a 120 min (passo 15 min)")

        # Calibração malha-aberta
        results = calibrator.calibrate_from_timeseries(
            start_time=calib_start,
            horizon_hours=args.horizon,
            t_half_range=(15.0, 120.0, 15.0),
            pv_forecaster=pv_forecaster,
            load_forecaster=load_forecaster
        )

        print(f"\n  Resultados MSE por t_1/2:")
        for t_half, mse in sorted(results['mse_results'].items()):
            marker = " <- MELHOR" if t_half == results['t_half_minutes'] else ""
            print(f"    t_1/2 = {t_half:5.1f} min  ->  MSE = {mse:.4f}{marker}")

        # Validação em dia diferente
        print("\n=== Validação em Dia Diferente ===")
        val_start = calib_start + timedelta(days=17)
        print(f"  Dia de validação: {val_start.date()}")

        val_results = calibrator.calibrate_from_timeseries(
            start_time=val_start,
            horizon_hours=args.horizon,
            t_half_range=(15.0, 120.0, 15.0),
            pv_forecaster=pv_forecaster,
            load_forecaster=load_forecaster
        )

        print(f"\n  Resultados MSE validação por t_1/2:")
        for t_half, mse in sorted(val_results['mse_results'].items()):
            marker = " <- MELHOR" if t_half == val_results['t_half_minutes'] else ""
            print(f"    t_1/2 = {t_half:5.1f} min  ->  MSE = {mse:.4f}{marker}")

        print(f"\n  t_1/2 validação: {val_results['t_half_minutes']:.1f} min")
        print(f"  σ validação: {val_results['sigma_kw']:.3f} kW")
        print(f"  MSE validação: {val_results['best_mse']:.4f}")

        # Valores médios
        final_t_half = (results['t_half_minutes'] + val_results['t_half_minutes']) / 2
        final_sigma = (results['sigma_kw'] + val_results['sigma_kw']) / 2

        print(f"\n=== Cálculo da Média ===")
        print(f"  Calibração: t_1/2 = {results['t_half_minutes']:.1f} min, σ = {results['sigma_kw']:.3f} kW")
        print(f"  Validação:  t_1/2 = {val_results['t_half_minutes']:.1f} min, σ = {val_results['sigma_kw']:.3f} kW")
        print(f"  Média:      t_1/2 = {final_t_half:.1f} min, σ = {final_sigma:.3f} kW")

    # ========== RESUMO FINAL ==========
    print("\n" + "="*80)
    print("PARÂMETROS RECOMENDADOS PARA SDP")
    print("="*80)
    print(f"\n  t_1/2 (meia-vida):  {final_t_half:.1f} minutos")
    print(f"  σ (ruído):          {final_sigma:.3f} kW")
    print("\n" + "="*80)
    print("Para usar estes parâmetros:")
    print("  1. Atualize config.yaml:")
    print(f"       half_life_minutes: {final_t_half:.0f}")
    print(f"       sigma_residual_kw: {final_sigma:.3f}")
    print("  2. Ou use diretamente:")
    print(f"       SDPParams(t_half_minutes={final_t_half:.0f}, sigma_R={final_sigma:.3f})")
    print("="*80 + "\n")
