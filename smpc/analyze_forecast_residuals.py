"""
Script para Análise de Previsões e Veracidade do Modelo de Resíduo

Este script analisa:
1. Acurácia das previsões de P_pv e P_load
2. Distribuição dos resíduos (R = P_pv - P_load)
3. Veracidade do modelo de meia-vida (kappa)
4. Adequação do sigma (desvio padrão do ruído)
5. Autocorrelação temporal dos resíduos

O objetivo é validar se os parâmetros do modelo de resíduo estão corretos
e se as previsões são confiáveis para o controlador SDP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import yaml
from typing import Tuple, Dict
from scipy import stats
from scipy.optimize import curve_fit

# Import components
from src.components.solar import SolarPanel
from src.components.house import House
from src.forecasters.solar_forecaster import SolarForecaster
from src.forecasters.load_forecaster import LoadForecaster
from src.forecasters.profile_persistence_forecaster import ProfilePersistenceForecaster


def load_config(config_path: str = "config.yaml") -> dict:
    """Carregar configuração do YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_components(config: dict) -> Tuple[SolarPanel, House]:
    """Criar componentes solar e house."""
    solar_config = config['solar']
    # Calculate scale factor from solar capacity and scale factor per 1kWp
    solar_capacity_kwp = solar_config.get('solar_capacity_kwp', 1.0)
    scale_factor = solar_capacity_kwp * solar_config.get('scale_factor_for_1kwp', 1.0)

    solar = SolarPanel(
        capacity_kw=solar_capacity_kwp,
        data_file=solar_config['data_file'],
        scale_factor=scale_factor
    )

    house = House(
        data_file=config['house']['data_file']
    )

    return solar, house


def collect_forecast_data(solar: SolarPanel,
                          house: House,
                          start_date: datetime,
                          end_date: datetime,
                          dt_minutes: int = 15,
                          forecast_horizon_hours: int = 24,
                          n_weeks: int = 3) -> pd.DataFrame:
    """
    Coletar dados de previsão e valores reais.

    Para cada timestep:
    - Faz previsão para o horizonte futuro
    - Registra valores reais
    - Calcula erros de previsão
    """
    # Criar forecasters simples
    solar_forecaster = SolarForecaster(solar)
    load_forecaster = LoadForecaster(house)

    # Criar dados históricos para profile persistence
    print("  Criando dados históricos para profile persistence...")
    historical_days = n_weeks * 7 + 7  # N semanas + 1 semana extra
    historical_start = start_date - timedelta(days=historical_days)

    # Gerar dados históricos
    timestamps = pd.date_range(
        start=historical_start,
        end=start_date,
        freq=f'{dt_minutes}min'
    )

    pv_values = []
    load_values = []
    for ts in timestamps:
        pv_values.append(solar.get_production(ts))
        load_values.append(house.get_consumption(ts))

    historical_data = pd.DataFrame({
        'timestamp': timestamps,
        'P_pv': pv_values,
        'P_load': load_values
    })

    # Criar forecasters de perfil de persistência (usado no SDP)
    pv_forecaster_profile = ProfilePersistenceForecaster(
        n_weeks=n_weeks,
        dt_minutes=dt_minutes
    )
    pv_data = historical_data[['timestamp', 'P_pv']].copy()
    pv_forecaster_profile.set_data(pv_data, value_column='P_pv')

    load_forecaster_profile = ProfilePersistenceForecaster(
        n_weeks=n_weeks,
        dt_minutes=dt_minutes
    )
    load_data = historical_data[['timestamp', 'P_load']].copy()
    load_forecaster_profile.set_data(load_data, value_column='P_load')

    results = []
    current_time = start_date
    dt = timedelta(minutes=dt_minutes)
    n_forecast_steps = int(forecast_horizon_hours * 60 / dt_minutes)

    print(f"Coletando dados de previsão de {start_date} até {end_date}...")
    print(f"  Horizonte de previsão: {forecast_horizon_hours}h ({n_forecast_steps} steps)")

    step_count = 0
    while current_time < end_date:
        if step_count % 96 == 0:  # A cada 24h
            progress = (current_time - start_date) / (end_date - start_date) * 100
            print(f"  Progresso: {progress:.1f}% - {current_time.strftime('%Y-%m-%d')}")

        # Valores reais no timestep atual
        P_pv_real = solar.get_production(current_time)
        P_load_real = house.get_consumption(current_time)
        R_real = P_pv_real - P_load_real

        # Previsões usando profile persistence (método usado no SDP)
        # Nota: set_data() renomeia as colunas para 'value', então usamos 'value' aqui
        P_pv_forecast = pv_forecaster_profile.get_forecast(
            start_time=current_time,
            n_steps=n_forecast_steps,
            value_column='value'
        )

        P_load_forecast = load_forecaster_profile.get_forecast(
            start_time=current_time,
            n_steps=n_forecast_steps,
            value_column='value'
        )

        # Calcular R_bar (previsão do resíduo)
        R_bar = P_pv_forecast - P_load_forecast

        # Registrar resultados (1 step ahead)
        result = {
            'timestamp': current_time,
            'hour': current_time.hour,
            'minute': current_time.minute,
            'day_of_week': current_time.weekday(),

            # Valores reais
            'P_pv_real': P_pv_real,
            'P_load_real': P_load_real,
            'R_real': R_real,

            # Previsões profile persistence (1 step ahead)
            'P_pv_forecast': P_pv_forecast[0] if len(P_pv_forecast) > 0 else 0,
            'P_load_forecast': P_load_forecast[0] if len(P_load_forecast) > 0 else 0,
            'R_bar': R_bar[0] if len(R_bar) > 0 else 0,

            # Erros de previsão
            'error_pv': P_pv_real - (P_pv_forecast[0] if len(P_pv_forecast) > 0 else 0),
            'error_load': P_load_real - (P_load_forecast[0] if len(P_load_forecast) > 0 else 0),
            'error_R': R_real - (R_bar[0] if len(R_bar) > 0 else 0),
        }

        results.append(result)

        # Atualizar forecasters com dados reais observados
        update_forecaster(pv_forecaster_profile, current_time, P_pv_real)
        update_forecaster(load_forecaster_profile, current_time, P_load_real)

        current_time += dt
        step_count += 1

    print(f"✓ Coletados {len(results)} timesteps de dados")
    return pd.DataFrame(results)


def analyze_forecast_errors(df: pd.DataFrame) -> Dict:
    """
    Analisar erros de previsão.

    Calcula métricas de erro:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - Bias (viés sistemático)
    """
    print("\n" + "="*70)
    print("ANÁLISE DE ERROS DE PREVISÃO")
    print("="*70)
    print("\n(Usando método Profile Persistence - mesmo usado no SDP)")

    # Erros de P_pv
    error_pv = df['error_pv'].values
    mae_pv = np.mean(np.abs(error_pv))
    rmse_pv = np.sqrt(np.mean(error_pv**2))
    bias_pv = np.mean(error_pv)

    print(f"\nP_pv (Produção Solar):")
    print(f"  MAE:  {mae_pv:.3f} kW")
    print(f"  RMSE: {rmse_pv:.3f} kW")
    print(f"  Bias: {bias_pv:.3f} kW")

    # Erros de P_load
    error_load = df['error_load'].values
    mae_load = np.mean(np.abs(error_load))
    rmse_load = np.sqrt(np.mean(error_load**2))
    bias_load = np.mean(error_load)

    print(f"\nP_load (Consumo):")
    print(f"  MAE:  {mae_load:.3f} kW")
    print(f"  RMSE: {rmse_load:.3f} kW")
    print(f"  Bias: {bias_load:.3f} kW")

    # Erros de R (resíduo)
    error_R = df['error_R'].values
    mae_R = np.mean(np.abs(error_R))
    rmse_R = np.sqrt(np.mean(error_R**2))
    bias_R = np.mean(error_R)

    print(f"\nR (Resíduo = P_pv - P_load):")
    print(f"  MAE:  {mae_R:.3f} kW")
    print(f"  RMSE: {rmse_R:.3f} kW")
    print(f"  Bias: {bias_R:.3f} kW")

    metrics = {
        'mae_pv': mae_pv,
        'rmse_pv': rmse_pv,
        'bias_pv': bias_pv,
        'mae_load': mae_load,
        'rmse_load': rmse_load,
        'bias_load': bias_load,
        'mae_R': mae_R,
        'rmse_R': rmse_R,
        'bias_R': bias_R,
    }

    return metrics


def analyze_residual_distribution(df: pd.DataFrame) -> Dict:
    """
    Analisar distribuição estatística dos resíduos.

    Verifica:
    - Normalidade (teste de Shapiro-Wilk)
    - Média e desvio padrão
    - Skewness e kurtosis
    """
    print("\n" + "="*70)
    print("ANÁLISE DE DISTRIBUIÇÃO DOS RESÍDUOS")
    print("="*70)

    error_R = df['error_R'].values
    error_R_clean = error_R[~np.isnan(error_R)]  # Remover NaN

    # Estatísticas básicas
    mean = np.mean(error_R_clean)
    std = np.std(error_R_clean)
    median = np.median(error_R_clean)

    print(f"\nEstatísticas do erro de resíduo:")
    print(f"  Média:   {mean:.3f} kW")
    print(f"  Mediana: {median:.3f} kW")
    print(f"  Std Dev: {std:.3f} kW")
    print(f"  Min:     {np.min(error_R_clean):.3f} kW")
    print(f"  Max:     {np.max(error_R_clean):.3f} kW")

    # Teste de normalidade
    if len(error_R_clean) > 3:
        statistic, p_value = stats.shapiro(error_R_clean[:5000])  # Limitar tamanho para Shapiro
        print(f"\nTeste de Normalidade (Shapiro-Wilk):")
        print(f"  Estatística: {statistic:.4f}")
        print(f"  P-value:     {p_value:.4f}")
        if p_value < 0.05:
            print(f"  ⚠️ Rejeita H0: Os resíduos NÃO são normalmente distribuídos")
        else:
            print(f"  ✓ Aceita H0: Os resíduos são normalmente distribuídos")

    # Skewness e Kurtosis
    skewness = stats.skew(error_R_clean)
    kurtosis = stats.kurtosis(error_R_clean)
    print(f"\nForma da distribuição:")
    print(f"  Skewness: {skewness:.3f} {'(assimétrica à direita)' if skewness > 0 else '(assimétrica à esquerda)'}")
    print(f"  Kurtosis: {kurtosis:.3f} {'(caudas pesadas)' if kurtosis > 0 else '(caudas leves)'}")

    stats_dict = {
        'mean': mean,
        'std': std,
        'median': median,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }

    return stats_dict


def analyze_autocorrelation(df: pd.DataFrame, max_lag: int = 24) -> Dict:
    """
    Analisar autocorrelação temporal dos resíduos.

    Verifica se o modelo de meia-vida é adequado analisando
    a correlação entre resíduos em diferentes lags temporais.
    """
    print("\n" + "="*70)
    print("ANÁLISE DE AUTOCORRELAÇÃO DOS RESÍDUOS")
    print("="*70)

    error_R = df['error_R'].values
    error_R_clean = error_R[~np.isnan(error_R)]

    # Calcular autocorrelação
    autocorr = []
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = np.corrcoef(error_R_clean[:-lag], error_R_clean[lag:])[0, 1]
            autocorr.append(corr if not np.isnan(corr) else 0.0)

    autocorr = np.array(autocorr)

    print(f"\nAutocorrelação (lags em 15 minutos):")
    for i in [0, 1, 2, 4, 8, 16, 24]:
        if i <= max_lag:
            time_minutes = i * 15
            print(f"  Lag {i:2d} ({time_minutes:3d} min): {autocorr[i]:.3f}")

    # Estimar meia-vida a partir da autocorrelação
    # Encontrar quando a autocorrelação cai para ~0.5
    try:
        lag_half = np.where(autocorr < 0.5)[0][0] if any(autocorr < 0.5) else max_lag
        time_half_minutes = lag_half * 15
        print(f"\nMeia-vida estimada: ~{time_half_minutes} minutos (lag={lag_half})")
    except:
        time_half_minutes = None
        print(f"\nMeia-vida estimada: Não encontrada (autocorr não cai abaixo de 0.5)")

    autocorr_dict = {
        'autocorr': autocorr,
        'time_half_minutes': time_half_minutes,
    }

    return autocorr_dict


def fit_kappa_from_data(df: pd.DataFrame, dt_minutes: int = 15) -> Dict:
    """
    Estimar o parâmetro kappa (meia-vida) a partir dos dados.

    Modelo: R_{k+1} = R_bar_{k+1} + kappa * (R_k - R_bar_k) + noise

    Vamos fazer uma regressão para estimar kappa.
    """
    print("\n" + "="*70)
    print("ESTIMATIVA DO PARÂMETRO KAPPA (MEIA-VIDA)")
    print("="*70)

    # Obter séries temporais
    R_real = df['R_real'].values
    R_bar = df['R_bar'].values

    # Construir dados para regressão
    # y = R_{k+1} - R_bar_{k+1}
    # x = R_k - R_bar_k
    # Modelo: y = kappa * x + noise

    y = (R_real[1:] - R_bar[1:])  # R_{k+1} - R_bar_{k+1}
    x = (R_real[:-1] - R_bar[:-1])  # R_k - R_bar_k

    # Remover NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) > 10:
        # Regressão linear: y = kappa * x + epsilon
        # kappa = cov(x, y) / var(x)
        kappa_estimated = np.cov(x_clean, y_clean)[0, 1] / np.var(x_clean)

        # Calcular meia-vida correspondente
        if 0 < kappa_estimated < 1:
            t_half_minutes = -dt_minutes * np.log(2) / np.log(kappa_estimated)
        else:
            t_half_minutes = np.inf if kappa_estimated >= 1 else 0

        # Calcular R² da regressão
        y_pred = kappa_estimated * x_clean
        ss_res = np.sum((y_clean - y_pred)**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calcular sigma do resíduo
        residuals = y_clean - y_pred
        sigma_estimated = np.std(residuals)

        print(f"\nResultados da regressão:")
        print(f"  Kappa estimado:     {kappa_estimated:.4f}")
        print(f"  Meia-vida estimada: {t_half_minutes:.1f} minutos")
        print(f"  R²:                 {r_squared:.4f}")
        print(f"  Sigma estimado:     {sigma_estimated:.3f} kW")

        kappa_dict = {
            'kappa': kappa_estimated,
            't_half_minutes': t_half_minutes,
            'r_squared': r_squared,
            'sigma': sigma_estimated,
        }
    else:
        print(f"  ⚠️ Dados insuficientes para estimar kappa")
        kappa_dict = None

    return kappa_dict


def update_forecaster(forecaster: ProfilePersistenceForecaster,
                      timestamp: datetime,
                      value: float):
    """
    Adicionar um novo ponto de dados ao forecaster e reconstruir o perfil.
    """
    # O forecaster internamente usa a coluna 'value'
    new_row = pd.DataFrame([{
        'timestamp': timestamp,
        'value': value
    }])
    new_row.set_index('timestamp', inplace=True)

    # Adicionar colunas de data/hora
    new_row['weekday'] = new_row.index.dayofweek
    new_row['hour'] = new_row.index.hour
    new_row['minute'] = new_row.index.minute

    # Concatenar com os dados existentes no forecaster
    # forecaster.data tem 'timestamp' como índice
    updated_data = pd.concat([forecaster.data, new_row])

    # Atualizar os dados internos e reconstruir o cache do perfil
    forecaster.data = updated_data
    forecaster._build_profile_cache()


def plot_forecast_analysis(df: pd.DataFrame, autocorr_dict: Dict,
                           output_dir: str = "results/forecast_analysis"):
    """
    Criar visualizações da análise de previsão.
    """
    print(f"\n{'='*70}")
    print("GERANDO GRÁFICOS DE ANÁLISE")
    print(f"{'='*70}\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- 1. Série temporal de previsões vs valores reais ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Pegar apenas os primeiros 7 dias para visualização
    n_days = 7
    n_points = n_days * 96
    df_plot = df.head(n_points)

    # P_pv
    axes[0].plot(df_plot['timestamp'], df_plot['P_pv_real'],
                 label='Real', linewidth=1.5, alpha=0.7)
    axes[0].plot(df_plot['timestamp'], df_plot['P_pv_forecast'],
                 label='Previsão', linewidth=1, alpha=0.7, linestyle='--')
    axes[0].set_ylabel('P_pv (kW)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Produção Solar: Real vs Previsão')

    # P_load
    axes[1].plot(df_plot['timestamp'], df_plot['P_load_real'],
                 label='Real', linewidth=1.5, alpha=0.7)
    axes[1].plot(df_plot['timestamp'], df_plot['P_load_forecast'],
                 label='Previsão', linewidth=1, alpha=0.7, linestyle='--')
    axes[1].set_ylabel('P_load (kW)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Consumo: Real vs Previsão')

    # R (resíduo)
    axes[2].plot(df_plot['timestamp'], df_plot['R_real'],
                 label='Real', linewidth=1.5, alpha=0.7)
    axes[2].plot(df_plot['timestamp'], df_plot['R_bar'],
                 label='Previsão', linewidth=1, alpha=0.7, linestyle='--')
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[2].set_ylabel('R (kW)')
    axes[2].set_xlabel('Tempo')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Resíduo (P_pv - P_load): Real vs Previsão')

    plt.tight_layout()
    filepath = Path(output_dir) / 'forecast_timeseries.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Salvo: {filepath}")
    plt.close()

    # --- 2. Distribuição dos erros ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    error_types = [
        ('error_pv', 'Erro P_pv (kW)'),
        ('error_load', 'Erro P_load (kW)'),
        ('error_R', 'Erro R (kW)'),
    ]

    for idx, (col, title) in enumerate(error_types):
        error_data = df[col].dropna().values

        # Histograma
        axes[0, idx].hist(error_data, bins=50, density=True, alpha=0.7, edgecolor='black')

        # Fit normal distribution
        mu, sigma = np.mean(error_data), np.std(error_data)
        x = np.linspace(error_data.min(), error_data.max(), 100)
        axes[0, idx].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                         label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')

        axes[0, idx].set_xlabel(title)
        axes[0, idx].set_ylabel('Densidade')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].set_title(f'Distribuição: {title}')

        # Q-Q plot
        stats.probplot(error_data, dist="norm", plot=axes[1, idx])
        axes[1, idx].set_title(f'Q-Q Plot: {title}')
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / 'error_distributions.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Salvo: {filepath}")
    plt.close()

    # --- 3. Autocorrelação ---
    fig, ax = plt.subplots(figsize=(10, 6))

    autocorr = autocorr_dict['autocorr']
    lags = np.arange(len(autocorr))
    time_lags = lags * 15  # minutos

    ax.bar(time_lags, autocorr, width=10, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Meia-vida (50%)')

    ax.set_xlabel('Lag (minutos)')
    ax.set_ylabel('Autocorrelação')
    ax.set_title('Autocorrelação dos Erros de Resíduo')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = Path(output_dir) / 'autocorrelation.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Salvo: {filepath}")
    plt.close()

    # --- 4. Scatter plots: Real vs Forecast ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    variables = [
        ('P_pv_real', 'P_pv_forecast', 'P_pv'),
        ('P_load_real', 'P_load_forecast', 'P_load'),
        ('R_real', 'R_bar', 'R'),
    ]

    for idx, (real_col, forecast_col, name) in enumerate(variables):
        real = df[real_col].values
        forecast = df[forecast_col].values

        # Scatter
        axes[idx].scatter(forecast, real, alpha=0.3, s=1)

        # Linha ideal (y=x)
        min_val = min(real.min(), forecast.min())
        max_val = max(real.max(), forecast.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')

        axes[idx].set_xlabel(f'{name} Previsto (kW)')
        axes[idx].set_ylabel(f'{name} Real (kW)')
        axes[idx].set_title(f'{name}: Real vs Previsto')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axis('equal')

    plt.tight_layout()
    filepath = Path(output_dir) / 'scatter_real_vs_forecast.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Salvo: {filepath}")
    plt.close()

    print(f"\n✓ Todos os gráficos salvos em: {output_dir}")


def compare_with_config(config: dict, kappa_dict: Dict):
    """
    Comparar parâmetros estimados com configuração atual.
    """
    print("\n" + "="*70)
    print("COMPARAÇÃO COM CONFIGURAÇÃO ATUAL")
    print("="*70)

    # Parâmetros do config
    config_t_half = config['controller']['sdp']['half_life_minutes']
    config_sigma = config['controller']['sdp']['sigma_residual_kw']

    # Calcular kappa do config
    dt_minutes = config['simulation']['timestep_minutes']
    config_kappa = 2.0 ** (-dt_minutes / config_t_half) if config_t_half > 0 else 0.0

    print(f"\nParâmetros no config.yaml:")
    print(f"  half_life_minutes:   {config_t_half} min")
    print(f"  sigma_residual_kw:   {config_sigma:.3f} kW")
    print(f"  kappa (calculado):   {config_kappa:.4f}")

    # Comparar com estimativas
    if kappa_dict is not None:
        estimated = kappa_dict

        print(f"\nParâmetros estimados dos dados:")
        print(f"  meia-vida estimada:  {estimated['t_half_minutes']:.1f} min")
        print(f"  sigma estimado:      {estimated['sigma']:.3f} kW")
        print(f"  kappa estimado:      {estimated['kappa']:.4f}")

        # Diferenças
        diff_t_half = estimated['t_half_minutes'] - config_t_half
        diff_sigma = estimated['sigma'] - config_sigma
        diff_kappa = estimated['kappa'] - config_kappa

        print(f"\nDiferenças (estimado - config):")
        print(f"  Δ meia-vida:  {diff_t_half:+.1f} min ({diff_t_half/config_t_half*100:+.1f}%)")
        print(f"  Δ sigma:      {diff_sigma:+.3f} kW ({diff_sigma/config_sigma*100:+.1f}%)")
        print(f"  Δ kappa:      {diff_kappa:+.4f} ({diff_kappa/config_kappa*100:+.1f}%)")

        # Recomendações
        print(f"\n{'='*70}")
        print("RECOMENDAÇÕES")
        print(f"{'='*70}")

        if abs(diff_t_half / config_t_half) > 0.5:
            print(f"⚠️  A meia-vida estimada difere significativamente do config!")
            print(f"    Considere ajustar 'half_life_minutes' para ~{estimated['t_half_minutes']:.0f}")
        else:
            print(f"✓  A meia-vida no config está razoável")

        if abs(diff_sigma / config_sigma) > 0.5:
            print(f"⚠️  O sigma estimado difere significativamente do config!")
            print(f"    Considere ajustar 'sigma_residual_kw' para ~{estimated['sigma']:.2f}")
        else:
            print(f"✓  O sigma no config está razoável")


def main():
    """Executar análise completa."""
    print("="*70)
    print("ANÁLISE DE PREVISÕES E MODELO DE RESÍDUO")
    print("="*70)

    # Carregar configuração
    config = load_config()
    print("\n✓ Configuração carregada")

    # Criar componentes
    solar, house = create_components(config)
    print("✓ Componentes criados")

    # Definir período de análise
    sim_config = config['simulation']
    start_date = datetime.strptime(sim_config['start_date'], "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(sim_config['end_date'], "%Y-%m-%d %H:%M:%S")
    dt_minutes = sim_config['timestep_minutes']

    # Limitar período para análise (usar 30 dias)
    analysis_days = 30
    analysis_end = start_date + timedelta(days=analysis_days)
    if analysis_end > end_date:
        analysis_end = end_date

    print(f"\nPeríodo de análise:")
    print(f"  Início: {start_date}")
    print(f"  Fim:    {analysis_end}")
    print(f"  Total:  {(analysis_end - start_date).days} dias")

    # Coletar dados de previsão
    n_weeks = config['controller']['sdp']['forecaster'].get('n_weeks', 3)
    df = collect_forecast_data(
        solar=solar,
        house=house,
        start_date=start_date,
        end_date=analysis_end,
        dt_minutes=dt_minutes,
        forecast_horizon_hours=24,
        n_weeks=n_weeks
    )

    # Salvar dados coletados
    output_dir = Path("results/forecast_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "forecast_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Dados salvos em: {csv_path}")

    # Análises
    metrics = analyze_forecast_errors(df)
    stats_dict = analyze_residual_distribution(df)
    autocorr_dict = analyze_autocorrelation(df, max_lag=24)
    kappa_dict = fit_kappa_from_data(df, dt_minutes=dt_minutes)

    # Comparar com config
    compare_with_config(config, kappa_dict)

    # Gerar gráficos
    plot_forecast_analysis(df, autocorr_dict, output_dir=str(output_dir))

    print(f"\n{'='*70}")
    print("ANÁLISE COMPLETA!")
    print(f"{'='*70}")
    print(f"\nResultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()