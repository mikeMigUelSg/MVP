"""
Stochastic Dynamic Programming (SDP) Controller for PV+Battery System

Implementa controle ótimo de bateria com PV considerando:
- Limite de injeção na rede
- Incerteza de previsão (modelo estocástico)
- Modelo AC simplificado (sem perdas de inversor, apenas bateria)
- Programação dinâmica estocástica com quadratura Gaussiana

Baseado no paper de SDP para sistemas PV+Bateria.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
from scipy.interpolate import interp1d, RectBivariateSpline
from dataclasses import dataclass


@dataclass
class SDPParams:
    """Parâmetros do SDP."""
    # Horizonte e passo temporal
    N: int = 36                     # Horizonte de planejamento (9h com dt=15min)
    dt_minutes: int = 15            # Passo temporal (minutos)

    # Discretização
    n_x: int = 150                  # Número de pontos SOC
    n_R: int = 50                   # Número de pontos Resíduo

    # Modelo de meia-vida
    t_half_minutes: float = 45.0    # Meia-vida do resíduo (minutos)
    sigma_R: float = 0.43           # Desvio padrão do ruído (kW)

    # Atualização de política
    policy_update_hours: int = 6    # Recalcular políticas a cada 6h

    # Custo terminal
    soc_target: float = 0.5         # SOC alvo no final (para penalização)
    terminal_cost_weight: float = 1.0  # Peso do custo terminal


class ResidualModel:
    """
    Modelo de resíduo R = P_pv - P_load com meia-vida.

    Implementa:
    - Modelo determinístico: R̂_{k+1} = R̄_{k+1} + κ(R̂_k - R̄_k)
    - Modelo estocástico: R_{k+1} = R̄_{k+1} + κ(R_k - R̄_k) + σ*n
    """

    def __init__(self, t_half_minutes: float, dt_minutes: float, sigma: float):
        """
        Args:
            t_half_minutes: Meia-vida do resíduo (minutos)
            dt_minutes: Passo temporal (minutos)
            sigma: Desvio padrão do ruído Gaussiano (kW)
        """
        self.t_half = t_half_minutes
        self.dt = dt_minutes
        self.sigma = sigma

        # Calcular κ
        if t_half_minutes > 0:
            self.kappa = 2.0 ** (-dt_minutes / t_half_minutes)
        else:
            self.kappa = 0.0  # Usar R̄ diretamente

    def deterministic_forecast(self,
                               R_bar: np.ndarray,
                               R_0: float) -> np.ndarray:
        """
        Previsão determinística do resíduo.

        Args:
            R_bar: Array com previsão base R̄ para cada passo
            R_0: Resíduo medido atual

        Returns:
            Array com R̂ para cada passo
        """
        N = len(R_bar)
        R_hat = np.zeros(N)
        R_hat[0] = R_bar[0] + self.kappa * (R_0 - R_bar[0])

        for k in range(1, N):
            R_hat[k] = R_bar[k] + self.kappa * (R_hat[k-1] - R_bar[k-1])

        return R_hat

    def stationary_variance(self) -> float:
        """
        Variância estacionária do processo estocástico.

        Var[R] = σ² / (1 - κ²)
        """
        if self.kappa >= 1.0:
            return np.inf
        return self.sigma**2 / (1.0 - self.kappa**2)

    def transition_mean(self, R_current: float, R_bar_next: float, R_bar_current: float) -> float:
        """
        Média da distribuição de transição.

        μ = R̄_{k+1} + κ(R_k - R̄_k)

        Args:
            R_current: Resíduo atual R_k
            R_bar_next: Previsão base do próximo passo R̄_{k+1}
            R_bar_current: Previsão base do passo atual R̄_k

        Returns:
            Média da distribuição de transição
        """
        return R_bar_next + self.kappa * (R_current - R_bar_current)


class PlantModel:
    """
    Modelo AC da planta PV+Bateria (simplificado, sem perdas de inversor).

    Inclui:
    - Dinâmica do SOC com perdas da bateria
    - Curtailment (limite de injeção)
    - Potência da rede
    """

    def __init__(self,
                 C_bat: float,           # Capacidade bateria (kWh)
                 P_nom: float,           # Potência nominal inversor (kW)
                 P_lim: float,           # Limite de injeção (kW)
                 eta_charge: float,      # Eficiência carga
                 eta_discharge: float,   # Eficiência descarga
                 dt_minutes: float,      # Passo temporal (min)
                 soc_min: float = 0.1,   # SOC mínimo
                 soc_max: float = 0.9):  # SOC máximo
        """
        Args:
            C_bat: Capacidade da bateria (kWh)
            P_nom: Potência nominal do inversor (kW)
            P_lim: Limite de injeção na rede (kW)
            eta_charge: Eficiência de carga (0-1)
            eta_discharge: Eficiência de descarga (0-1)
            dt_minutes: Passo temporal (minutos)
            soc_min: SOC mínimo permitido
            soc_max: SOC máximo permitido
        """
        self.C_bat = C_bat
        self.P_nom = P_nom
        self.P_lim = P_lim
        self.eta_c = eta_charge
        self.eta_d = eta_discharge
        self.dt_hours = dt_minutes / 60.0
        self.soc_min = soc_min
        self.soc_max = soc_max

    def p_eff(self, u: float) -> float:
        """
        Potência efetiva considerando perdas da bateria.

        Args:
            u: Ação (kW AC), <0 carrega, >0 descarrega

        Returns:
            Potência efetiva (kW)
        """
        if u < 0:  # Carregando
            return u * self.eta_c
        elif u > 0:  # Descarregando
            return u / self.eta_d
        else:
            return 0.0

    def next_soc(self, x: float, u: float) -> float:
        """
        Próximo SOC após aplicar ação u.

        x_{k+1} = x_k - (p_eff(u) * Δt) / C_bat

        Args:
            x: SOC atual (0-1)
            u: Ação (kW AC)

        Returns:
            Próximo SOC (0-1)
        """
        p_eff = self.p_eff(u)
        x_next = x - (p_eff * self.dt_hours) / self.C_bat
        return np.clip(x_next, self.soc_min, self.soc_max)

    def curtailment(self, u: float, R: float) -> float:
        """
        Curtailment devido ao limite de injeção.

        φ(u,R) = max(0, u + R - P_lim)

        Args:
            u: Ação da bateria (kW AC)
            R: Resíduo P_pv - P_load (kW)

        Returns:
            Curtailment (kW)
        """
        return max(0.0, u + R - self.P_lim)

    def grid_power(self, u: float, R: float) -> float:
        """
        Potência da rede.

        P_g = -u - R + φ(u,R)

        onde:
        - u > 0: descarrega bateria (fornece à rede/carga)
        - u < 0: carrega bateria (consome da rede/solar)
        - R > 0: excesso solar
        - R < 0: déficit (carga > solar)

        P_g > 0: importação da rede
        P_g < 0: exportação para rede

        Args:
            u: Ação da bateria (kW AC)
            R: Resíduo P_pv - P_load (kW)

        Returns:
            Potência da rede (kW)
        """
        phi = self.curtailment(u, R)
        return -u - R + phi

    def feasible_actions(self,
                        x: float,
                        x_grid: np.ndarray) -> np.ndarray:
        """
        Ações viáveis para um dado SOC.

        Gera ações que transitam para outros estados na grelha,
        respeitando limites do inversor e SOC.

        Args:
            x: SOC atual
            x_grid: Grelha de SOC

        Returns:
            Array de ações viáveis (kW AC)
        """
        actions = []

        # Para cada possível SOC futuro na grelha
        for x_next in x_grid:
            # Calcular ação necessária
            # x_next = x - (p_eff(u) * dt) / C_bat
            # p_eff(u) = (x - x_next) * C_bat / dt

            p_eff_needed = (x - x_next) * self.C_bat / self.dt_hours

            # Converter para ação u
            if p_eff_needed < 0:  # Precisa carregar
                u = p_eff_needed / self.eta_c
            elif p_eff_needed > 0:  # Precisa descarregar
                u = p_eff_needed * self.eta_d
            else:
                u = 0.0

            # Verificar se ação está dentro dos limites
            if abs(u) <= self.P_nom:
                actions.append(u)

        return np.array(actions)


class SDPController:
    """
    Controlador SDP para sistema PV+Bateria.

    Resolve o problema de otimização estocástica usando:
    - Discretização de estados (SOC e Resíduo)
    - Backward DP com quadratura Gaussiana de 5 pontos
    - Interpolação bilinear para execução em tempo real
    """

    def __init__(self,
                 params: SDPParams,
                 plant: PlantModel,
                 residual_model: ResidualModel,
                 c_s: float,  # Tarifa compra (€/kWh)
                 c_f: float):  # Tarifa venda (€/kWh)
        """
        Args:
            params: Parâmetros do SDP
            plant: Modelo da planta
            residual_model: Modelo do resíduo
            c_s: Tarifa de compra (€/kWh)
            c_f: Tarifa de venda (€/kWh)
        """
        self.params = params
        self.plant = plant
        self.residual_model = residual_model
        self.c_s = c_s
        self.c_f = c_f

        # Grelhas de discretização
        self.X_grid = None  # Grelha de SOC
        self.R_grids = None  # Grelhas de resíduo (uma por passo k)

        # Política ótima
        self.policy = None  # policy[k][i, j] = u*(x_i, R_{k,j})
        self.value_function = None  # J[k][i, j]

        # Cache para evitar recálculo
        self.last_policy_update = None
        self.R_bar_forecast = None  # Previsão base R̄

        # Nome do controlador
        self.name = "SDP"

    def stage_cost(self, u: float, R: float) -> float:
        """
        Custo por etapa.

        g(u, R) = (P_g^+ * c_s - P_g^- * c_f) * Δt

        Args:
            u: Ação da bateria (kW)
            R: Resíduo (kW)

        Returns:
            Custo (€)
        """
        P_g = self.plant.grid_power(u, R)

        # P_g > 0: compra da rede
        P_g_plus = max(0.0, P_g)

        # P_g < 0: venda para rede
        P_g_minus = max(0.0, -P_g)

        cost = (P_g_plus * self.c_s - P_g_minus * self.c_f) * self.plant.dt_hours
        return cost

    def terminal_cost(self, x: float) -> float:
        """
        Custo terminal.

        Penaliza desvio do SOC alvo.

        Args:
            x: SOC final

        Returns:
            Custo terminal (€)
        """
        x_target = self.params.soc_target
        weight = self.params.terminal_cost_weight
        return weight * abs(x - x_target) * self.plant.C_bat * self.c_s

    def create_grids(self, R_bar: np.ndarray):
        """
        Criar grelhas de discretização.

        Args:
            R_bar: Previsão base R̄ para cada passo k (tamanho N+1)
        """
        N = self.params.N
        n_x = self.params.n_x
        n_R = self.params.n_R

        # Grelha de SOC (uniforme em [soc_min, soc_max])
        self.X_grid = np.linspace(self.plant.soc_min, self.plant.soc_max, n_x)

        # Grelhas de resíduo (uma por passo k, incluindo k=N)
        # Cada grelha centrada em R̄_k com cobertura de ±3σ
        # Precisamos de N+1 grelhas (k=0 a k=N) para calcular transições
        self.R_grids = []
        sigma = self.residual_model.sigma

        for k in range(N + 1):
            R_center = R_bar[k]
            R_min = R_center - 3 * sigma
            R_max = R_center + 3 * sigma
            R_grid_k = np.linspace(R_min, R_max, n_R)
            self.R_grids.append(R_grid_k)

    def gaussian_quadrature_5pt(self,
                                 rho: float,
                                 sigma: float,
                                 R_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quadratura Gaussiana de 5 pontos.

        Avalia em pontos ρ + m*σ com m ∈ {-2, -1, 0, 1, 2}
        com pesos proporcionais a exp(-m²/2).

        Args:
            rho: Média da distribuição
            sigma: Desvio padrão
            R_grid: Grelha de resíduo para verificar limites

        Returns:
            (pontos, pesos) para integração
        """
        m_values = np.array([-2, -1, 0, 1, 2])
        points = rho + m_values * sigma

        # Pesos sem normalizar
        weights_unnorm = np.exp(-0.5 * m_values**2)

        # Filtrar pontos fora do intervalo
        R_min, R_max = R_grid.min(), R_grid.max()
        valid_mask = (points >= R_min) & (points <= R_max)

        valid_points = points[valid_mask]
        valid_weights = weights_unnorm[valid_mask]

        # Normalizar pesos
        if len(valid_weights) > 0:
            valid_weights = valid_weights / valid_weights.sum()
        else:
            # Todos os pontos fora do intervalo - usar apenas o centro
            valid_points = np.array([np.clip(rho, R_min, R_max)])
            valid_weights = np.array([1.0])

        return valid_points, valid_weights

    def solve_sdp(self, R_bar: np.ndarray, verbose: bool = False):
        """
        Resolver SDP usando backward DP com quadratura.

        Args:
            R_bar: Previsão base R̄ para cada passo k (tamanho N+1)
            verbose: Imprimir progresso
        """
        N = self.params.N
        n_x = len(self.X_grid)

        # Inicializar value function e política
        self.value_function = [None] * (N + 1)
        self.policy = [None] * N

        # Condição terminal (k=N)
        n_R_N = len(self.R_grids[N])
        self.value_function[N] = np.zeros((n_x, n_R_N))
        for i in range(n_x):
            terminal_val = self.terminal_cost(self.X_grid[i])
            self.value_function[N][i, :] = terminal_val

        # Backward DP
        for k in range(N - 1, -1, -1):
            if verbose and k % 10 == 0:
                print(f"  Solving step {k}/{N}...")

            n_R_k = len(self.R_grids[k])
            J_k = np.zeros((n_x, n_R_k))
            mu_k = np.zeros((n_x, n_R_k))

            for i in range(n_x):
                x_i = self.X_grid[i]

                # Ações viáveis
                U_i = self.plant.feasible_actions(x_i, self.X_grid)

                for j in range(n_R_k):
                    R_kj = self.R_grids[k][j]

                    # Minimização sobre ações
                    min_cost = np.inf
                    best_u = 0.0

                    for u in U_i:
                        # Custo imediato
                        g = self.stage_cost(u, R_kj)

                        # Próximo estado
                        x_next = self.plant.next_soc(x_i, u)

                        # Esperança do custo futuro
                        E_J = self._expected_future_cost(
                            x_next, R_kj, k, R_bar
                        )

                        total_cost = g + E_J

                        if total_cost < min_cost:
                            min_cost = total_cost
                            best_u = u

                    J_k[i, j] = min_cost
                    mu_k[i, j] = best_u

            self.value_function[k] = J_k
            self.policy[k] = mu_k

        if verbose:
            print("  SDP solved!")

    def _expected_future_cost(self,
                             x_next: float,
                             R_k: float,
                             k: int,
                             R_bar: np.ndarray) -> float:
        """
        Calcular esperança do custo futuro usando quadratura.

        E[J_{k+1}(x', R_{k+1})] usando quadratura de 5 pontos.

        Args:
            x_next: Próximo SOC
            R_k: Resíduo atual
            k: Passo atual
            R_bar: Previsão base

        Returns:
            Esperança do custo futuro
        """
        # Distribuição de transição
        rho = self.residual_model.transition_mean(
            R_k, R_bar[k+1], R_bar[k]
        )
        sigma = self.residual_model.sigma

        # Quadratura de 5 pontos
        points, weights = self.gaussian_quadrature_5pt(
            rho, sigma, self.R_grids[k+1]
        )

        # Avaliar J_{k+1} em cada ponto
        E_J = 0.0
        for R_next, w in zip(points, weights):
            J_val = self._interpolate_value_function(x_next, R_next, k+1)
            E_J += w * J_val

        return E_J

    def _interpolate_value_function(self,
                                    x: float,
                                    R: float,
                                    k: int) -> float:
        """
        Interpolar value function J_k(x, R).

        Usa interpolação bilinear.

        Args:
            x: SOC
            R: Resíduo
            k: Passo

        Returns:
            J_k(x, R) interpolado
        """
        # Encontrar índices vizinhos em X_grid
        i_x = np.searchsorted(self.X_grid, x)
        i_x = np.clip(i_x, 1, len(self.X_grid) - 1)
        i_x_low = i_x - 1
        i_x_high = i_x

        x_low = self.X_grid[i_x_low]
        x_high = self.X_grid[i_x_high]

        # Encontrar índices vizinhos em R_grids[k]
        R_grid_k = self.R_grids[k]
        i_R = np.searchsorted(R_grid_k, R)
        i_R = np.clip(i_R, 1, len(R_grid_k) - 1)
        i_R_low = i_R - 1
        i_R_high = i_R

        R_low = R_grid_k[i_R_low]
        R_high = R_grid_k[i_R_high]

        # Valores nos cantos
        J_ll = self.value_function[k][i_x_low, i_R_low]
        J_lh = self.value_function[k][i_x_low, i_R_high]
        J_hl = self.value_function[k][i_x_high, i_R_low]
        J_hh = self.value_function[k][i_x_high, i_R_high]

        # Interpolação bilinear
        if x_high == x_low:
            w_x = 0.5
        else:
            w_x = (x - x_low) / (x_high - x_low)

        if R_high == R_low:
            w_R = 0.5
        else:
            w_R = (R - R_low) / (R_high - R_low)

        J_interp = (1 - w_x) * (1 - w_R) * J_ll + \
                   (1 - w_x) * w_R * J_lh + \
                   w_x * (1 - w_R) * J_hl + \
                   w_x * w_R * J_hh

        return J_interp

    def get_action(self,
                   x: float,
                   R: float,
                   k: int) -> float:
        """
        Obter ação ótima por interpolação.

        Usa interpolação bilinear na política.

        Args:
            x: SOC atual
            R: Resíduo atual
            k: Passo atual

        Returns:
            Ação ótima u*
        """
        if self.policy is None or k >= len(self.policy):
            return 0.0

        # Clamping de R se fora do intervalo
        R_grid_k = self.R_grids[k]
        R = np.clip(R, R_grid_k.min(), R_grid_k.max())

        # Encontrar vizinhos em X_grid
        i_x = np.searchsorted(self.X_grid, x)
        i_x = np.clip(i_x, 1, len(self.X_grid) - 1)
        i_x_low = i_x - 1
        i_x_high = i_x

        x_low = self.X_grid[i_x_low]
        x_high = self.X_grid[i_x_high]

        # Encontrar vizinhos em R_grids[k]
        i_R = np.searchsorted(R_grid_k, R)
        i_R = np.clip(i_R, 1, len(R_grid_k) - 1)
        i_R_low = i_R - 1
        i_R_high = i_R

        R_low = R_grid_k[i_R_low]
        R_high = R_grid_k[i_R_high]

        # Ações nos cantos
        u_ll = self.policy[k][i_x_low, i_R_low]
        u_lh = self.policy[k][i_x_low, i_R_high]
        u_hl = self.policy[k][i_x_high, i_R_low]
        u_hh = self.policy[k][i_x_high, i_R_high]

        # Interpolação bilinear
        if x_high == x_low:
            w_x = 0.5
        else:
            w_x = (x - x_low) / (x_high - x_low)

        if R_high == R_low:
            w_R = 0.5
        else:
            w_R = (R - R_low) / (R_high - R_low)

        u_interp = (1 - w_x) * (1 - w_R) * u_ll + \
                   (1 - w_x) * w_R * u_lh + \
                   w_x * (1 - w_R) * u_hl + \
                   w_x * w_R * u_hh

        return u_interp

    def compute_action(self,
                      timestamp: datetime,
                      solar_power: float,
                      load_power: float,
                      battery,
                      tariff,
                      solar_panel,
                      house,
                      pv_forecaster=None,
                      load_forecaster=None) -> float:
        """
        Computar ação para o passo atual.

        Args:
            timestamp: Timestamp atual
            solar_power: Produção solar atual (kW)
            load_power: Consumo atual (kW)
            battery: Objeto Battery
            tariff: Objeto Tariff
            solar_panel: Objeto SolarPanel
            house: Objeto House
            pv_forecaster: Forecaster de PV (ProfilePersistenceForecaster)
            load_forecaster: Forecaster de carga (ProfilePersistenceForecaster)

        Returns:
            Ação ótima (kW), positivo=carga, negativo=descarga
        """
        # Verificar se precisa recalcular política
        need_update = False
        if self.policy is None or self.last_policy_update is None:
            need_update = True
        else:
            hours_since_update = (timestamp - self.last_policy_update).total_seconds() / 3600
            if hours_since_update >= self.params.policy_update_hours:
                need_update = True

        if need_update:
            # Recalcular política
            self._update_policy(timestamp, pv_forecaster, load_forecaster)

        # Calcular resíduo atual
        R_current = solar_power - load_power

        # Obter SOC atual
        x_current = battery.get_soc()

        # Determinar passo k no horizonte
        # (Assumindo que sempre recalculamos do início, k=0)
        k = 0

        # Obter ação ótima
        u_star = self.get_action(x_current, R_current, k)

        # Retornar (note: convenção de sinal pode ser diferente da Battery)
        # Battery: positivo=carga, negativo=descarga
        # SDP: u>0 descarrega, u<0 carrega
        # Então invertemos o sinal
        return -u_star

    def _update_policy(self,
                      timestamp: datetime,
                      pv_forecaster,
                      load_forecaster):
        """
        Atualizar política resolvendo SDP.

        Args:
            timestamp: Timestamp atual
            pv_forecaster: Forecaster de PV
            load_forecaster: Forecaster de carga
        """
        print(f"\n[SDP] Updating policy at {timestamp}...")

        # Obter previsões base R̄
        R_bar = self._get_base_forecast(
            timestamp, pv_forecaster, load_forecaster
        )

        # Salvar previsão base
        self.R_bar_forecast = R_bar

        # Criar grelhas
        self.create_grids(R_bar)

        # Resolver SDP
        self.solve_sdp(R_bar, verbose=True)

        # Atualizar timestamp
        self.last_policy_update = timestamp

    def _get_base_forecast(self,
                          start_time: datetime,
                          pv_forecaster,
                          load_forecaster) -> np.ndarray:
        """
        Obter previsão base R̄ = P̄_pv - P̄_load.

        Args:
            start_time: Timestamp inicial
            pv_forecaster: Forecaster de PV
            load_forecaster: Forecaster de carga

        Returns:
            Array R̄ para horizonte N+1 (índices 0 a N)
            Precisa de N+1 elementos para calcular transições em k=N-1
        """
        N = self.params.N

        # Obter previsões para N+1 passos (0 a N)
        if pv_forecaster is not None:
            P_pv_bar = pv_forecaster.get_forecast(start_time, N + 1)
        else:
            P_pv_bar = np.zeros(N + 1)

        if load_forecaster is not None:
            P_load_bar = load_forecaster.get_forecast(start_time, N + 1)
        else:
            P_load_bar = np.zeros(N + 1)

        # Calcular resíduo
        R_bar = P_pv_bar - P_load_bar

        return R_bar
