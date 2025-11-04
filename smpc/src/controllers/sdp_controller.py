"""
Stochastic Dynamic Programming (SDP) Controller for PV+Battery System (Numba JIT Optimized)

Implementa controle ótimo de bateria com PV considerando:
- Limite de injeção na rede
- Incerteza de previsão (modelo estocástico)
- Modelo AC simplificado (sem perdas de inversor, apenas bateria)
- Programação dinâmica estocástica com quadratura Gaussiana

Baseado no paper de SDP para sistemas PV+Bateria.

Esta versão usa Numba JIT para acelerar o loop principal do SDP.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List as PythonList
from scipy.interpolate import interp1d, RectBivariateSpline
from dataclasses import dataclass
import time
import logging

# --- Importações Numba ---
from numba import jit, prange
from numba.typed import List
import numba.types as types

# Configure logging
logger = logging.getLogger(__name__)


# ############################################################################
# Funções JIT (Lógica de Computação Principal)
# Estas funções são definidas fora das classes para compilação nopython
# ############################################################################

# --- Funções JIT do Modelo da Planta ---

@jit(nopython=True, cache=True)
def p_eff_jit(u: float, eta_c: float, eta_d: float) -> float:
    """Potência efetiva (JIT)."""
    if u < 0:  # Carregando
        return u * eta_c
    elif u > 0:  # Descarregando
        return u / eta_d
    else:
        return 0.0

@jit(nopython=True, cache=True)
def next_soc_jit(x: float, u: float, C_bat: float, dt_hours: float,
                 eta_c: float, eta_d: float,
                 soc_min: float, soc_max: float) -> float:
    """Próximo SOC (JIT)."""
    p_eff = p_eff_jit(u, eta_c, eta_d)
    x_next = x - (p_eff * dt_hours) / C_bat
    # Clip manual para Numba
    if x_next < soc_min:
        x_next = soc_min
    elif x_next > soc_max:
        x_next = soc_max
    return x_next

@jit(nopython=True, cache=True)
def curtailment_jit(u: float, R: float, P_lim: float) -> float:
    """Curtailment (JIT)."""
    return max(0.0, u + R - P_lim)

@jit(nopython=True, cache=True)
def grid_power_jit(u: float, R: float, P_lim: float) -> float:
    """Potência da rede (JIT)."""
    phi = curtailment_jit(u, R, P_lim)
    return -u - R + phi


# --- Funções JIT do Controlador SDP ---

@jit(nopython=True, cache=True)
def stage_cost_jit(u: float, R: float, P_lim: float, c_s: float, c_f: float, dt_hours: float) -> float:
    """Custo por etapa (JIT)."""
    P_g = grid_power_jit(u, R, P_lim)
    P_g_plus = max(0.0, P_g)
    P_g_minus = max(0.0, -P_g)
    cost = (P_g_plus * c_s - P_g_minus * c_f) * dt_hours
    return cost




@jit(nopython=True, cache=True)
def terminal_cost_jit(x: float, soc_target: float, weight: float, C_bat: float, c_s: float, c_f: float) -> float:
    # versão do paper: ignora soc_target/weight; usa média (c_s + c_f)/2
    return - x * C_bat * 0.5 * (c_s + c_f)


@jit(nopython=True, cache=True)
def transition_mean_jit(R_current: float, R_bar_next: float, R_bar_current: float, kappa: float) -> float:
    """Média da transição do resíduo (JIT)."""
    return R_bar_next + kappa * (R_current - R_bar_current)

@jit(nopython=True, cache=True)
def gaussian_quadrature_5pt_jit(rho: float, sigma: float,
                                R_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quadratura Gaussiana 5 pontos (JIT)."""
    m_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    points = rho + m_values * sigma
    weights_unnorm = np.exp(-0.5 * m_values**2)

    R_min = R_grid[0]
    R_max = R_grid[-1]

    # Filtro Numba-friendly
    valid_mask = (points >= R_min) & (points <= R_max)
    valid_points = points[valid_mask]
    valid_weights = weights_unnorm[valid_mask]

    if len(valid_weights) > 0:
        norm_sum = valid_weights.sum()
        if norm_sum > 1e-9:  # Evitar divisão por zero
            valid_weights = valid_weights / norm_sum
        else:
            # Caso raro: pesos muito pequenos
            valid_weights = np.zeros_like(valid_weights)
    else:
        # --- CORREÇÃO APLICADA AQUI ---
        # Todos os pontos fora - clipa a média para o ponto mais próximo
        # np.clip(float, float, float) não é suportado pelo Numba
        
        if rho < R_min:
            clipped_rho = R_min
        elif rho > R_max:
            clipped_rho = R_max
        else:
            clipped_rho = rho
        # --- FIM DA CORREÇÃO ---
            
        valid_points = np.array([clipped_rho], dtype=np.float64)
        valid_weights = np.array([1.0], dtype=np.float64)

    return valid_points, valid_weights

@jit(nopython=True, cache=True)
def _interpolate_value_function_jit(x: float, R: float,
                                    X_grid: np.ndarray,
                                    R_grid_k: np.ndarray,
                                    value_function_k: np.ndarray) -> float:
    """Interpolação bilinear (JIT)."""
    
    # --- CORREÇÃO 1 ---
    len_X_minus_1 = len(X_grid) - 1
    i_x = np.searchsorted(X_grid, x)
    if i_x < 1:
        i_x = 1
    elif i_x > len_X_minus_1:
        i_x = len_X_minus_1
    # --- FIM DA CORREÇÃO 1 ---
        
    i_x_low = i_x - 1
    i_x_high = i_x
    x_low = X_grid[i_x_low]
    x_high = X_grid[i_x_high]

    # --- CORREÇÃO 2 ---
    len_R_minus_1 = len(R_grid_k) - 1
    i_R = np.searchsorted(R_grid_k, R)
    if i_R < 1:
        i_R = 1
    elif i_R > len_R_minus_1:
        i_R = len_R_minus_1
    # --- FIM DA CORREÇÃO 2 ---
        
    i_R_low = i_R - 1
    i_R_high = i_R
    R_low = R_grid_k[i_R_low]
    R_high = R_grid_k[i_R_high]

    # Valores
    J_ll = value_function_k[i_x_low, i_R_low]
    J_lh = value_function_k[i_x_low, i_R_high]
    J_hl = value_function_k[i_x_high, i_R_low]
    J_hh = value_function_k[i_x_high, i_R_high]

    # Pesos
    if x_high == x_low:
        w_x = 0.5
    else:
        w_x = (x - x_low) / (x_high - x_low)

    if R_high == R_low:
        w_R = 0.5
    else:
        w_R = (R - R_low) / (R_high - R_low)

    # Interpolação
    J_interp = (1 - w_x) * (1 - w_R) * J_ll + \
               (1 - w_x) * w_R * J_lh + \
               w_x * (1 - w_R) * J_hl + \
               w_x * w_R * J_hh

    return J_interp

@jit(nopython=True, cache=True)
def _expected_future_cost_jit(x_next: float, R_k: float, k: int,
                              R_bar: np.ndarray, kappa: float, sigma: float,
                              X_grid: np.ndarray,
                              R_grids_list: List[np.ndarray],
                              value_function_list: List[np.ndarray]
                              ) -> float:
    """Custo futuro esperado (JIT)."""
    rho = transition_mean_jit(R_k, R_bar[k+1], R_bar[k], kappa)

    R_grid_k_plus_1 = R_grids_list[k+1]
    points, weights = gaussian_quadrature_5pt_jit(rho, sigma, R_grid_k_plus_1)

    E_J = 0.0
    value_function_k_plus_1 = value_function_list[k+1]

    for i in range(len(points)):
        R_next = points[i]
        w = weights[i]
        J_val = _interpolate_value_function_jit(
            x_next, R_next,
            X_grid,
            R_grid_k_plus_1,
            value_function_k_plus_1
        )
        E_J += w * J_val

    return E_J


@jit(nopython=True, parallel=True)
def _solve_sdp_jit(N: int, X_grid: np.ndarray,
                   R_grids_list: List[np.ndarray],
                   _feasible_actions_cache: List[np.ndarray],
                   R_bar: np.ndarray,
                   # Plant params
                   C_bat: float, dt_hours: float, eta_c: float, eta_d: float,
                   soc_min: float, soc_max: float, P_nom: float, P_lim: float,
                   # Residual params
                   kappa: float, sigma: float,
                   # Cost params
                   c_s: float, c_f: float,
                   soc_target: float, terminal_cost_weight: float
                   ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Loop principal do SDP otimizado com Numba JIT e parallel=True."""
    
    n_x = len(X_grid)

    # Listas tipadas Numba para armazenar resultados
    value_function_list = List.empty_list(types.float64[:, :])
    policy_list = List.empty_list(types.float64[:, :])
    
    # Pré-alocar espaço (necessário para Numba)
    for _ in range(N):
        policy_list.append(np.empty((0, 0), dtype=np.float64))
    for _ in range(N + 1):
        value_function_list.append(np.empty((0, 0), dtype=np.float64))

    # --- Condição Terminal (k=N) ---
    n_R_N = len(R_grids_list[N])
    J_N = np.zeros((n_x, n_R_N), dtype=np.float64)
    # Paraleliza o loop sobre SOC
    for i in prange(n_x):
        terminal_val = terminal_cost_jit(
            X_grid[i], soc_target, terminal_cost_weight, C_bat, c_s,c_f
        )
        J_N[i, :] = terminal_val
    value_function_list[N] = J_N

    # --- Backward DP ---
    for k in range(N - 1, -1, -1):
        n_R_k = len(R_grids_list[k])
        J_k = np.zeros((n_x, n_R_k), dtype=np.float64)
        mu_k = np.zeros((n_x, n_R_k), dtype=np.float64)

        # Paraleliza o loop externo (sobre SOC)
        for i in prange(n_x):
            x_i = X_grid[i]
            
            # Obter ações do cache
            U_i = _feasible_actions_cache[i]

            if len(U_i) == 0:
                # Se não houver ações, custo é infinito
                J_k[i, :] = np.inf
                mu_k[i, :] = 0.0 # Ação padrão
                continue

            for j in range(n_R_k):
                R_kj = R_grids_list[k][j]

                min_cost = np.inf
                best_u = 0.0

                # Loop sobre ações (não pode ser paralelizado, é uma minimização)
                for u_idx in range(len(U_i)):
                    u = U_i[u_idx]

                    # Custo imediato
                    g = stage_cost_jit(
                        u, R_kj, P_lim, c_s, c_f, dt_hours
                    )

                    # Próximo estado
                    x_next = next_soc_jit(
                        x_i, u, C_bat, dt_hours, eta_c, eta_d, soc_min, soc_max
                    )

                    # Custo futuro esperado
                    E_J = _expected_future_cost_jit(
                        x_next, R_kj, k, R_bar, kappa, sigma,
                        X_grid,
                        R_grids_list,
                        value_function_list
                    )
                    
                    total_cost = g + E_J

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_u = u

                J_k[i, j] = min_cost
                mu_k[i, j] = best_u

        value_function_list[k] = J_k
        policy_list[k] = mu_k
        
    return policy_list, value_function_list


# ############################################################################
# Classes Originais (Quase Inalteradas)
# ############################################################################

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
        # Chama a versão JIT
        return transition_mean_jit(R_current, R_bar_next, R_bar_current, self.kappa)


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
        (Wrapper para JIT)
        """
        return p_eff_jit(u, self.eta_c, self.eta_d)

    def next_soc(self, x: float, u: float) -> float:
        """
        Próximo SOC após aplicar ação u.
        (Wrapper para JIT)
        """
        return next_soc_jit(x, u, self.C_bat, self.dt_hours, self.eta_c,
                            self.eta_d, self.soc_min, self.soc_max)

    def curtailment(self, u: float, R: float) -> float:
        """
        Curtailment devido ao limite de injeção.
        (Wrapper para JIT)
        """
        return curtailment_jit(u, R, self.P_lim)

    def grid_power(self, u: float, R: float) -> float:
        """
        Potência da rede.
        (Wrapper para JIT)
        """
        return grid_power_jit(u, R, self.P_lim)

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
        # Esta função é chamada poucas vezes (em create_grids),
        # não precisa ser JIT, mas seu resultado será usado pelo JIT.
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

        return np.array(actions, dtype=np.float64)


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
        self.R_grids: PythonList[np.ndarray] = None  # Grelhas de resíduo
        
        # --- Listas tipadas Numba para JIT ---
        self.R_grids_list: List[np.ndarray] = None
        self._feasible_actions_cache: List[np.ndarray] = None

        # Política ótima (listas Python padrão)
        self.policy: PythonList[np.ndarray] = None
        self.value_function: PythonList[np.ndarray] = None

        # Cache para evitar recálculo
        self.last_policy_update = None
        self.R_bar_forecast = None  # Previsão base R̄

        # Nome do controlador
        self.name = "SDP (Numba JIT)"

    def stage_cost(self, u: float, R: float) -> float:
        """
        Custo por etapa.
        (Wrapper para JIT)
        """
        return stage_cost_jit(u, R, self.plant.P_lim, self.c_s,
                              self.c_f, self.plant.dt_hours)

    def terminal_cost(self, x: float) -> float:
        """
        Custo terminal.
        (Wrapper para JIT)
        """
        return terminal_cost_jit(x, self.params.soc_target,
                                 self.params.terminal_cost_weight,
                                 self.plant.C_bat, self.c_s)

    def create_grids(self, R_bar: np.ndarray):
        """
        Criar grelhas de discretização.

        *** MODIFICADO ***
        Cria listas tipadas Numba para JIT.

        Args:
            R_bar: Previsão base R̄ para cada passo k (tamanho N+1)
        """
        N = self.params.N
        n_x = self.params.n_x
        n_R = self.params.n_R

        # Grelha de SOC (uniforme em [soc_min, soc_max])
        self.X_grid = np.linspace(self.plant.soc_min, self.plant.soc_max, n_x, dtype=np.float64)

        # Pre-compute feasible actions
        # *** Usa Numba typed List ***
        self._feasible_actions_cache = List.empty_list(types.float64[:])
        for x_i in self.X_grid:
            actions = self.plant.feasible_actions(x_i, self.X_grid)
            self._feasible_actions_cache.append(actions)

        # Grelhas de resíduo (uma por passo k, incluindo k=N)
        self.R_grids = []
        # *** Usa Numba typed List ***
        self.R_grids_list = List.empty_list(types.float64[:])
        
        sigma = self.residual_model.sigma

        for k in range(N + 1):
            R_center = R_bar[k]
            R_min = R_center - 3 * sigma
            R_max = R_center + 3 * sigma
            R_grid_k = np.linspace(R_min, R_max, n_R, dtype=np.float64)
            
            # Armazena em ambos os formatos
            self.R_grids.append(R_grid_k)
            self.R_grids_list.append(R_grid_k)

    def gaussian_quadrature_5pt(self,
                                 rho: float,
                                 sigma: float,
                                 R_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quadratura Gaussiana de 5 pontos.
        (Wrapper para JIT)
        """
        # Nota: Esta função não é mais chamada pelo solve_sdp,
        # mas mantida para compatibilidade. O JIT usa a versão _jit.
        return gaussian_quadrature_5pt_jit(rho, sigma, R_grid)

    def solve_sdp(self, R_bar: np.ndarray, verbose: bool = False):
        """
        Resolver SDP usando backward DP com quadratura.

        *** MODIFICADO ***
        Este método agora é um "wrapper" que chama a função _solve_sdp_jit.

        Args:
            R_bar: Previsão base R̄ para cada passo k (tamanho N+1)
            verbose: Imprimir progresso
        """
        solve_start_time = time.time()
        
        if verbose:
            print(f"  [SDP] Chamando _solve_sdp_jit (pode compilar na 1ª execução)...")

        # Chama a função JIT principal com todos os parâmetros
        jit_start = time.time()
        policy_list_jit, value_function_list_jit = _solve_sdp_jit(
            self.params.N,
            self.X_grid,
            self.R_grids_list, # Passa a lista tipada
            self._feasible_actions_cache, # Passa a lista tipada
            R_bar,
            # Plant params
            self.plant.C_bat, self.plant.dt_hours, self.plant.eta_c, self.plant.eta_d,
            self.plant.soc_min, self.plant.soc_max, self.plant.P_nom, self.plant.P_lim,
            # Residual params
            self.residual_model.kappa, self.residual_model.sigma,
            # Cost params
            self.c_s, self.c_f,
            self.params.soc_target, self.params.terminal_cost_weight
        )
        jit_time = time.time() - jit_start
        logger.info(f"  [SDP] JIT solve time: {jit_time:.4f}s")
        
        if verbose:
             print(f"  [SDP] JIT solve concluído em {jit_time:.4f}s")

        # Converte Numba Lists de volta para listas Python padrão
        # para compatibilidade com o resto do código (ex: get_action)
        self.policy = [policy_list_jit[i] for i in range(len(policy_list_jit))]
        self.value_function = [value_function_list_jit[i] for i in range(len(value_function_list_jit))]

        total_solve_time = time.time() - solve_start_time
        logger.info(f"  [SDP] Total solve_sdp time: {total_solve_time:.4f}s")
        if verbose:
            print(f"  SDP solved in {total_solve_time:.2f}s!")

    def _expected_future_cost(self,
                             x_next: float,
                             R_k: float,
                             k: int,
                             R_bar: np.ndarray) -> float:
        """
        Calcular esperança do custo futuro usando quadratura.

        (Esta função não é mais chamada por solve_sdp, pois
         a lógica está em _expected_future_cost_jit.
         Mantida para compatibilidade.)
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
        (Função original mantida, chamada por get_action, etc.)
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
        (Função original mantida, pois é rápida o suficiente
         para ser chamada uma vez por passo de tempo.)
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
        (Função original mantida.)
        """
        action_start = time.time()

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
        interpolation_start = time.time()
        u_star = self.get_action(x_current, R_current, k)
        interpolation_time = time.time() - interpolation_start

        total_action_time = time.time() - action_start
        logger.debug(f"  [SDP] compute_action: {total_action_time:.6f}s (interpolation: {interpolation_time:.6f}s)")

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
        (Função original mantida.)
        """
        policy_update_start = time.time()
        print(f"\n[{self.name}] Updating policy at {timestamp}...")
        logger.info(f"[{self.name}] Starting policy update at {timestamp}")

        # Obter previsões base R̄
        forecast_start = time.time()
        R_bar = self._get_base_forecast(
            timestamp, pv_forecaster, load_forecaster
        )
        forecast_time = time.time() - forecast_start
        logger.info(f"  [{self.name}] Forecast generation: {forecast_time:.4f}s")

        # Salvar previsão base
        self.R_bar_forecast = R_bar

        # Criar grelhas (agora cria Numba Lists)
        grid_start = time.time()
        self.create_grids(R_bar)
        grid_time = time.time() - grid_start
        logger.info(f"  [{self.name}] Grid creation: {grid_time:.4f}s")

        # Resolver SDP (agora chama o wrapper JIT)
        solve_start = time.time()
        self.solve_sdp(R_bar, verbose=True)
        solve_time = time.time() - solve_start
        logger.info(f"  [{self.name}] SDP solve: {solve_time:.4f}s")

        # Atualizar timestamp
        self.last_policy_update = timestamp

        total_update_time = time.time() - policy_update_start
        logger.info(f"[{self.name}] Policy update completed in {total_update_time:.4f}s")
        print(f"[{self.name}] Policy update completed in {total_update_time:.2f}s")

    def _get_base_forecast(self,
                          start_time: datetime,
                          pv_forecaster,
                          load_forecaster) -> np.ndarray:
        """
        Obter previsão base R̄ = P̄_pv - P̄_load.
        (Função original mantida.)
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