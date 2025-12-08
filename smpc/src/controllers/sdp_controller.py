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
from numba import jit, prange, njit
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
def stage_cost_jit(u: float, R: float, P_lim: float, c_s: float, c_f: float, dt_hours: float,
                   c_deg: float, eta_c: float, eta_d: float) -> float:
    """
    Custo por etapa (JIT).

    Inclui:
    - Custo de importação da rede: P_g_plus * c_s
    - Receita de exportação: -P_g_minus * c_f
    - Custo de degradação da bateria: |u| * throughput_factor * c_deg

    Args:
        u: Ação (potência da bateria, negativo=carga, positivo=descarga)
        R: Resíduo (net_load - previsão)
        P_lim: Limite de injeção na rede
        c_s: Preço de compra (€/kWh)
        c_f: Preço de venda (€/kWh)
        dt_hours: Timestep em horas
        c_deg: Custo de degradação (€/kWh de throughput)
        eta_c: Eficiência de carga
        eta_d: Eficiência de descarga
    """
    P_g = grid_power_jit(u, R, P_lim)
    P_g_plus = max(0.0, P_g)      # importação da rede
    P_g_minus = max(0.0, -P_g)    # exportação para a rede

    # Custo da rede
    grid_cost = (P_g_plus * c_s - P_g_minus * c_f) * dt_hours

    # Custo de degradação da bateria (throughput considerando eficiência)
    if u < 0:  # Carregando
        # Energia que entra na bateria (considerando perdas na entrada)
        throughput = abs(u) * eta_c
    elif u > 0:  # Descarregando
        # Energia que sai da bateria (considerando perdas na saída)
        throughput = abs(u) / eta_d
    else:
        throughput = 0.0

    degradation_cost = throughput * dt_hours * c_deg

    return grid_cost + degradation_cost

@njit(parallel=False, cache=True)
def _interp_in_R_at_xindex(R: float,
                           x_index: int,
                           R_grid: np.ndarray,
                           J: np.ndarray) -> float:
    """
    Interpola J(x_index, R) apenas ao longo de R (x fixo por índice).
    Evita bilinear por ação quando x_next cai exatamente na grelha.
    """
    iR = np.searchsorted(R_grid, R)
    if iR < 1:
        iR = 1
    elif iR > len(R_grid) - 1:
        iR = len(R_grid) - 1

    lo = iR - 1
    hi = iR
    Rl = R_grid[lo]
    Rh = R_grid[hi]

    if Rh == Rl:
        w = 0.5
    else:
        w = (R - Rl) / (Rh - Rl)

    return (1.0 - w) * J[x_index, lo] + w * J[x_index, hi]


@njit(parallel=True, cache=True)
def _solve_sdp_jit_fast(N: int,
                        X_grid: np.ndarray,
                        R_grids_list: List[np.ndarray],
                        feasible_actions_list: List[np.ndarray],
                        next_x_index_list: List[np.ndarray],
                        R_bar: np.ndarray,
                        # Plant params
                        C_bat: float, dt_hours: float, eta_c: float, eta_d: float,
                        soc_min: float, soc_max: float, P_nom: float, P_lim: float,
                        # Residual params
                        kappa: float, sigma: float,
                        # Cost params - agora arrays ao longo do horizonte
                        c_s_array: np.ndarray, c_f_array: np.ndarray, c_deg: float,
                        soc_target: float, terminal_cost_weight: float
                        ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Versão rápida do loop SDP:
      - Para cada (k, j) pré-calcula EJ_cache[x_next, j] = E[J_{k+1}(x_next, R_{k+1})]
      - Minimização por ação vira: custo_imediato(u) + EJ_cache[idx_x_next(u), j]
    Preserva a mesma precisão do original.
    """
    n_x = len(X_grid)

    value_list = List.empty_list(types.float64[:, :])
    policy_list = List.empty_list(types.float64[:, :])

    for _ in range(N):
        policy_list.append(np.empty((0, 0), dtype=np.float64))
    for _ in range(N + 1):
        value_list.append(np.empty((0, 0), dtype=np.float64))

    # --- Terminal (k = N) ---
    n_R_N = len(R_grids_list[N])
    JN = np.empty((n_x, n_R_N), dtype=np.float64)
    for i in prange(n_x):
        # Usar preço do step final N
        JN[i, :] = terminal_cost_jit(X_grid[i], soc_target, terminal_cost_weight,
                                     C_bat, c_s_array[N], c_f_array[N])
    value_list[N] = JN

    # --- Backward DP ---
    for k in range(N - 1, -1, -1):
        Rk  = R_grids_list[k]
        Rk1 = R_grids_list[k + 1]
        n_Rk = len(Rk)

        Jk  = np.empty((n_x, n_Rk), dtype=np.float64)
        Muk = np.empty((n_x, n_Rk), dtype=np.float64)

        # 1) Pré-cálculo: EJ_cache[x_next_idx, j] para todos x_next, j
        EJ_cache = np.empty((n_x, n_Rk), dtype=np.float64)

        for j in prange(n_Rk):
            R_kj = Rk[j]
            rho = transition_mean_jit(R_kj, R_bar[k + 1], R_bar[k], kappa)
            pts, wts = gaussian_quadrature_5pt_jit(rho, sigma, Rk1)

            # Integra para CADA x_next (índice m) contra os pontos de quadratura
            for m in range(n_x):
                acc = 0.0
                for q in range(len(pts)):
                    acc += wts[q] * _interp_in_R_at_xindex(pts[q], m, Rk1, value_list[k + 1])
                EJ_cache[m, j] = acc

        # 2) Minimização por ação usando apenas lookups + custo imediato
        for i in prange(n_x):
            U_i      = feasible_actions_list[i]
            next_idx = next_x_index_list[i]

            if len(U_i) == 0:
                for j in range(n_Rk):
                    Jk[i, j]  = np.inf
                    Muk[i, j] = 0.0
                continue

            for j in range(n_Rk):
                R_kj = Rk[j]
                best_cost = 1e300
                best_u    = 0.0

                # Obter preços para o step k atual
                c_s_k = c_s_array[k]
                c_f_k = c_f_array[k]

                for a in range(len(U_i)):
                    u = U_i[a]

                    # custo imediato (inline do stage_cost_jit para evitar alocação)
                    phi = 0.0
                    tmp = u + R_kj - P_lim
                    if tmp > 0.0:
                        phi = tmp

                    P_g = -u - R_kj + phi
                    P_g_plus  = P_g if P_g > 0.0 else 0.0
                    P_g_minus = -P_g if P_g < 0.0 else 0.0
                    grid_cost = (P_g_plus * c_s_k - P_g_minus * c_f_k) * dt_hours

                    if u < 0.0:
                        throughput = -u * eta_c
                    elif u > 0.0:
                        throughput =  u / eta_d
                    else:
                        throughput = 0.0

                    deg_cost = throughput * dt_hours * c_deg

                    total = grid_cost + deg_cost + EJ_cache[next_idx[a], j]

                    if total < best_cost:
                        best_cost = total
                        best_u    = u

                Jk[i, j]  = best_cost
                Muk[i, j] = best_u

        value_list[k] = Jk
        policy_list[k] = Muk

    return policy_list, value_list




@jit(nopython=True, cache=True)
def terminal_cost_jit(x: float, soc_target: float, weight: float, C_bat: float, c_s: float, c_f: float) -> float:
    """
    Terminal cost: valor da energia armazenada no final do horizonte.

    Se c_f > 0: energia vale a média entre comprar (c_s) e vender (c_f)
    Se c_f <= 0: energia vale apenas evitar comprar (c_s), pois vender dá prejuízo

    Fórmula do paper (Eq. 9): g_N(x_N) = -x_N * C_bat * (c_s + c_f) / 2

    CORREÇÃO: Quando c_f <= 0, não deve dividir por 2, pois não há opção de vender.
    O valor da energia é apenas o custo evitado de compra (c_s).
    """
    if c_f > 0.0:
        # Pode vender com lucro: valor é a média entre comprar e vender
        return -x * C_bat * (c_s + c_f) / 2.0
    else:
        # Não pode vender com lucro: valor é apenas evitar comprar
        # NÃO dividir por 2 neste caso!
        return -x * C_bat * c_s


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


# ############################################################################
# Funções Vetorizadas para Processamento em Batch de Ações
# ############################################################################

@jit(nopython=True, cache=True)
def p_eff_vectorized(u_array: np.ndarray, eta_c: float, eta_d: float) -> np.ndarray:
    """Potência efetiva vetorizada para array de ações."""
    n = len(u_array)
    p_eff = np.empty(n, dtype=np.float64)
    for i in range(n):
        u = u_array[i]
        if u < 0:  # Carregando
            p_eff[i] = u * eta_c
        elif u > 0:  # Descarregando
            p_eff[i] = u / eta_d
        else:
            p_eff[i] = 0.0
    return p_eff


@jit(nopython=True, cache=True)
def next_soc_vectorized(x: float, u_array: np.ndarray, C_bat: float, dt_hours: float,
                        eta_c: float, eta_d: float, soc_min: float, soc_max: float) -> np.ndarray:
    """Próximo SOC vetorizado para array de ações."""
    p_eff_array = p_eff_vectorized(u_array, eta_c, eta_d)
    x_next_array = x - (p_eff_array * dt_hours) / C_bat
    # Clip
    for i in range(len(x_next_array)):
        if x_next_array[i] < soc_min:
            x_next_array[i] = soc_min
        elif x_next_array[i] > soc_max:
            x_next_array[i] = soc_max
    return x_next_array


@jit(nopython=True, cache=True)
def stage_cost_vectorized(u_array: np.ndarray, R: float, P_lim: float,
                          c_s: float, c_f: float, dt_hours: float,
                          c_deg: float, eta_c: float, eta_d: float) -> np.ndarray:
    """Custo por etapa vetorizado para array de ações."""
    n = len(u_array)
    costs = np.empty(n, dtype=np.float64)

    for i in range(n):
        u = u_array[i]
        # Grid power
        phi = max(0.0, u + R - P_lim)
        P_g = -u - R + phi
        P_g_plus = max(0.0, P_g)
        P_g_minus = max(0.0, -P_g)

        # Grid cost
        grid_cost = (P_g_plus * c_s - P_g_minus * c_f) * dt_hours

        # Degradation cost
        if u < 0:  # Carregando
            throughput = abs(u) * eta_c
        elif u > 0:  # Descarregando
            throughput = abs(u) / eta_d
        else:
            throughput = 0.0

        degradation_cost = throughput * dt_hours * c_deg
        costs[i] = grid_cost + degradation_cost

    return costs


@jit(nopython=True, cache=True)
def expected_future_cost_vectorized(x_next_array: np.ndarray, R_k: float, k: int,
                                    R_bar: np.ndarray, kappa: float, sigma: float,
                                    X_grid: np.ndarray,
                                    R_grids_list: List[np.ndarray],
                                    value_function_list: List[np.ndarray]) -> np.ndarray:
    """Custo futuro esperado vetorizado para array de estados futuros."""
    n = len(x_next_array)
    E_J_array = np.empty(n, dtype=np.float64)

    for i in range(n):
        E_J_array[i] = _expected_future_cost_jit(
            x_next_array[i], R_k, k,
            R_bar, kappa, sigma,
            X_grid, R_grids_list, value_function_list
        )

    return E_J_array


@jit(nopython=True, parallel=True, cache=True)
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
                   c_s: float, c_f: float, c_deg: float,
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

                # VERSÃO VETORIZADA: Calcular custos de todas as ações de uma vez
                # Custo imediato de todas as ações
                g_array = stage_cost_vectorized(
                    U_i, R_kj, P_lim, c_s, c_f, dt_hours,
                    c_deg, eta_c, eta_d
                )

                # Próximo estado para todas as ações
                x_next_array = next_soc_vectorized(
                    x_i, U_i, C_bat, dt_hours, eta_c, eta_d, soc_min, soc_max
                )

                # Custo futuro esperado para todos os estados futuros
                E_J_array = expected_future_cost_vectorized(
                    x_next_array, R_kj, k, R_bar, kappa, sigma,
                    X_grid,
                    R_grids_list,
                    value_function_list
                )

                # Custo total para cada ação
                total_cost_array = g_array + E_J_array

                # Encontrar ação ótima
                best_idx = np.argmin(total_cost_array)
                min_cost = total_cost_array[best_idx]
                best_u = U_i[best_idx]

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

    # Ajuste de ação para cancelar resíduo
    residual_match_threshold: float = 0.5  # Tolerância (kW) para ajustar u para -R


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
                 soc_max: float = 0.9,   # SOC máximo
                 c_deg: float = 0.01):   # Custo de degradação (€/kWh throughput)
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
            c_deg: Custo de degradação da bateria (€/kWh throughput)
        """
        self.C_bat = C_bat
        self.P_nom = P_nom
        self.P_lim = P_lim
        self.eta_c = eta_charge
        self.eta_d = eta_discharge
        self.dt_hours = dt_minutes / 60.0
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.c_deg = c_deg

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
                 c_s: float,  # Tarifa compra (€/kWh) - usado como default
                 c_f: float,  # Tarifa venda (€/kWh)
                 tariff=None,  # Objeto tariff para preços variáveis
                 export_price: float = None):  # Preço de exportação
        """
        Args:
            params: Parâmetros do SDP
            plant: Modelo da planta
            residual_model: Modelo do resíduo
            c_s: Tarifa de compra (€/kWh) - média ponderada usada como default
            c_f: Tarifa de venda (€/kWh)
            tariff: Objeto tariff (opcional) para obter preços variáveis ao longo do tempo
            export_price: Preço de exportação (€/kWh), usado se tariff não fornecer
        """
        self.params = params
        self.plant = plant
        self.residual_model = residual_model
        self.c_s = c_s
        self.c_f = c_f
        self.tariff = tariff  # Armazena objeto tariff para uso dinâmico
        self.export_price = export_price if export_price is not None else c_f

        # Grelhas de discretização
        self.X_grid = None  # Grelha de SOC
        self.R_grids: PythonList[np.ndarray] = None  # Grelhas de resíduo
        
        # --- Listas tipadas Numba para JIT ---
        self.R_grids_list: List[np.ndarray] = None
        self._feasible_actions_cache: List[np.ndarray] = None
        self._next_x_index_cache: List[np.ndarray] = None   

        # Política ótima (listas Python padrão)
        self.policy: PythonList[np.ndarray] = None
        self.value_function: PythonList[np.ndarray] = None

        # Cache para evitar recálculo
        self.last_policy_update = None
        self.R_bar_forecast = None  # Previsão base R̄

        # Nome do controlador
        self.name = "SDP (Numba JIT)"

        # Interactive visualization (optional)
        self.visualizer = None
        self.enable_visualization = False

    def enable_interactive_visualization(self, output_dir: str = "results/interactive_plots"):
        """
        Enable interactive 3D visualization of interpolation.

        Args:
            output_dir: Directory to save HTML plots
        """
        try:
            from src.interactive_visualization import InterpolationVisualizer
            self.visualizer = InterpolationVisualizer(output_dir=output_dir)
            self.enable_visualization = True
            print(f"[SDP] Interactive visualization enabled. Plots will be saved to {output_dir}")
        except ImportError as e:
            print(f"[SDP] Warning: Could not enable visualization: {e}")
            print("[SDP] Make sure plotly is installed: pip install plotly")
            self.enable_visualization = False

    def save_visualizations(self, prefix: str = "interpolation"):
        """
        Save all collected visualization data to HTML files.

        Args:
            prefix: Prefix for output filenames
        """
        if self.visualizer is not None and self.enable_visualization:
            self.visualizer.save_all_plots(prefix=prefix)
        else:
            print("[SDP] Visualization not enabled or no data collected.")

    def stage_cost(self, u: float, R: float) -> float:
        """
        Custo por etapa.
        (Wrapper para JIT)
        """
        return stage_cost_jit(u, R, self.plant.P_lim, self.c_s,
                              self.c_f, self.plant.dt_hours,
                              self.plant.c_deg, self.plant.eta_c, self.plant.eta_d)

    def terminal_cost(self, x: float) -> float:
        """
        Custo terminal.
        (Wrapper para JIT)
        """
        return terminal_cost_jit(x, self.params.soc_target,
                                 self.params.terminal_cost_weight,
                                 self.plant.C_bat, self.c_s, self.c_f)

    def create_grids(self, R_bar: np.ndarray):
        """
        Criar grelhas de discretização + caches de ações/índices.
        Agora também pré-calcula, para cada ação viável em x_i, o índice do x_next na X_grid.
        """
        N   = self.params.N
        n_x = self.params.n_x
        n_R = self.params.n_R

        # --- Grelha de SOC ---
        self.X_grid = np.linspace(self.plant.soc_min, self.plant.soc_max, n_x, dtype=np.float64)

        # --- Caches de ações e índices (typed lists Numba) ---
        self._feasible_actions_cache = List.empty_list(types.float64[:])
        self._next_x_index_cache     = List.empty_list(types.int64[:])

        dt   = self.plant.dt_hours
        C    = self.plant.C_bat
        eta_c, eta_d = self.plant.eta_c, self.plant.eta_d

        for x_i in self.X_grid:
            # ações viáveis que levam exatamente a pontos da grelha (sua função já faz isso)
            actions = np.ascontiguousarray(self.plant.feasible_actions(x_i, self.X_grid))

            # mapeia cada ação para o índice do x_next correspondente (sem recomputar depois)
            idx_arr = np.empty(len(actions), dtype=np.int64)
            for a_idx in range(len(actions)):
                u = actions[a_idx]
                # p_eff(u)
                if u < 0.0:
                    p_eff = u * eta_c
                elif u > 0.0:
                    p_eff = u / eta_d
                else:
                    p_eff = 0.0

                x_next = x_i - (p_eff * dt) / C

                # escolhe o índice de grelha mais próximo (evita drift de arredondamento)
                iR = np.searchsorted(self.X_grid, x_next)
                if iR < 1:
                    iR = 1
                elif iR > len(self.X_grid) - 1:
                    iR = len(self.X_grid) - 1
                lo = iR - 1
                hi = iR
                idx_best = lo if abs(self.X_grid[lo] - x_next) <= abs(self.X_grid[hi] - x_next) else hi
                idx_arr[a_idx] = idx_best

            self._feasible_actions_cache.append(actions)                  # float64[:]
            self._next_x_index_cache.append(np.ascontiguousarray(idx_arr))# int64[:]

        # --- Grelhas de resíduo (uma por k) ---
        self.R_grids      = []
        self.R_grids_list = List.empty_list(types.float64[:])

        sigma = self.residual_model.sigma

        # Usar ±3σ: boa cobertura sem perder resolução (se observar clamping, subir p/ 4–5)
        sigma_multiplier = 3  # Increased from 3 to 5 to handle large forecast errors

        for k in range(N + 1):
            R_center = R_bar[k]
            R_min = R_center - sigma_multiplier * sigma
            R_max = R_center + sigma_multiplier * sigma
            R_grid_k = np.linspace(R_min, R_max, n_R, dtype=np.float64)

            self.R_grids.append(R_grid_k)
            self.R_grids_list.append(np.ascontiguousarray(R_grid_k))


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

    def solve_sdp(self, R_bar: np.ndarray, c_s_array: np.ndarray, c_f_array: np.ndarray, verbose: bool = False):
        """
        Resolver SDP usando backward DP com quadratura.
        Wrapper que chama a versão _fast (pré-cálculo de E[J]).

        Agora usa arrays de preços variáveis ao longo do horizonte, permitindo
        modelar corretamente tarifas bi-horárias (ponta vs fora de ponta).

        Args:
            R_bar: Previsão base do resíduo [N+1]
            c_s_array: Array de preços de compra ao longo do horizonte [N+1]
            c_f_array: Array de preços de venda ao longo do horizonte [N+1]
            verbose: Se True, mostra informação detalhada
        """
        solve_start_time = time.time()

        if verbose:
            print(f" [SDP] Chamando _solve_sdp_jit_fast (pode compilar na 1ª execução)...")
            print(f" [SDP] Price range: buy [{c_s_array.min():.4f}, {c_s_array.max():.4f}] €/kWh")
            print(f" [SDP] Price range: sell [{c_f_array.min():.4f}, {c_f_array.max():.4f}] €/kWh")

        jit_start = time.time()
        policy_list_jit, value_function_list_jit = _solve_sdp_jit_fast(
            self.params.N,
            self.X_grid,
            self.R_grids_list,
            self._feasible_actions_cache,
            self._next_x_index_cache,
            R_bar,
            # Plant params
            self.plant.C_bat, self.plant.dt_hours, self.plant.eta_c, self.plant.eta_d,
            self.plant.soc_min, self.plant.soc_max, self.plant.P_nom, self.plant.P_lim,
            # Residual params
            self.residual_model.kappa, self.residual_model.sigma,
            # Cost params - agora arrays variáveis ao longo do horizonte
            c_s_array, c_f_array, self.plant.c_deg,
            self.params.soc_target, self.params.terminal_cost_weight
        )
        jit_time = time.time() - jit_start
        logger.info(f" [SDP] JIT fast solve time: {jit_time:.4f}s")

        if verbose:
            print(f" [SDP] JIT fast solve concluído em {jit_time:.4f}s")

        # converter para listas Python para compatibilidade externa
        self.policy = [policy_list_jit[i] for i in range(len(policy_list_jit))]
        self.value_function = [value_function_list_jit[i] for i in range(len(value_function_list_jit))]

        total_solve_time = time.time() - solve_start_time
        logger.info(f" [SDP] Total solve_sdp time: {total_solve_time:.4f}s")
        if verbose:
            print(f" SDP solved in {total_solve_time:.2f}s!")


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
                   k: int,
                   timestamp: datetime) -> float:
        """
        Obter ação ótima por interpolação.
        (Função original mantida, pois é rápida o suficiente
         para ser chamada uma vez por passo de tempo.)
        """
        if self.policy is None or k >= len(self.policy):
            return 0.0

        # DEBUG: Log some policy values to understand what's happening
        if k == 0:
            logger.debug(f"  get_action: x={x:.4f}, R={R:.4f}, k={k}")
            logger.debug(f"  R_grid range: [{self.R_grids[k].min():.4f}, {self.R_grids[k].max():.4f}]")

        # Clamping de R se fora do intervalo
        R_grid_k = self.R_grids[k]
        R_clamped = np.clip(R, R_grid_k.min(), R_grid_k.max())

        if R != R_clamped:
            # Get predicted R from forecast (if available)
            R_predicted = self.R_bar_forecast[k] if self.R_bar_forecast is not None and k < len(self.R_bar_forecast) else None
            if R_predicted is not None:
                logger.warning(f"  R_actual={R:.4f} vs R_predicted={R_predicted:.4f} (error={R - R_predicted:.4f}) → clamped to {R_clamped:.4f} at {timestamp} k={k}")
            else:
                logger.warning(f"  R_actual={R:.4f} clamped to {R_clamped:.4f} at {timestamp} k={k}")

        # Encontrar vizinhos em X_grid
        i_x = np.searchsorted(self.X_grid, x)
        i_x = np.clip(i_x, 1, len(self.X_grid) - 1)
        i_x_low = i_x - 1
        i_x_high = i_x

        x_low = self.X_grid[i_x_low]
        x_high = self.X_grid[i_x_high]

        # Encontrar vizinhos em R_grids[k] - use R_clamped!
        i_R = np.searchsorted(R_grid_k, R_clamped)
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

        # DEBUG: Log corner values
        if k == 0:
            logger.debug(f"  Corner actions: ll={u_ll:.4f}, lh={u_lh:.4f}, hl={u_hl:.4f}, hh={u_hh:.4f}")

        # Interpolação bilinear
        if x_high == x_low:
            w_x = 0.5
        else:
            w_x = (x - x_low) / (x_high - x_low)

        if R_high == R_low:
            w_R = 0.5
        else:
            w_R = (R_clamped - R_low) / (R_high - R_low)

        u_interp = (1 - w_x) * (1 - w_R) * u_ll + \
                   (1 - w_x) * w_R * u_lh + \
                   w_x * (1 - w_R) * u_hl + \
                   w_x * w_R * u_hh

        if k == 0:
            logger.debug(f"  Interpolated action: {u_interp:.4f}, weights: w_x={w_x:.3f}, w_R={w_R:.3f}")

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

        # Determinar passo k no horizonte baseado no tempo desde última atualização
        if self.last_policy_update is not None:
            minutes_since_update = (timestamp - self.last_policy_update).total_seconds() / 60
            k = int(minutes_since_update / self.params.dt_minutes)
            # Garantir que k não exceda o horizonte
            k = min(k, self.params.N - 1)
        else:
            k = 0

        # Obter ação ótima
        interpolation_start = time.time()
        u_star = self.get_action(x_current, R_current, k, timestamp)
        interpolation_time = time.time() - interpolation_start

        # Ajuste para dar match ao resíduo real (correção da discretização)
        # Se a intenção do controlador é cancelar o resíduo (u ≈ -R),
        # ajustar para o valor exato -R para minimizar troca com a rede
        u_ideal = -R_current  # Ação que cancelaria completamente o resíduo
        match_threshold = self.params.residual_match_threshold  # kW - tolerância para considerar que o controlador quer dar match

        # Verificar se u_star está próximo de u_ideal (dentro da tolerância)
        if abs(u_star - u_ideal) < match_threshold:
            # Verificar se u_ideal respeita os limites da bateria
            if abs(u_ideal) <= self.plant.P_nom:
                # Verificar se a ação não viola os limites de SOC
                x_next_ideal = next_soc_jit(
                    x_current, u_ideal, self.plant.C_bat, self.plant.dt_hours,
                    self.plant.eta_c, self.plant.eta_d,
                    self.plant.soc_min, self.plant.soc_max
                )
                # Se o SOC resultante está dentro dos limites, usar u_ideal
                if self.plant.soc_min <= x_next_ideal <= self.plant.soc_max:
                    logger.debug(f"  [SDP] Adjusting u_star={u_star:.3f} to u_ideal={u_ideal:.3f} to match residual R={R_current:.3f}")
                    u_star = u_ideal

        total_action_time = time.time() - action_start
        logger.debug(f"  [SDP] compute_action: k={k}, {total_action_time:.6f}s (interpolation: {interpolation_time:.6f}s)")

        # Collect visualization data if enabled
        if self.enable_visualization and self.visualizer is not None:
            # Get value function for current state
            J_star = self._interpolate_value_function(x_current, R_current, k)

            # Store data for visualization
            self.visualizer.add_timestep_data(
                timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                x_current=x_current,
                R_current=R_current,
                k=k,
                X_grid=self.X_grid,
                R_grid=self.R_grids[k],
                policy=self.policy[k],
                value_function=self.value_function[k],
                u_star=u_star,
                J_star=J_star
            )

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

        # Obter arrays de preços ao longo do horizonte
        c_s_array, c_f_array = self._get_price_arrays(timestamp)
        c_s_min, c_s_max = c_s_array.min(), c_s_array.max()

        # Mostrar informação sobre preços
        if c_s_min != c_s_max:
            print(f"  Time-varying tariff detected!")
            print(f"  Buy price range: {c_s_min:.4f} - {c_s_max:.4f} €/kWh")
            logger.info(f"  Variable tariff: buy [{c_s_min:.4f}, {c_s_max:.4f}] €/kWh")
        else:
            print(f"  Fixed tariff: buy {c_s_min:.4f} €/kWh, sell {c_f_array[0]:.4f} €/kWh")
            logger.info(f"  Fixed tariff: buy {c_s_min:.4f}, sell {c_f_array[0]:.4f} €/kWh")

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

        # Resolver SDP com preços variáveis
        solve_start = time.time()
        self.solve_sdp(R_bar, c_s_array, c_f_array, verbose=True)
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

    def _get_price_arrays(self, start_time: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obter arrays de preços de compra e venda ao longo do horizonte.

        Retorna arrays c_s[k] e c_f[k] para k=0,...,N onde:
        - c_s[k]: preço de compra no step k
        - c_f[k]: preço de venda no step k (geralmente constante)

        Se tariff não foi fornecido, usa valores constantes self.c_s e self.c_f.

        Args:
            start_time: Timestamp inicial do horizonte

        Returns:
            (c_s_array, c_f_array): Arrays de preços ao longo do horizonte
        """
        N = self.params.N
        dt_minutes = self.params.dt_minutes

        # Inicializar arrays com valores default
        c_s_array = np.full(N + 1, self.c_s)
        c_f_array = np.full(N + 1, self.c_f)

        # Se temos objeto tariff, obter preços variáveis
        if self.tariff is not None:
            for k in range(N + 1):
                # Calcular timestamp para este step
                current_time = start_time + timedelta(minutes=k * dt_minutes)

                # Obter preço de compra do tariff (varia com bi-horária)
                c_s_array[k] = self.tariff.get_price(current_time)

                # Preço de venda (export_price) geralmente é constante
                c_f_array[k] = self.export_price

        return c_s_array, c_f_array