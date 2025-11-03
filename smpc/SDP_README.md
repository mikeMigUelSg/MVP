# Controlador SDP para Sistema PV+Bateria

Implementação de Programação Dinâmica Estocástica (SDP) para otimização de bateria em sistemas fotovoltaicos, baseado em paper de controle ótimo com incerteza de previsão.

## Características Principais

- **Modelo AC simplificado**: Dinâmica da bateria com perdas (sem perdas de inversor)
- **Limite de injeção**: Curtailment quando excede limite da rede
- **Previsão estocástica**: Modelo de meia-vida com ruído Gaussiano
- **Quadratura de 5 pontos**: Integração eficiente da esperança
- **Interpolação bilinear**: Execução em tempo real com política pré-calculada
- **Calibração automática**: Ajuste de parâmetros t₁/₂ e σ

## Arquitetura

```
src/
├── forecasters/
│   └── profile_persistence_forecaster.py  # Persistência por perfil (média semanal)
└── controllers/
    ├── sdp_controller.py                  # Controlador SDP principal
    └── sdp_calibration.py                 # Calibração de parâmetros
```

## Uso Rápido

### Exemplo Simples (Dados Sintéticos)

```bash
python example_sdp_simple.py
```

Este exemplo usa dados sintéticos e demonstra o controlador em ação sem precisar de arquivos de dados.

### Teste Completo (Dados Reais)

```bash
python test_sdp_controller.py
```

Este script usa dados reais de `data/load/` e `data/solar/`:
1. Carrega dados históricos
2. Cria forecasters de persistência por perfil
3. Calibra automaticamente t₁/₂ e σ
4. Simula um dia
5. Gera gráficos

## Componentes Principais

### 1. ProfilePersistenceForecaster

Previsão por persistência de perfil: média do mesmo dia-da-semana nas últimas N semanas.

```python
from src.forecasters import ProfilePersistenceForecaster

forecaster = ProfilePersistenceForecaster(n_weeks=3, dt_minutes=15)
forecaster.set_data(data, value_column='P_pv')
forecast = forecaster.get_forecast(start_time, n_steps=36)
```

### 2. ResidualModel

Modelo de resíduo R = P_pv - P_load com meia-vida:

```
R̂_{k+1} = R̄_{k+1} + κ(R̂_k - R̄_k)
κ = 2^(-Δt/t₁/₂)
```

### 3. PlantModel

Modelo AC da planta:
- Dinâmica SOC com perdas da bateria
- Curtailment (limite de injeção)
- Potência da rede

### 4. SDPController

Controlador principal:
- Backward DP com quadratura Gaussiana
- Interpolação bilinear em tempo real
- Recalcula política a cada 6h

### 5. Calibração

```python
from src.controllers import quick_calibrate

results = quick_calibrate(
    historical_data=df,  # ['timestamp', 'P_pv', 'P_load']
    start_time=datetime(2024, 1, 1),
    pv_forecaster=pv_forecaster,
    load_forecaster=load_forecaster
)
# Retorna: t_half_minutes, sigma_kw, best_mse
```

## Integração Básica

```python
from src.controllers import (
    SDPController, SDPParams, PlantModel, ResidualModel
)
from src.forecasters import ProfilePersistenceForecaster

# 1. Criar forecasters
pv_forecaster = ProfilePersistenceForecaster(n_weeks=3)
pv_forecaster.set_data(pv_data, 'P_pv')

load_forecaster = ProfilePersistenceForecaster(n_weeks=3)
load_forecaster.set_data(load_data, 'P_load')

# 2. Calibrar
from src.controllers import quick_calibrate
calib = quick_calibrate(historical_data, start_time,
                       pv_forecaster, load_forecaster)

# 3. Criar controlador
params = SDPParams(
    N=36,
    dt_minutes=15,
    n_x=150,
    n_R=50,
    t_half_minutes=calib['t_half_minutes'],
    sigma_R=calib['sigma_kw']
)

plant = PlantModel(
    C_bat=10.0, P_nom=5.0, P_lim=10.0,
    eta_charge=0.95, eta_discharge=0.95,
    dt_minutes=15
)

residual_model = ResidualModel(
    params.t_half_minutes, 15, params.sigma_R
)

controller = SDPController(
    params, plant, residual_model,
    c_s=0.20, c_f=0.05
)

# 4. Usar em simulação
action = controller.compute_action(
    timestamp=t,
    solar_power=pv,
    load_power=load,
    battery=battery,
    tariff=None,
    solar_panel=solar,
    house=house,
    pv_forecaster=pv_forecaster,
    load_forecaster=load_forecaster
)
```

## Parâmetros Recomendados

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| N | 36 | Horizonte (9h @ 15min) |
| dt_minutes | 15 | Resolução temporal |
| n_x | 150 | Pontos SOC |
| n_R | 50 | Pontos Resíduo |
| t_half_minutes | 45-60 | Meia-vida (calibrar) |
| sigma_R | 0.4-0.6 | Desvio padrão (calibrar) |
| policy_update_hours | 6 | Recalcular política |

Para simulações mais rápidas (com perda de precisão):
```python
params = SDPParams(n_x=100, n_R=30, N=24)
```

## Algoritmo

### Backward DP
```
Para k = N-1 ... 0:
    Para cada (x_i, R_k,j):
        J_k(x_i, R_k,j) = min_u [g(u, R) + E[J_{k+1}(x', R')]]
```

### Quadratura de 5 Pontos
- Distribuição: R_{k+1} ~ N(ρ, σ)
- Pontos: ρ + m·σ com m ∈ {-2,-1,0,1,2}
- Pesos: w_m ∝ exp(-m²/2)

### Execução Tempo Real
1. Medir (x*, R*)
2. Interpolar: u* = μ_k(x*, R*)
3. Aplicar u*

## Convenções de Sinal

**Bateria (u):**
- u > 0: descarga
- u < 0: carga

**Rede (P_g):**
- P_g > 0: importação
- P_g < 0: exportação

**Nota:** `compute_action` retorna sinal invertido para compatibilidade com classe `Battery`.

## Arquivos de Dados

### Entrada
```python
# PV
pv_data = pd.DataFrame({
    'timestamp': [...],
    'P_pv': [...]  # kW
})

# Carga
load_data = pd.DataFrame({
    'timestamp': [...],
    'P_load': [...]  # kW
})
```

### Saída
```python
results = {
    'timestamp': [...],
    'soc': [...],           # 0-1
    'battery_power': [...], # kW
    'solar_power': [...],   # kW
    'load_power': [...],    # kW
    'grid_power': [...],    # kW (+ = compra)
    'residual': [...]       # kW (pv - load)
}
```

## Troubleshooting

**Otimização lenta:**
- Reduzir n_x e n_R (ex: 100, 30)
- Reduzir horizonte N (ex: 24)

**Previsão ruim:**
- Verificar ≥3 semanas de histórico
- Calibrar com período mais longo

**"No data in window":**
- Ajustar start_time para data válida

## Limitações

1. Sem perdas de inversor (apenas bateria)
2. Tarifas fixas (não dinâmicas)
3. Limite de injeção constante
4. Previsão base usa persistência

## Próximos Passos

- [ ] Adicionar perdas quadráticas do inversor
- [ ] Tarifas dinâmicas (time-of-use)
- [ ] Integração com ML forecasts
- [ ] Paralelização do backward DP
- [ ] Análise de sensibilidade
- [ ] Comparação com MPC

## Referências

Baseado em paper de SDP para PV+Bateria com quadratura Gaussiana e persistência por perfil.
