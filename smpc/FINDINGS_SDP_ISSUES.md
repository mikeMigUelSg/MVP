# Análise Completa: Porque o SDP é Pior que o Rule-Based

## Resumo dos Resultados
- **SDP: 6.74 €** (22.4% PIOR)
- **Rule-Based: 5.51 €**
- **Diferença: +1.24 €**

## 🔴 PROBLEMAS PRINCIPAIS IDENTIFICADOS

### 1. SUB-UTILIZAÇÃO DA BATERIA (PROBLEMA CRÍTICO)

O SDP descarrega a bateria com **menos potência** do que o necessário:

```
Potência média de descarga durante demanda:
- SDP:        0.52 kW  ❌
- Rule-Based: 0.62 kW  ✓ (+19.2%)
```

**Impacto:**
- SDP descarrega 5 kWh a MENOS que o rule-based (52.82 kWh vs 57.74 kWh)
- Resultado: Compra mais energia da rede durante períodos de demanda
- **Custo extra em períodos de demanda: +16.3%**

### 2. DECISÕES CONSERVADORAS DURANTE EXCESSO SOLAR

O SDP não aproveita bem a energia solar disponível:

```
Bateria a carregar durante excesso solar:
- SDP:        323/388 períodos (83.2%)  ❌
- Rule-Based: 342/388 períodos (88.1%)  ✓

SDP desperdiça energia solar:
- 15 timesteps com SOC ≤ 0.85 mas NÃO carrega
- Excesso solar médio desperdiçado: 0.11 kW
```

**Impacto:**
- SDP carrega 5 kWh a MENOS que o rule-based (64.92 kWh vs 69.90 kWh)
- **Custo extra em períodos de excesso solar: +11.4%**

### 3. PIOR DESEMPENHO NO FIM DO DIA (19h-23h)

As piores decisões do SDP concentram-se nas horas 19-23:

```
Horas com maior diferença de custo:
  Hora    SDP (€)   RB (€)    Diferença
  21h     0.0074    0.0026    +0.0048 (185% pior!)
  20h     0.0067    0.0025    +0.0043 (168% pior!)
  19h     0.0051    0.0029    +0.0023 (76% pior!)
  22h     0.0071    0.0048    +0.0023 (48% pior!)
  23h     0.0094    0.0072    +0.0022 (31% pior!)
```

**Conclusão:** O SDP está a ser **MUITO conservador** na descarga da bateria durante o pico de demanda noturno.

### 4. DECISÃO OCASIONALMENTE MÁ: CARGA DA REDE

```
Carga da bateria DURANTE demanda:
- SDP:        6 timesteps  ❌
- Rule-Based: 0 timesteps  ✓

Total carregado da rede: 0.03 kWh
Custo: ~0.01 € (pequeno mas indevido)
```

## 🔍 ANÁLISE DE CAUSA RAIZ

### Porque é que o SDP descarrega menos?

O problema está provavelmente em **um ou mais** destes componentes:

#### A) **Modelo de Resíduo Muito Conservador**

```yaml
Parâmetros atuais:
  half_life_minutes: 90.5 min   # Persistência do erro de previsão
  sigma_residual_kw: 1.2 kW     # Incerteza do resíduo
```

**Hipótese:**
- Se `sigma` for muito alto → SDP assume grande incerteza
- Se `half_life` for muito alto → SDP assume que erros de previsão persistem muito tempo
- **Resultado:** SDP fica "com medo" de descarregar a bateria porque acha que pode precisar dela mais tarde

#### B) **Previsões de Load/Solar Incorretas**

O SDP usa **ProfilePersistenceForecaster** (média de N semanas):
- Se as previsões forem sistematicamente erradas
- O SDP toma decisões baseadas em informação falsa

**PRECISA VERIFICAR:**
1. Qual é o erro médio das previsões (MAE, RMSE)?
2. As previsões são pessimistas (subestimam solar, sobrestimam load)?

#### C) **Horizonte de Planejamento vs Atualização**

```yaml
horizon_steps: 96      # 24h
policy_update_hours: 6  # Recalcula a cada 6h
```

- O SDP otimiza para 24h à frente
- Mas só recalcula a cada 6h
- **Gap:** Se as condições mudarem muito, o SDP continua a seguir uma política desatualizada

#### D) **Penalização Implícita de Descarga**

O SDP pode estar a evitar descarregar porque:
1. Não quer arriscar ficar com SOC baixo (sem terminal cost explícito)
2. A função de valor pode ter "aprendido" que manter SOC alto é melhor

## 💡 SOLUÇÕES PROPOSTAS (POR PRIORIDADE)

### 🎯 Solução 1: CALIBRAR PARÂMETROS DO MODELO DE RESÍDUO

**Ação:**
1. Analisar erro real de previsão (load e solar)
2. Ajustar `sigma_residual_kw` para refletir incerteza real
3. Ajustar `half_life_minutes` para tempo de correlação real

**Implementação:**
```python
# Executar calibração automática
from src.controllers.sdp_calibration import calibrate_residual_model

sigma_opt, half_life_opt = calibrate_residual_model(
    historical_data,
    pv_forecaster,
    load_forecaster
)
```

**Modificar config.yaml:**
```yaml
sdp:
  calibration:
    enabled: true  # Ativar calibração automática
    calibration_days: 30  # Usar mais dados
```

---

### 🎯 Solução 2: AUMENTAR AGRESSIVIDADE DE DESCARGA

**Problema:** SDP descarrega com ~84% da potência do rule-based (0.52 vs 0.62 kW)

**Hipótese:** O modelo está a ser demasiado conservador porque teme ficar sem bateria

**Ação:** Adicionar um "prêmio" por usar a bateria durante demanda:

```python
@jit(nopython=True, cache=True)
def stage_cost_jit(u, R, P_lim, c_s, c_f, dt_hours,
                   battery_discharge_bonus=0.001):  # NOVO
    """Custo por etapa com bônus por descarga durante demanda"""
    P_g = grid_power_jit(u, R, P_lim)
    P_g_plus = max(0.0, P_g)  # Import
    P_g_minus = max(0.0, -P_g)  # Export

    cost = (P_g_plus * c_s - P_g_minus * c_f) * dt_hours

    # NOVO: Incentivo por descarregar durante demanda (R < 0)
    if u > 0 and R < 0:  # Descarga durante demanda líquida
        cost -= battery_discharge_bonus * u * dt_hours

    return cost
```

---

### 🎯 Solução 3: RECALCULAR POLÍTICA MAIS FREQUENTEMENTE

**Modificar config.yaml:**
```yaml
sdp:
  policy_update_hours: 3  # Era 6h → Reduzir para 3h ou até 1h
```

**Pros:** Política mais adaptativa às condições reais
**Cons:** Mais computação (mas com Numba JIT, deve ser viável)

---

### 🎯 Solução 4: VERIFICAR E CORRIGIR FORECASTERS

**Ação:** Analisar qualidade das previsões

```python
# Script de análise
from src.forecasters.profile_persistence_forecaster import ProfilePersistenceForecaster

# Para cada timestep da simulação:
for t in timestamps:
    P_pv_real = solar.get_production(t)
    P_pv_pred = pv_forecaster.get_forecast(t, 1)[0]

    P_load_real = house.get_consumption(t)
    P_load_pred = load_forecaster.get_forecast(t, 1)[0]

    # Calcular MAE, RMSE, bias
```

**Se forecasters tiverem bias:**
- Solar subestimado → SDP não carrega o suficiente
- Load sobrestimado → SDP é conservador na descarga

**Correção:** Calibrar ou usar forecasters melhores (ARIMA, ML, etc.)

---

### 🎯 Solução 5: AJUSTAR GRID DE AÇÕES

**Problema potencial:** O SDP discretiza ações possíveis

**Verificar em** `PlantModel.feasible_actions()`:
- Quantas ações estão disponíveis?
- Há granularidade suficiente?

**Ação:**
```yaml
sdp:
  n_soc_points: 150  # Aumentar de 100
  n_residual_points: 50  # Aumentar de 25
```

Mais pontos → Mais precisão → Melhores decisões
(Trade-off: Tempo de computação)

---

## 📊 PRÓXIMOS PASSOS

### Passo 1: Análise de Previsões
```bash
python analyze_forecast_quality.py
```
→ Verificar se previsões são o problema

### Passo 2: Calibração de Parâmetros
```yaml
# config.yaml
sdp:
  calibration:
    enabled: true
```
→ Re-executar simulação

### Passo 3: Se ainda não resolver...
- Implementar Solução 2 (bonus por descarga)
- Reduzir policy_update_hours para 3h ou 1h
- Considerar modelo de incerteza mais sofisticado

---

## 🎓 CONCLUSÃO

O problema **NÃO É** o custo de degradação ou terminal cost.

O problema **É** que o SDP está a ser **EXCESSIVAMENTE CONSERVADOR**:
1. ❌ Descarrega com menos potência (19% menos que rule-based)
2. ❌ Não aproveita bem excesso solar (5% menos períodos de carga)
3. ❌ Especialmente ruim no pico noturno (19h-23h)

**Causa raiz mais provável:**
- Modelo de resíduo com parâmetros mal calibrados (sigma/half-life)
- Previsões com bias ou erro sistemático
- Horizonte longo (24h) com atualização pouco frequente (6h)

**Solução prioritária:**
1. 🔧 Calibrar `sigma_residual_kw` e `half_life_minutes` com dados reais
2. 🔬 Analisar qualidade dos forecasters (MAE, bias)
3. ⚡ Reduzir `policy_update_hours` de 6h para 3h

Implementar estas soluções deve trazer o SDP para pelo menos **igualar ou superar** o rule-based.
