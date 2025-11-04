# Energy Management System - MPC Framework

Framework modular para gestão de energia doméstica com bateria, painel solar e otimização via MPC (Model Predictive Control) ou controlo baseado em regras.

## Características

### Componentes

- **Bateria**: Sistema de armazenamento com modelação de degradação e eficiência
- **Painel Solar**: Produção fotovoltaica com dados reais
- **Casa**: Consumo residencial com dados reais
- **Tarifa**: Suporte para tarifas simples e bi-horárias

### Controladores

- **Rule-Based**: Controlo heurístico simples baseado em regras
  - Carrega bateria com excesso solar
  - Descarrega bateria durante períodos de preço alto

- **MPC**: Model Predictive Control com otimização linear
  - Minimiza custos totais (compra + degradação)
  - Horizonte de previsão configurável
  - Considera previsões de solar e consumo

### Sistema

- **Balanço de Potência**: Garante sempre o equilíbrio energético
- **Custos de Degradação**: Modela desgaste da bateria
- **Análise de Poupanças**: Compara com cenário baseline (sem bateria)
- **Visualizações**: Gráficos detalhados de comportamento do sistema

## Estrutura do Projeto

```
mpc/
├── config.yaml              # Configuração principal
├── simulate.py              # Script de simulação
├── requirements.txt         # Dependências Python
├── data/
│   ├── load/               # Dados de consumo
│   └── solar/              # Dados de produção solar
├── src/
│   ├── components/         # Componentes do sistema
│   │   ├── battery.py     # Bateria
│   │   ├── solar.py       # Painel solar
│   │   ├── house.py       # Casa
│   │   └── tariff.py      # Tarifas
│   ├── controllers/        # Controladores
│   │   ├── rule_based.py
│   │   └── mpc_controller.py
│   ├── system.py          # Sistema integrado
│   └── visualization.py   # Visualizações
└── results/               # Resultados (gerado automaticamente)
    ├── plots/
    └── data/
```

## Instalação

```bash
# Instalar dependências
pip install -r requirements.txt
```

## Configuração

Edite `config.yaml` para configurar:

### Simulação
```yaml
simulation:
  start_date: "2025-01-01 00:00:00"
  end_date: "2025-01-08 00:00:00"
  timestep_minutes: 15
```

### Bateria
```yaml
battery:
  capacity_kwh: 10.0
  max_power_kw: 5.0
  efficiency_charge: 0.95
  efficiency_discharge: 0.95
  degradation_cost_per_kwh: 0.01
```

### Tarifa
```yaml
tariff:
  type: "bihoraria"  # ou "simple"
  bihoraria:
    peak_price: 0.18
    off_peak_price: 0.10
```

### Controlador
```yaml
controller:
  type: "mpc"  # ou "rule_based"
  mpc:
    horizon_steps: 96  # 24h com timestep de 15min
```

## Uso

### Simulação Simples

```bash
python simulate.py
```

### Modo Comparação

Edite `config.yaml`:
```yaml
comparison:
  enabled: true
  controllers: ["rule_based", "mpc"]
```

Depois execute:
```bash
python simulate.py
```

## Dados

### Formato dos Dados

**Consumo** (`merged_consumos.xlsx`):
- Colunas: `Data`, `Hora`, `Consumo registado (kW)`
- Formato: Dados de 15 em 15 minutos

**Solar** (`pvdata.csv`):
- Formato: `timestamp;pv_1;pv_2;...`
- Separador: ponto e vírgula (`;`)

**Nota Importante**: O ano dos dados históricos é ignorado. O matching é feito apenas por mês/dia/hora/minuto, permitindo usar dados de qualquer ano em simulações de outros anos.

## Saídas

### Fatura

O sistema imprime uma fatura detalhada com:
- Fluxos de energia (produção, consumo, importação, exportação)
- Performance da bateria (ciclos, degradação)
- Custos totais
- Poupanças vs baseline

### Gráficos

1. **System Behavior**: Fluxos de potência, SOC, grid, preços
2. **Daily Analysis**: Análise agregada por hora do dia
3. **Comparison**: Comparação entre controladores (modo comparação)

### Ficheiros CSV

Resultados detalhados timestamp a timestamp em `results/data/`

## Equação de Balanço de Potência

```
solar + grid_import = load + battery_charge + grid_export
```

Onde:
- `solar`: produção fotovoltaica
- `grid_import`: importação da rede
- `load`: consumo da casa
- `battery_charge`: carga da bateria (negativo = descarga)
- `grid_export`: exportação para a rede

## MPC - Model Predictive Control

O controlador MPC resolve um problema de otimização linear:

**Minimizar**: Custo total (compra de energia + degradação da bateria)

**Sujeito a**:
- Balanço de potência
- Limites de potência da bateria
- Limites de SOC (State of Charge)
- Dinâmica da bateria (eficiências)

O horizonte de previsão é recalculado a cada timestep (receding horizon).

## Exemplo de Saída

```
============================================================
           ENERGY MANAGEMENT SYSTEM - INVOICE
============================================================

Controller: MPC
------------------------------------------------------------

ENERGY FLOWS:
  Solar Production:          350.25 kWh
  House Consumption:         420.50 kWh
  Grid Import:                85.30 kWh
  Grid Export:                15.05 kWh

BATTERY PERFORMANCE:
  Battery Cycles:              2.50
  Degradation Cost:            0.25 €
  Self-Consumption Rate:       79.7 %
  Self-Sufficiency Rate:       83.3 %

COSTS:
  Total System Cost:          15.80 €
  Baseline Cost (no bat):     25.50 €

============================================================
TOTAL SAVINGS:                 9.70 €
SAVINGS PERCENTAGE:           38.0 %
============================================================
```

## Desenvolvimento

### Adicionar Novo Componente

1. Criar classe em `src/components/`
2. Implementar métodos: `step()`, `reset()`, `get_stats()`
3. Adicionar ao `__init__.py`

### Adicionar Novo Controlador

1. Criar classe em `src/controllers/`
2. Implementar método: `compute_action()`
3. Adicionar suporte em `simulate.py`

## Licença

MIT

## Contacto

Para questões ou sugestões, criar issue no repositório.
