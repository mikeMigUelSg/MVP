# Análise Interativa de Previsões

Script interativo para analisar e inspecionar previsões de carga e produção fotovoltaica usadas pelo otimizador MPC.

## Funcionalidades

- **Visualização de séries temporais**: Mostra carga e produção solar reais
- **Clique interativo**: Clique em qualquer ponto para ver as previsões feitas naquele momento
- **Comparação detalhada**: Compara previsões vs realidade com métricas de erro (MAE)
- **Múltiplos métodos**: Suporta diferentes métodos de previsão (mean, naive, moving_average)

## Uso Básico

```bash
# Análise padrão (7 dias a partir de 01/01/2025, método 'mean')
python3 analyze_forecasts.py

# Especificar período e método
python3 analyze_forecasts.py --start "2025-01-15 00:00:00" --days 3 --method moving_average

# Usar configuração customizada
python3 analyze_forecasts.py --config custom_config.yaml
```

## Argumentos

- `--config`: Caminho para arquivo de configuração (default: `config.yaml`)
- `--start`: Timestamp inicial no formato "YYYY-MM-DD HH:MM:SS" (default: "2025-01-01 00:00:00")
- `--days`: Número de dias a analisar (default: 7.0)
- `--method`: Método de previsão - `mean`, `naive`, ou `moving_average` (default: `mean`)

## Como Usar o Modo Interativo

1. **Execute o script**: O script abrirá uma janela com 3 gráficos
   - Gráfico superior: Consumo de carga
   - Gráfico do meio: Produção solar
   - Gráfico inferior: Comparação (inicialmente vazio)

2. **Clique num ponto**: Clique em qualquer ponto nos gráficos superiores
   - Uma linha verde tracejada aparecerá mostrando a previsão feita naquele momento
   - O ponto clicado será marcado com um círculo vermelho
   - O gráfico inferior mostrará a comparação detalhada entre previsão e realidade

3. **Interprete os resultados**:
   - Linhas sólidas = valores reais
   - Linhas tracejadas = valores previstos
   - MAE (Mean Absolute Error) = erro médio da previsão

## Métodos de Previsão

### Historical (RECOMENDADO - Padrão)
```bash
python3 analyze_forecasts.py --method historical
```
Usa dados históricos de **7 dias atrás** (mesmo dia da semana). Por exemplo, para prever Quarta 15 Jan às 12h, usa dados reais de Quarta 8 Jan às 12h. Este é o método mais realista e **usado por padrão no MPC**.

**Vantagens:**
- Usa apenas dados PASSADOS (não trapaceia com dados do futuro!)
- Captura variações horárias naturais
- Reflete padrões do mesmo dia da semana
- Mais realista para otimização MPC

**Como funciona:**
- Momento de previsão: 15 Jan 2025, 12:00
- Dados usados: 8 Jan 2025, 12:00 (7 dias atrás)
- Horizonte: Próximas 24 horas (96 steps)

### Mean Forecast
```bash
python3 analyze_forecasts.py --method mean
```
Usa a média dos últimos 24h escalada ao padrão histórico. Suaviza variações mas mantém tendência temporal.

### Naive Persistence
```bash
python3 analyze_forecasts.py --method naive
```
Escala o padrão histórico pelo último valor conhecido. Útil para horizontes muito curtos.

### Moving Average
```bash
python3 analyze_forecasts.py --method moving_average
```
Usa média móvel dos últimos 12 valores (3 horas) escalada ao padrão histórico. Equilibra reatividade e estabilidade.

## Exemplos

### Analisar fim de semana com método naive
```bash
python3 analyze_forecasts.py --start "2025-01-04 00:00:00" --days 2 --method naive
```

### Analisar uma semana completa
```bash
python3 analyze_forecasts.py --start "2025-01-01 00:00:00" --days 7 --method mean
```

### Análise de curto prazo com alta resolução
```bash
python3 analyze_forecasts.py --start "2025-01-15 12:00:00" --days 1 --method moving_average
```

## Interpretação dos Resultados

### MAE (Mean Absolute Error)
- **< 0.5 kW**: Previsão excelente
- **0.5 - 1.0 kW**: Previsão boa
- **1.0 - 2.0 kW**: Previsão aceitável
- **> 2.0 kW**: Previsão necessita melhorias

### Análise Visual
- **Previsão acompanha realidade**: Método adequado
- **Previsão constante, realidade variável**: Considerar método mais adaptativo
- **Previsão com offset constante**: Possível viés nos dados históricos

## Dicas

1. **Clique em diferentes momentos do dia**: Manhã, tarde, noite para ver como as previsões variam
2. **Compare métodos**: Execute com diferentes `--method` para ver qual funciona melhor
3. **Períodos críticos**: Analise transições (amanhecer/anoitecer) onde previsões são mais difíceis
4. **Horizonte efetivo**: O gráfico de comparação mostra apenas até onde há dados reais

## Troubleshooting

### Erro ao carregar dados
```
Erro: FileNotFoundError
```
**Solução**: Verifique que os arquivos de dados existem em `data/load/` e `data/solar/`

### Plot não aparece
**Solução**: Certifique-se de que tem matplotlib instalado:
```bash
pip install matplotlib
```

### Previsões fora do esperado
**Solução**: Verifique se há dados suficientes no período histórico (mínimo 24h antes do start)

## Estrutura do Output

Ao clicar num ponto, o terminal mostra:
```
→ Clique detectado em: 2025-01-15 12:00:00
  Gerando previsões com método 'mean'...
  ✓ Previsão visualizada
  MAE Carga: 0.234 kW
  MAE Solar: 0.156 kW
  Horizonte efetivo: 96 steps
```

## Próximos Passos

Após analisar as previsões:
1. Se MAE é alto, considere usar métodos mais sofisticados (deep learning)
2. Ajuste o horizonte de contexto (`context_hours`) nos forecasters
3. Experimente diferentes janelas para moving average
4. Integre previsões meteorológicas para solar
