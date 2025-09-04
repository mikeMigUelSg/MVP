# Oferta Indexada (BTN) — Fórmula e Tabelas (sem Tarifa Social)



## Fórmula da fatura (oferta indexada — BTN)

$$
\text{Custo Total} \,=\, \sum_i \Big[\big(POMIE_i \times (1+\text{Perdas}) \times K1 \, + \, K2 \, + \, \text{TAR}_{\text{Energia},i}\big)\times \text{Consumo}_i\Big] \, + \, \big(K3 \, + \, \text{TAR}_{\text{Potência}}\big)\times N_{\text{dias}}
$$

- \(i\): intervalo quarto-horário (ou horário)  
- \(POMIE_i\): preço OMIE em €/kWh  
- **Perdas**: coeficiente médio de perdas de rede  
- **K1, K2, K3**: componentes comerciais do comercializador  
- **TAR\(_{\text{Energia},i}\)**: Tarifa de Acesso às Redes — energia (BTN) do período \(i\)  
- **TAR\(_{\text{Potência}}\)**: Tarifa de Acesso às Redes — potência (BTN) por dia  
- \(N_{\text{dias}}\): número de dias do período de faturação

---
Consumo de energia (kWh): com potência até 6,9 kVA, aplica-se 6% aos primeiros 200 kWh por 30 dias
## Componentes comerciais (EDP — oferta indexada)

| Componente | Valor | Unidade | Observação |
|---|---:|:---:|---|
| **K1** | **1,28** | — | Fator multiplicativo aplicado ao OMIE |
| **K2** | **0,01720** | €/kWh | Custos variáveis não ligados diretamente à evolução do preço de mercado |
| **K3** | **0,09410** | €/dia | Custo fixo diário do comercializador |

---

## TAR (ERSE) — BTN (≤ 20,7 kVA) — Energia

| Ciclo | Período | Valor (€/kWh) |
|---|---|---:|
| **Simples** | — | **0,0600** |
| **Bi-horária** | Fora de vazio | **0,0830** |
|  | Vazio | **0,0149** |
| **Tri-horária** | Ponta | **0,2469** |
|  | Cheias | **0,0388** |
|  | Vazio | **0,0149** |

---

## TAR (ERSE) — BTN (≤ 20,7 kVA) — Potência

**Base:** **0,0460 €/kVA·dia**

| Potência contratada (kVA) | €/dia |
|---:|---:|
| 1,15 | 0,0529 |
| 2,30 | 0,1058 |
| 3,45 | 0,1587 |
| 4,60 | 0,2116 |
| 5,75 | 0,2645 |
| 6,90 | 0,3174 |
| 10,35 | 0,4761 |
| 13,80 | 0,6348 |
| 17,25 | 0,7935 |
| 20,70 | 0,9522 |

---

### Notas
- Os valores de **TAR** são definidos pela **ERSE** e variam por ciclo horário e potência contratada.
- Os valores **K1/K2/K3** são específicos da oferta indexada do comercializador.
- Este ficheiro **ignora** quaisquer descontos da **Tarifa Social**.
