"""
Análise detalhada do impacto do Terminal Cost no SDP

Investiga como o terminal cost weight e soc_target afetam as decisões do SDP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def analyze_terminal_cost():
    """Analisa o impacto do terminal cost nas decisões do SDP"""

    print("="*80)
    print("ANÁLISE: IMPACTO DO TERMINAL COST NO SDP")
    print("="*80)

    # Load results
    results_dir = Path("results/data")

    try:
        sdp_results = pd.read_csv(results_dir / "results_sdp (numba jit).csv")
        rb_results = pd.read_csv(results_dir / "results_rule-based.csv")

        # Convert timestamp to datetime
        sdp_results['timestamp'] = pd.to_datetime(sdp_results['timestamp'])
        rb_results['timestamp'] = pd.to_datetime(rb_results['timestamp'])

    except FileNotFoundError as e:
        print(f"Erro: Ficheiros não encontrados: {e}")
        return

    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    sdp_config = config['controller']['sdp']

    print("\n" + "="*80)
    print("CONFIGURAÇÃO DO TERMINAL COST")
    print("="*80)

    print(f"\nParâmetros:")
    print(f"  soc_target:           {sdp_config['soc_target']}")
    print(f"  terminal_cost_weight: {sdp_config['terminal_cost_weight']}")
    print(f"  Horizonte:            {sdp_config['horizon_steps']} passos = {sdp_config['horizon_steps']*15/60:.1f}h")
    print(f"  Policy update:        A cada {sdp_config['policy_update_hours']}h")

    battery_capacity = config['battery']['capacity_kwh']
    tariff_price = config['tariff']['simple']['price']

    print(f"\n  Battery capacity:     {battery_capacity} kWh")
    print(f"  Tariff price:         {tariff_price} €/kWh")

    # Calculate terminal cost magnitude
    # terminal_cost = weight * |x - soc_target| * C_bat * c_s
    max_deviation = 1.0 - sdp_config['soc_target']  # Maximum possible deviation
    max_terminal_cost = sdp_config['terminal_cost_weight'] * max_deviation * battery_capacity * tariff_price

    print(f"\n  Custo terminal máximo: {max_terminal_cost:.2f} €")
    print(f"    (Quando SOC desvia {max_deviation*100:.0f}% do target)")

    # Cost per timestep for comparison
    dt_hours = 0.25
    typical_import_cost = 2.0 * tariff_price * dt_hours  # 2kW import

    print(f"\n  Custo típico por timestep: {typical_import_cost:.4f} €")
    print(f"    (Importar 2kW durante 15min)")

    ratio = max_terminal_cost / typical_import_cost
    print(f"\n  ⚠️  Ratio: Terminal cost é {ratio:.1f}x um timestep normal!")

    print("\n" + "="*80)
    print("ANÁLISE DA TRAJETÓRIA DO SOC")
    print("="*80)

    print("\nEstatísticas do SOC:")
    print(f"\n  SDP:")
    print(f"    Média:  {sdp_results['battery_soc'].mean():.3f}")
    print(f"    Std:    {sdp_results['battery_soc'].std():.3f}")
    print(f"    Min:    {sdp_results['battery_soc'].min():.3f}")
    print(f"    Max:    {sdp_results['battery_soc'].max():.3f}")

    print(f"\n  Rule-Based:")
    print(f"    Média:  {rb_results['battery_soc'].mean():.3f}")
    print(f"    Std:    {rb_results['battery_soc'].std():.3f}")
    print(f"    Min:    {rb_results['battery_soc'].min():.3f}")
    print(f"    Max:    {rb_results['battery_soc'].max():.3f}")

    # Check how close SDP SOC is to target
    sdp_deviation_from_target = np.abs(sdp_results['battery_soc'] - sdp_config['soc_target'])
    rb_deviation_from_target = np.abs(rb_results['battery_soc'] - sdp_config['soc_target'])

    print(f"\n  Desvio médio do target ({sdp_config['soc_target']}):")
    print(f"    SDP:        {sdp_deviation_from_target.mean():.3f}")
    print(f"    Rule-Based: {rb_deviation_from_target.mean():.3f}")

    if sdp_deviation_from_target.mean() < rb_deviation_from_target.mean():
        print(f"\n  ✓ SDP mantém SOC mais próximo do target!")
        print(f"    Isto CONFIRMA que o terminal cost está a influenciar as decisões")

    print("\n" + "="*80)
    print("ANÁLISE TEMPORAL: QUANDO O SDP RECALCULA A POLÍTICA")
    print("="*80)

    # Policy updates happen every 6 hours
    policy_update_hours = sdp_config['policy_update_hours']

    print(f"\nO SDP recalcula a política a cada {policy_update_hours}h")
    print(f"Durante esse período, ele usa a mesma política calculada")

    # Find times when policy likely updates (every 6h from start)
    start_time = sdp_results['timestamp'].iloc[0]

    print(f"\nÚltimos policy updates estimados:")
    for i in range(1, 5):
        update_time = start_time + pd.Timedelta(hours=policy_update_hours * i)
        if update_time <= sdp_results['timestamp'].iloc[-1]:
            # Find SOC at that time
            idx = sdp_results['timestamp'].searchsorted(update_time)
            if idx < len(sdp_results):
                soc_at_update = sdp_results.iloc[idx]['battery_soc']
                deviation = abs(soc_at_update - sdp_config['soc_target'])
                print(f"  {update_time}: SOC = {soc_at_update:.3f} (desvio = {deviation:.3f})")

    print("\n" + "="*80)
    print("PROBLEMA: COMO O TERMINAL COST AFETA AS DECISÕES")
    print("="*80)

    print("\nQuando o SDP recalcula a política no instante t:")
    print(f"  1. Olha {sdp_config['horizon_steps']*15/60:.0f}h para a frente (até t+{sdp_config['horizon_steps']*15/60:.0f}h)")
    print(f"  2. Impõe que no final (t+{sdp_config['horizon_steps']*15/60:.0f}h), o SOC deve estar em {sdp_config['soc_target']}")
    print(f"  3. Penaliza com peso {sdp_config['terminal_cost_weight']} qualquer desvio desse target")

    print("\n⚠️  PROBLEMA:")
    print("    Se o terminal cost weight é muito alto, o SDP vai:")
    print("    a) Priorizar atingir o SOC target no fim do horizonte")
    print("    b) Em vez de minimizar o custo total de energia")

    print("\n    Exemplo:")
    print("    - Se SOC atual = 0.2 e target = 0.5")
    print("    - SDP vai CARREGAR a bateria mesmo que seja caro")
    print("    - Só para garantir que atinge 0.5 no fim do horizonte")

    print("\n    - Se SOC atual = 0.8 e target = 0.5")
    print("    - SDP vai DESCARREGAR a bateria mesmo que não seja necessário")
    print("    - Só para garantir que atinge 0.5 no fim do horizonte")

    print("\n" + "="*80)
    print("COMPARAÇÃO COM RULE-BASED")
    print("="*80)

    print("\nRule-Based NÃO tem terminal cost:")
    print("  ✓ Reage às condições ATUAIS")
    print("  ✓ Excesso solar → carrega")
    print("  ✓ Alta tarifa + demanda → descarrega")
    print("  ✓ Não se preocupa com SOC futuro artificial")

    print("\nSDP COM terminal cost:")
    print("  ❌ Tenta atingir SOC=0.5 no futuro")
    print("  ❌ Pode ignorar oportunidades boas (ex: carregar com solar grátis)")
    print("  ❌ Pode fazer decisões más (ex: descarregar quando não é necessário)")

    print("\n" + "="*80)
    print("VERIFICAÇÃO: SOC vs SOLAR POWER")
    print("="*80)

    # Analyze when there's excess solar but SDP doesn't charge
    sdp_excess_solar = sdp_results[sdp_results['solar_power'] > sdp_results['load_power']].copy()
    sdp_excess_solar['net_solar'] = sdp_excess_solar['solar_power'] - sdp_excess_solar['load_power']

    # When there's excess solar, battery should charge (positive power)
    sdp_excess_not_charging = sdp_excess_solar[sdp_excess_solar['battery_power'] <= 0]

    print(f"\nPeríodos com excesso solar:")
    print(f"  Total: {len(sdp_excess_solar)} timesteps")
    print(f"  SDP NÃO carrega: {len(sdp_excess_not_charging)} timesteps ({len(sdp_excess_not_charging)/len(sdp_excess_solar)*100:.1f}%)")

    if len(sdp_excess_not_charging) > 0:
        print(f"\n  ⚠️  Durante estes períodos:")
        print(f"    SOC médio: {sdp_excess_not_charging['battery_soc'].mean():.3f}")
        print(f"    Excesso solar médio: {sdp_excess_not_charging['net_solar'].mean():.2f} kW")

        # Check if SOC is above target
        high_soc = sdp_excess_not_charging[sdp_excess_not_charging['battery_soc'] > sdp_config['soc_target']]
        print(f"\n  SOC > target ({sdp_config['soc_target']}): {len(high_soc)} timesteps")
        if len(high_soc) > 0:
            print(f"    → SDP não carrega porque SOC já está acima do target!")
            print(f"    → Energia solar DESPERDIÇADA!")

    print("\n" + "="*80)
    print("SOLUÇÕES")
    print("="*80)

    print("\nOpção 1: REMOVER o terminal cost")
    print("  → terminal_cost_weight: 0.0")
    print("  Pros: SDP fica livre para otimizar sem restrições artificiais")
    print("  Cons: Pode levar a SOC extremos no fim do horizonte")

    print("\nOpção 2: REDUZIR drasticamente o terminal cost weight")
    print("  → terminal_cost_weight: 0.01 ou 0.001")
    print("  Pros: Pequeno incentivo para manter SOC razoável")
    print("  Cons: Pode ainda influenciar decisões")

    print("\nOpção 3: MUDAR o soc_target dinamicamente")
    print("  → Usar soc_target baseado em previsão de preços")
    print("  → Se preços altos esperados → target mais alto")
    print("  → Se preços baixos esperados → target mais baixo")

    print("\nOpção 4: AUMENTAR o horizonte")
    print("  → Se o horizonte fosse muito longo (ex: 96h)")
    print("  → O terminal cost teria menos impacto nas decisões imediatas")
    print("  Cons: Muito mais pesado computacionalmente")

    print("\n" + "="*80)
    print("RECOMENDAÇÃO")
    print("="*80)

    print("\n🎯 Começar com: terminal_cost_weight = 0.0")
    print("   (Remover completamente o terminal cost)")
    print("\n   Depois verificar:")
    print("   - Se o SOC fica em valores razoáveis")
    print("   - Se o custo melhora em relação ao rule-based")
    print("   - Se há melhor aproveitamento do solar")

    print("\n" + "="*80)

    # Create a plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: SOC comparison
    axes[0].plot(sdp_results['timestamp'], sdp_results['battery_soc'],
                 label='SDP', alpha=0.7, linewidth=1)
    axes[0].plot(rb_results['timestamp'], rb_results['battery_soc'],
                 label='Rule-Based', alpha=0.7, linewidth=1)
    axes[0].axhline(sdp_config['soc_target'], color='red', linestyle='--',
                    label=f'SOC Target ({sdp_config["soc_target"]})', linewidth=2)
    axes[0].set_ylabel('SOC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Comparação de SOC: SDP vs Rule-Based')

    # Plot 2: Battery power
    axes[1].plot(sdp_results['timestamp'], sdp_results['battery_power'],
                 label='SDP', alpha=0.7, linewidth=1)
    axes[1].plot(rb_results['timestamp'], rb_results['battery_power'],
                 label='Rule-Based', alpha=0.7, linewidth=1)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Battery Power (kW)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Potência da Bateria (+ = carga, - = descarga)')

    # Plot 3: Solar vs Load
    axes[2].plot(sdp_results['timestamp'], sdp_results['solar_power'],
                 label='Solar', alpha=0.7, linewidth=1)
    axes[2].plot(sdp_results['timestamp'], sdp_results['load_power'],
                 label='Load', alpha=0.7, linewidth=1)
    axes[2].fill_between(sdp_results['timestamp'],
                         sdp_results['solar_power'],
                         sdp_results['load_power'],
                         where=sdp_results['solar_power'] > sdp_results['load_power'],
                         alpha=0.3, label='Excesso Solar', color='green')
    axes[2].set_ylabel('Power (kW)')
    axes[2].set_xlabel('Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Produção Solar vs Consumo')

    plt.tight_layout()
    output_file = Path("results/plots/terminal_cost_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")

    return sdp_results, rb_results

if __name__ == "__main__":
    results = analyze_terminal_cost()
