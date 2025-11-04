"""
Análise profunda das decisões do SDP vs Rule-Based

Investiga momento-a-momento onde o SDP toma decisões piores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

def deep_analyze_decisions():
    """Analisa decisões momento a momento"""

    print("="*80)
    print("ANÁLISE PROFUNDA: DECISÕES SDP vs RULE-BASED")
    print("="*80)

    # Load results
    results_dir = Path("results/data")

    try:
        sdp_results = pd.read_csv(results_dir / "results_sdp (numba jit).csv")
        rb_results = pd.read_csv(results_dir / "results_rule-based.csv")

        # Convert timestamp to datetime
        sdp_results['timestamp'] = pd.to_datetime(sdp_results['timestamp'])
        rb_results['timestamp'] = pd.to_datetime(rb_results['timestamp'])

        with open(results_dir / "summary_sdp (numba jit).yaml") as f:
            sdp_summary = yaml.load(f, Loader=yaml.UnsafeLoader)
        with open(results_dir / "summary_rule-based.yaml") as f:
            rb_summary = yaml.load(f, Loader=yaml.UnsafeLoader)

    except FileNotFoundError as e:
        print(f"Erro: Ficheiros não encontrados: {e}")
        print("Execute primeiro a simulação com terminal_cost_weight = 0.0")
        return None, None

    dt_hours = 0.25
    tariff_price = 0.15
    export_price = 0.045

    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS")
    print("="*80)

    print(f"\nCustos totais (SEM degradação):")
    print(f"  SDP:        {sdp_summary['total_cost']:.2f} €")
    print(f"  Rule-Based: {rb_summary['total_cost']:.2f} €")
    print(f"  Diferença:  {sdp_summary['total_cost'] - rb_summary['total_cost']:.2f} €")

    print(f"\nImportação/Exportação:")
    print(f"  SDP:        Import: {sdp_summary['total_grid_import_kwh']:.2f} kWh, Export: {sdp_summary['total_grid_export_kwh']:.2f} kWh")
    print(f"  Rule-Based: Import: {rb_summary['total_grid_import_kwh']:.2f} kWh, Export: {rb_summary['total_grid_export_kwh']:.2f} kWh")

    print("\n" + "="*80)
    print("ANÁLISE 1: EXCESSO SOLAR - DECISÕES DE CARGA")
    print("="*80)

    # Find periods with excess solar
    sdp_excess = sdp_results[sdp_results['solar_power'] > sdp_results['load_power']].copy()
    rb_excess = rb_results[rb_results['solar_power'] > rb_results['load_power']].copy()

    sdp_excess['net_solar'] = sdp_excess['solar_power'] - sdp_excess['load_power']
    rb_excess['net_solar'] = rb_excess['solar_power'] - rb_excess['load_power']

    print(f"\nPeríodos com excesso solar: {len(sdp_excess)} timesteps")

    # When there's excess solar, check if battery is charging
    sdp_charging = sdp_excess[sdp_excess['battery_power'] > 0]
    rb_charging = rb_excess[rb_excess['battery_power'] > 0]

    print(f"\nBateria a carregar durante excesso solar:")
    print(f"  SDP:        {len(sdp_charging)}/{len(sdp_excess)} ({len(sdp_charging)/len(sdp_excess)*100:.1f}%)")
    print(f"  Rule-Based: {len(rb_charging)}/{len(rb_excess)} ({len(rb_charging)/len(rb_excess)*100:.1f}%)")

    # When NOT charging during excess solar
    sdp_not_charging = sdp_excess[sdp_excess['battery_power'] <= 0]
    rb_not_charging = rb_excess[rb_excess['battery_power'] <= 0]

    if len(sdp_not_charging) > 0:
        print(f"\n⚠️  SDP NÃO carrega durante excesso solar: {len(sdp_not_charging)} timesteps")
        print(f"    SOC médio: {sdp_not_charging['battery_soc'].mean():.3f}")
        print(f"    Excesso solar médio: {sdp_not_charging['net_solar'].mean():.2f} kW")

        # Check why - is it because battery is full?
        near_full = sdp_not_charging[sdp_not_charging['battery_soc'] > 0.85]
        print(f"    SOC > 0.85 (quase cheio): {len(near_full)} timesteps")

        not_full = sdp_not_charging[sdp_not_charging['battery_soc'] <= 0.85]
        if len(not_full) > 0:
            print(f"\n    ❌ SOC ≤ 0.85 mas NÃO carrega: {len(not_full)} timesteps")
            print(f"       SOC médio: {not_full['battery_soc'].mean():.3f}")
            print(f"       Excesso solar médio: {not_full['net_solar'].mean():.2f} kW")
            print(f"       → Energia solar DESPERDIÇADA!")

    print("\n" + "="*80)
    print("ANÁLISE 2: DEMANDA LÍQUIDA - DECISÕES DE DESCARGA")
    print("="*80)

    # Find periods with net demand (load > solar)
    sdp_demand = sdp_results[sdp_results['load_power'] > sdp_results['solar_power']].copy()
    rb_demand = rb_results[rb_results['load_power'] > rb_results['solar_power']].copy()

    sdp_demand['net_demand'] = sdp_demand['load_power'] - sdp_demand['solar_power']
    rb_demand['net_demand'] = rb_demand['load_power'] - rb_demand['solar_power']

    print(f"\nPeríodos com demanda líquida: {len(sdp_demand)} timesteps")

    # When there's demand, check if battery is discharging
    sdp_discharging = sdp_demand[sdp_demand['battery_power'] < 0]
    rb_discharging = rb_demand[rb_demand['battery_power'] < 0]

    print(f"\nBateria a descarregar durante demanda:")
    print(f"  SDP:        {len(sdp_discharging)}/{len(sdp_demand)} ({len(sdp_discharging)/len(sdp_demand)*100:.1f}%)")
    print(f"  Rule-Based: {len(rb_discharging)}/{len(rb_demand)} ({len(rb_discharging)/len(rb_demand)*100:.1f}%)")

    # Average discharge power
    if len(sdp_discharging) > 0:
        print(f"\nPotência média de descarga:")
        print(f"  SDP:        {-sdp_discharging['battery_power'].mean():.2f} kW")
        print(f"  Rule-Based: {-rb_discharging['battery_power'].mean():.2f} kW")

    print("\n" + "="*80)
    print("ANÁLISE 3: COMPARAÇÃO DIRETA - MESMAS CONDIÇÕES")
    print("="*80)

    # Merge datasets to compare decisions under same conditions
    merged = sdp_results.merge(rb_results, on='timestamp', suffixes=('_sdp', '_rb'))

    # Calculate cost per timestep for both
    merged['import_cost_sdp'] = merged['grid_import_sdp'] * dt_hours * tariff_price
    merged['export_revenue_sdp'] = merged['grid_export_sdp'] * dt_hours * export_price
    merged['net_cost_sdp'] = merged['import_cost_sdp'] - merged['export_revenue_sdp']

    merged['import_cost_rb'] = merged['grid_import_rb'] * dt_hours * tariff_price
    merged['export_revenue_rb'] = merged['grid_export_rb'] * dt_hours * export_price
    merged['net_cost_rb'] = merged['import_cost_rb'] - merged['export_revenue_rb']

    merged['cost_diff'] = merged['net_cost_sdp'] - merged['net_cost_rb']

    print(f"\nCusto médio por timestep:")
    print(f"  SDP:        {merged['net_cost_sdp'].mean():.4f} €")
    print(f"  Rule-Based: {merged['net_cost_rb'].mean():.4f} €")

    # Find worst decisions (where SDP costs much more)
    worst_decisions = merged[merged['cost_diff'] > 0.01].sort_values('cost_diff', ascending=False).head(20)

    if len(worst_decisions) > 0:
        print(f"\n⚠️  Piores decisões do SDP (top 20):")
        print(f"    Total timesteps com custo extra > 0.01€: {len(merged[merged['cost_diff'] > 0.01])}")
        print(f"    Custo extra acumulado: {merged[merged['cost_diff'] > 0]['cost_diff'].sum():.2f} €")

        print("\n    Top 5 piores momentos:")
        print(f"    Colunas disponíveis: {list(worst_decisions.columns[:10])}")  # Debug
        for idx, row in worst_decisions.head(5).iterrows():
            try:
                print(f"\n    {row['timestamp']}")
                print(f"      Solar: {row['solar_power_sdp']:.2f} kW, Load: {row['load_power_sdp']:.2f} kW")
                print(f"      SDP:  SOC={row['battery_soc_sdp']:.2f}, Battery={row['battery_power_sdp']:+.2f} kW, Grid={row['grid_power_sdp']:+.2f} kW")
                print(f"      RB:   SOC={row['battery_soc_rb']:.2f}, Battery={row['battery_power_rb']:+.2f} kW, Grid={row['grid_power_rb']:+.2f} kW")
                print(f"      Cost: SDP={row['net_cost_sdp']:.4f} €, RB={row['net_cost_rb']:.4f} € (diff={row['cost_diff']:.4f} €)")
            except KeyError as e:
                print(f"      Error accessing column: {e}")

    print("\n" + "="*80)
    print("ANÁLISE 4: PADRÕES DE ERRO")
    print("="*80)

    # Categorize timesteps
    merged['situation'] = 'neutral'
    merged.loc[merged['solar_power_sdp'] > merged['load_power_sdp'], 'situation'] = 'excess_solar'
    merged.loc[merged['solar_power_sdp'] < merged['load_power_sdp'], 'situation'] = 'net_demand'

    print("\nCusto médio por situação:")
    for situation in ['excess_solar', 'net_demand']:
        subset = merged[merged['situation'] == situation]
        if len(subset) > 0:
            sdp_cost = subset['net_cost_sdp'].mean()
            rb_cost = subset['net_cost_rb'].mean()
            diff = sdp_cost - rb_cost
            print(f"\n  {situation}:")
            print(f"    SDP:        {sdp_cost:.4f} €")
            print(f"    Rule-Based: {rb_cost:.4f} €")
            print(f"    Diferença:  {diff:.4f} € ({diff/rb_cost*100:+.1f}%)")

    print("\n" + "="*80)
    print("ANÁLISE 5: TIMING DAS DECISÕES")
    print("="*80)

    # Analyze by hour of day
    merged['hour'] = merged['timestamp'].dt.hour

    print("\nCusto médio por hora do dia:")
    hourly_analysis = merged.groupby('hour').agg({
        'net_cost_sdp': 'mean',
        'net_cost_rb': 'mean',
        'cost_diff': 'mean'
    }).round(4)

    print("\nHoras com maior diferença de custo (SDP pior):")
    worst_hours = hourly_analysis.sort_values('cost_diff', ascending=False).head(5)
    print(worst_hours)

    print("\n" + "="*80)
    print("ANÁLISE 6: UTILIZAÇÃO DA BATERIA")
    print("="*80)

    # Battery utilization patterns
    print("\nPadrão de utilização da bateria:")

    sdp_charge_total = merged[merged['battery_power_sdp'] > 0]['battery_power_sdp'].sum() * dt_hours
    rb_charge_total = merged[merged['battery_power_rb'] > 0]['battery_power_rb'].sum() * dt_hours

    sdp_discharge_total = -merged[merged['battery_power_sdp'] < 0]['battery_power_sdp'].sum() * dt_hours
    rb_discharge_total = -merged[merged['battery_power_rb'] < 0]['battery_power_rb'].sum() * dt_hours

    print(f"\nTotal carregado:")
    print(f"  SDP:        {sdp_charge_total:.2f} kWh")
    print(f"  Rule-Based: {rb_charge_total:.2f} kWh")
    print(f"  Diferença:  {sdp_charge_total - rb_charge_total:.2f} kWh")

    print(f"\nTotal descarregado:")
    print(f"  SDP:        {sdp_discharge_total:.2f} kWh")
    print(f"  Rule-Based: {rb_discharge_total:.2f} kWh")
    print(f"  Diferença:  {sdp_discharge_total - rb_discharge_total:.2f} kWh")

    # Check if SDP charges from grid
    sdp_charging_from_grid = merged[
        (merged['battery_power_sdp'] > 0) &  # Battery charging
        (merged['solar_power_sdp'] < merged['load_power_sdp'])  # Net demand (not excess solar)
    ]

    rb_charging_from_grid = merged[
        (merged['battery_power_rb'] > 0) &
        (merged['solar_power_rb'] < merged['load_power_rb'])
    ]

    print(f"\n⚠️  Carga da bateria DURANTE demanda (potencialmente da rede):")
    print(f"  SDP:        {len(sdp_charging_from_grid)} timesteps")
    print(f"  Rule-Based: {len(rb_charging_from_grid)} timesteps")

    if len(sdp_charging_from_grid) > 0:
        print(f"\n  ❌ SDP carrega da rede (decisão potencialmente má):")
        print(f"     Total carregado: {sdp_charging_from_grid['battery_power_sdp'].sum() * dt_hours:.2f} kWh")
        print(f"     Custo estimado: {sdp_charging_from_grid['battery_power_sdp'].sum() * dt_hours * tariff_price:.2f} €")

    print("\n" + "="*80)
    print("ANÁLISE 7: RESTRIÇÕES DO SISTEMA")
    print("="*80)

    # Check constraint violations
    print("\nVerificar constraint: Bateria não pode descarregar para exportar")

    # Periods where battery discharges AND there's export
    sdp_violation = merged[
        (merged['battery_power_sdp'] < 0) &  # Battery discharging
        (merged['grid_export_sdp'] > 0.01)   # Exporting to grid
    ]

    rb_violation = merged[
        (merged['battery_power_rb'] < 0) &
        (merged['grid_export_rb'] > 0.01)
    ]

    print(f"\nDescarga durante exportação (pode violar constraint):")
    print(f"  SDP:        {len(sdp_violation)} timesteps")
    print(f"  Rule-Based: {len(rb_violation)} timesteps")

    if len(sdp_violation) > 0:
        print(f"\n  ⚠️  SDP viola constraint em {len(sdp_violation)} momentos")
        print(f"     Energia exportada da bateria: {(-sdp_violation['battery_power_sdp']).sum() * dt_hours:.2f} kWh")

    print("\n" + "="*80)
    print("CONCLUSÕES PRELIMINARES")
    print("="*80)

    print("\n1. ANÁLISE DE CUSTOS:")
    total_cost_diff = sdp_summary['total_cost'] - rb_summary['total_cost']
    print(f"   Diferença total: {total_cost_diff:.2f} €")

    import_diff = (sdp_summary['total_grid_import_kwh'] - rb_summary['total_grid_import_kwh']) * tariff_price
    export_diff = (sdp_summary['total_grid_export_kwh'] - rb_summary['total_grid_export_kwh']) * export_price

    print(f"   Importação extra: {import_diff:.2f} €")
    print(f"   Exportação extra: {export_diff:.2f} € (receita perdida)")

    print("\n2. PRINCIPAIS PROBLEMAS IDENTIFICADOS:")
    issues = []

    if len(sdp_not_charging) > len(rb_not_charging):
        issues.append("SDP não carrega durante excesso solar")

    if len(sdp_charging_from_grid) > len(rb_charging_from_grid):
        issues.append("SDP carrega da rede durante demanda")

    if sdp_discharge_total < rb_discharge_total:
        issues.append("SDP descarrega menos que deveria")

    if len(issues) > 0:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("   Nenhum problema óbvio identificado")

    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Plot 1: Cost difference over time
    axes[0].plot(merged['timestamp'], merged['cost_diff'], alpha=0.7, linewidth=0.8)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].fill_between(merged['timestamp'], 0, merged['cost_diff'],
                         where=merged['cost_diff'] > 0, alpha=0.3, color='red', label='SDP pior')
    axes[0].fill_between(merged['timestamp'], 0, merged['cost_diff'],
                         where=merged['cost_diff'] < 0, alpha=0.3, color='green', label='SDP melhor')
    axes[0].set_ylabel('Cost Diff (€)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Diferença de Custo por Timestep (SDP - Rule-Based)')

    # Plot 2: Battery power comparison
    axes[1].plot(merged['timestamp'], merged['battery_power_sdp'], label='SDP', alpha=0.7, linewidth=1)
    axes[1].plot(merged['timestamp'], merged['battery_power_rb'], label='Rule-Based', alpha=0.7, linewidth=1)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Battery Power (kW)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Decisões de Bateria')

    # Plot 3: SOC comparison
    axes[2].plot(merged['timestamp'], merged['battery_soc_sdp'], label='SDP', alpha=0.7, linewidth=1)
    axes[2].plot(merged['timestamp'], merged['battery_soc_rb'], label='Rule-Based', alpha=0.7, linewidth=1)
    axes[2].set_ylabel('SOC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Trajetória do SOC')

    # Plot 4: Grid power
    axes[3].plot(merged['timestamp'], merged['grid_power_sdp'], label='SDP', alpha=0.7, linewidth=1)
    axes[3].plot(merged['timestamp'], merged['grid_power_rb'], label='Rule-Based', alpha=0.7, linewidth=1)
    axes[3].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[3].set_ylabel('Grid Power (kW)')
    axes[3].set_xlabel('Time')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title('Potência da Rede (+ = import, - = export)')

    plt.tight_layout()
    output_file = Path("results/plots/deep_decision_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")

    return merged, worst_decisions

if __name__ == "__main__":
    merged, worst = deep_analyze_decisions()
