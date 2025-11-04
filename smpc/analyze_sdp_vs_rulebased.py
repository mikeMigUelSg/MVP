"""
Análise detalhada da diferença entre SDP e Rule-Based

Este script identifica os problemas principais:
1. SDP não inclui custo de degradação da bateria na função de custo
2. SDP pode estar a sobre-utilizar a bateria sem penalização adequada
3. Comparação de parâmetros e comportamento
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def analyze_controllers():
    """Analisa e compara os resultados dos controladores"""

    print("="*80)
    print("ANÁLISE: SDP vs RULE-BASED")
    print("="*80)

    # Load results
    results_dir = Path("results/data")

    try:
        sdp_results = pd.read_csv(results_dir / "results_sdp (numba jit).csv")
        rb_results = pd.read_csv(results_dir / "results_rule-based.csv")

        with open(results_dir / "summary_sdp (numba jit).yaml") as f:
            sdp_summary = yaml.load(f, Loader=yaml.UnsafeLoader)
        with open(results_dir / "summary_rule-based.yaml") as f:
            rb_summary = yaml.load(f, Loader=yaml.UnsafeLoader)
    except FileNotFoundError as e:
        print(f"Erro: Ficheiros de resultados não encontrados: {e}")
        print("\nExecute primeiro as simulações com ambos os controladores.")
        return

    print("\n" + "="*80)
    print("PROBLEMA 1: CUSTO DE DEGRADAÇÃO NÃO INCLUÍDO NO SDP")
    print("="*80)

    print("\nO SDP tem uma FALHA CRÍTICA na função de custo:")
    print("❌ A função stage_cost_jit() (linha 78-85 em sdp_controller.py) NÃO inclui")
    print("   o custo de degradação da bateria!")
    print("\n➤ Função atual:")
    print("""
    def stage_cost_jit(u, R, P_lim, c_s, c_f, dt_hours):
        P_g = grid_power_jit(u, R, P_lim)
        P_g_plus = max(0.0, P_g)
        P_g_minus = max(0.0, -P_g)
        cost = (P_g_plus * c_s - P_g_minus * c_f) * dt_hours
        return cost  # ❌ Falta custo de degradação!
    """)

    print("\n➤ Deveria ser:")
    print("""
    def stage_cost_jit(u, R, P_lim, c_s, c_f, dt_hours,
                       C_bat, eta_c, eta_d, c_deg):
        # Custo da rede
        P_g = grid_power_jit(u, R, P_lim)
        P_g_plus = max(0.0, P_g)
        P_g_minus = max(0.0, -P_g)
        grid_cost = (P_g_plus * c_s - P_g_minus * c_f) * dt_hours

        # ✅ Custo de degradação da bateria
        p_eff = abs(p_eff_jit(u, eta_c, eta_d))
        degradation_cost = p_eff * dt_hours * c_deg

        return grid_cost + degradation_cost
    """)

    print("\n" + "="*80)
    print("PROBLEMA 2: CONSEQUÊNCIAS DA FALTA DE PENALIZAÇÃO")
    print("="*80)

    dt_hours = 0.25

    # Battery usage
    sdp_charged = sdp_results[sdp_results['battery_power'] > 0]['battery_power'].sum() * dt_hours
    sdp_discharged = -sdp_results[sdp_results['battery_power'] < 0]['battery_power'].sum() * dt_hours

    rb_charged = rb_results[rb_results['battery_power'] > 0]['battery_power'].sum() * dt_hours
    rb_discharged = -rb_results[rb_results['battery_power'] < 0]['battery_power'].sum() * dt_hours

    print(f"\nUtilização da Bateria:")
    print(f"  SDP:        Carregada: {sdp_charged:.2f} kWh, Descarregada: {sdp_discharged:.2f} kWh")
    print(f"  Rule-Based: Carregada: {rb_charged:.2f} kWh, Descarregada: {rb_discharged:.2f} kWh")

    print(f"\nCiclos da Bateria (capacity = 10 kWh):")
    sdp_cycles = (sdp_charged + sdp_discharged) / 20
    rb_cycles = (rb_charged + rb_discharged) / 20
    print(f"  SDP:        {sdp_cycles:.2f} ciclos")
    print(f"  Rule-Based: {rb_cycles:.2f} ciclos")

    print(f"\nCusto de Degradação:")
    print(f"  SDP:        {sdp_summary['battery_degradation_cost']:.2f} €")
    print(f"  Rule-Based: {rb_summary['battery_degradation_cost']:.2f} €")

    print("\n⚠️  O SDP usa MENOS a bateria que o rule-based, MAS com piores resultados!")
    print("    Isto sugere que o SDP está a tomar decisões SUB-ÓTIMAS porque:")
    print("    - Não penaliza o uso da bateria corretamente")
    print("    - Pode estar a carregar/descarregar em momentos errados")
    print("    - Não considera o custo total real do sistema")

    print("\n" + "="*80)
    print("PROBLEMA 3: DIFERENÇAS NA GESTÃO DE ENERGIA")
    print("="*80)

    print(f"\nImportação/Exportação:")
    print(f"  SDP:        Import: {sdp_summary['total_grid_import_kwh']:.2f} kWh, Export: {sdp_summary['total_grid_export_kwh']:.2f} kWh")
    print(f"  Rule-Based: Import: {rb_summary['total_grid_import_kwh']:.2f} kWh, Export: {rb_summary['total_grid_export_kwh']:.2f} kWh")

    sdp_export_excess = sdp_summary['total_grid_export_kwh'] - rb_summary['total_grid_export_kwh']
    sdp_import_excess = sdp_summary['total_grid_import_kwh'] - rb_summary['total_grid_import_kwh']

    print(f"\n❌ SDP exporta MAIS {sdp_export_excess:.2f} kWh do que deveria")
    print(f"❌ SDP importa MAIS {sdp_import_excess:.2f} kWh do que deveria")

    print("\nIsto significa que o SDP:")
    print("  1. Não está a aproveitar bem a energia solar")
    print("  2. Está a deixar escapar energia valiosa para a rede (a preço baixo)")
    print("  3. Está a comprar mais energia da rede (a preço alto)")

    print("\n" + "="*80)
    print("PROBLEMA 4: ANÁLISE DE CUSTOS")
    print("="*80)

    print(f"\nCusto Total:")
    print(f"  SDP:        {sdp_summary['total_cost']:.2f} €")
    print(f"  Rule-Based: {rb_summary['total_cost']:.2f} €")
    print(f"  Diferença:  {sdp_summary['total_cost'] - rb_summary['total_cost']:.2f} € ({(sdp_summary['total_cost'] - rb_summary['total_cost'])/rb_summary['total_cost']*100:.1f}% PIOR)")

    # Break down the cost difference
    tariff_price = 0.15  # €/kWh
    export_price = 0.045  # €/kWh

    # Cost from import/export difference
    import_cost_diff = sdp_import_excess * tariff_price
    export_revenue_diff = sdp_export_excess * export_price
    grid_cost_diff = import_cost_diff + export_revenue_diff

    deg_cost_diff = sdp_summary['battery_degradation_cost'] - rb_summary['battery_degradation_cost']

    print(f"\nDecomposição da diferença de custo:")
    print(f"  Importação extra:      +{import_cost_diff:.2f} €")
    print(f"  Exportação extra:      +{export_revenue_diff:.2f} € (receita perdida)")
    print(f"  Degradação:            {deg_cost_diff:+.2f} €")
    print(f"  Total explicado:       ~{grid_cost_diff + deg_cost_diff:.2f} €")

    print("\n" + "="*80)
    print("PROBLEMA 5: PARÂMETROS DO SDP")
    print("="*80)

    print("\nLer config.yaml para verificar parâmetros...")
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        sdp_config = config['controller']['sdp']

        print(f"\nParâmetros do SDP:")
        print(f"  Horizonte:           {sdp_config['horizon_steps']} passos = {sdp_config['horizon_steps']*15/60:.1f}h")
        print(f"  SOC target:          {sdp_config['soc_target']}")
        print(f"  Terminal cost weight: {sdp_config['terminal_cost_weight']}")
        print(f"  Half-life:           {sdp_config['half_life_minutes']} min")
        print(f"  Sigma:               {sdp_config['sigma_residual_kw']} kW")

        print("\n⚠️  O terminal cost pode estar a forçar o SDP a atingir SOC=0.5")
        print("    no fim do horizonte, o que pode levar a decisões sub-ótimas")

    except Exception as e:
        print(f"Erro ao ler config: {e}")

    print("\n" + "="*80)
    print("SOLUÇÕES RECOMENDADAS")
    print("="*80)

    print("\n1. CRÍTICO: Adicionar custo de degradação à função stage_cost_jit()")
    print("   ➤ Modificar src/controllers/sdp_controller.py")
    print("   ➤ Incluir parâmetro de degradação e calcular custo por kWh throughput")

    print("\n2. IMPORTANTE: Ajustar terminal cost")
    print("   ➤ Reduzir terminal_cost_weight ou")
    print("   ➤ Tornar soc_target mais flexível (ex: range ao invés de valor fixo)")

    print("\n3. VERIFICAR: Parâmetros do modelo de resíduo")
    print("   ➤ half_life_minutes pode estar muito alto")
    print("   ➤ sigma_residual_kw pode não refletir a incerteza real")

    print("\n4. TESTAR: Horizonte mais longo")
    print("   ➤ Aumentar horizon_steps para 96 (24h) pode melhorar decisões")

    print("\n" + "="*80)
    print("PRÓXIMOS PASSOS")
    print("="*80)

    print("\n1. Corrigir a função stage_cost_jit() para incluir degradação")
    print("2. Re-executar simulação e comparar resultados")
    print("3. Se ainda houver problemas, calibrar parâmetros do modelo")

    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_controllers()
