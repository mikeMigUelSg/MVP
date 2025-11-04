#!/usr/bin/env python3
"""
Script para testar diferentes configurações e comparar resultados
"""
import yaml
import subprocess
import pandas as pd

configs_to_test = {
    "original": {
        "battery": {"degradation_cost_per_kwh": 0.01},
        "controller": {"mpc": {"export_price": 0.05}}
    },
    "fix1_low_degradation": {
        "battery": {"degradation_cost_per_kwh": 0.002},  # Reduzir 5x
        "controller": {"mpc": {"export_price": 0.05}}
    },
    "fix2_realistic_export": {
        "battery": {"degradation_cost_per_kwh": 0.01},
        "controller": {"mpc": {"export_price": 0.10}}  # Dobrar export price
    },
    "fix3_both": {
        "battery": {"degradation_cost_per_kwh": 0.002},
        "controller": {"mpc": {"export_price": 0.10}}
    }
}

print("=" * 80)
print("TESTE DE DIFERENTES CONFIGURAÇÕES")
print("=" * 80)

for config_name, changes in configs_to_test.items():
    print(f"\n🧪 Testando: {config_name}")
    print(f"   Degradation: {changes['battery']['degradation_cost_per_kwh']} €/kWh")
    print(f"   Export price: {changes['controller']['mpc']['export_price']} €/kWh")
    
print("\n💡 Para executar os testes, rode:")
print("   python3 test_configs.py run")

