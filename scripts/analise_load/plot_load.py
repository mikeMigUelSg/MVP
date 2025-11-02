import pandas as pd
import matplotlib.pyplot as plt

# ========== VARIÁVEIS HARD CODED ==========
DATA_PATH = 'data/merged_consumos.xlsx'
CONSUMO_COLUMN = 'Consumo registado (kW)'
DATE_BEGIN = '2025-01-01'
DATE_END = '2025-01-31'
# ==========================================

def plot_consumos():
    """
    Carrega e faz plot de todos os consumos no período especificado
    """
    print(f"A carregar dados de {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)

    # Criar coluna de datetime combinando Data e Hora
    df['DateTime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'])

    # Filtrar por período
    date_begin = pd.to_datetime(DATE_BEGIN)
    date_end = pd.to_datetime(DATE_END)
    df = df[(df['DateTime'] >= date_begin) & (df['DateTime'] <= date_end)]

    # Ordenar por data/hora
    df = df.sort_values('DateTime')

    print(f"\nPeríodo: {DATE_BEGIN} a {DATE_END}")
    print(f"Total de registos: {len(df)}")

    # Criar o plot
    plt.figure(figsize=(14, 6))

    plt.plot(df['DateTime'], df[CONSUMO_COLUMN], linewidth=1, alpha=0.7)

    # Adicionar linhas verticais a cada segunda-feira às 00:00
    date_range = pd.date_range(start=date_begin, end=date_end, freq='D')
    mondays = [d for d in date_range if d.weekday() == 0]  # 0 = segunda-feira
    for monday in mondays:
        plt.axvline(x=monday, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    plt.xlabel('Data/Hora')
    plt.ylabel('Consumo (kW)')
    plt.title(f'Consumos ao longo do tempo ({DATE_BEGIN} a {DATE_END})')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Estatísticas
    print(f"\n=== Estatísticas ===")
    print(f"Média: {df[CONSUMO_COLUMN].mean():.2f} kW")
    print(f"Máximo: {df[CONSUMO_COLUMN].max():.2f} kW")
    print(f"Mínimo: {df[CONSUMO_COLUMN].min():.2f} kW")

    plt.show()

if __name__ == '__main__':
    plot_consumos()
