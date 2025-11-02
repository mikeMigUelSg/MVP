import pandas as pd
import os

# ========== VARIÁVEIS HARD CODED ==========
# Caminho do ficheiro Excel de entrada
INPUT_XLSX = '../data/merged_consumos.xlsx'

# Colunas a utilizar
DATE_COLUMN = 'Data'
TIME_COLUMN = 'Hora'
TARGET_COLUMN = 'Consumo registado (kW)'

# Período para filtrar
DATE_BEGIN = '2025-04-01'
DATE_END = '2025-07-01' 

# Caminho de saída
OUTPUT_CSV = 'data/data.csv'
# ==========================================

def preprocess_data():
    """
    Preprocessa os dados do Excel para formato CSV adequado ao Autoformer
    """
    print(f"A carregar dados de {INPUT_XLSX}...")
    df = pd.read_excel(INPUT_XLSX)

    print(f"Total de registos originais: {len(df)}")

    # Criar coluna DateTime combinando Data e Hora
    df['DateTime'] = pd.to_datetime(df[DATE_COLUMN] + ' ' + df[TIME_COLUMN])

    # Filtrar por período
    date_begin = pd.to_datetime(DATE_BEGIN)
    date_end = pd.to_datetime(DATE_END)
    df = df[(df['DateTime'] >= date_begin) & (df['DateTime'] <= date_end)]

    print(f"Registos após filtro de datas ({DATE_BEGIN} a {DATE_END}): {len(df)}")

    # Ordenar por data/hora
    df = df.sort_values('DateTime')

    # Selecionar apenas as colunas necessárias e renomear
    df_output = pd.DataFrame({
        'date': df['DateTime'],
        'target': df[TARGET_COLUMN]
    })

    # Criar diretório de saída se não existir
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Guardar como CSV
    df_output.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDados guardados em: {OUTPUT_CSV}")
    print(f"\n=== Estatísticas ===")
    print(f"Total de registos: {len(df_output)}")
    print(f"Média: {df_output['target'].mean():.2f} kW")
    print(f"Máximo: {df_output['target'].max():.2f} kW")
    print(f"Mínimo: {df_output['target'].min():.2f} kW")
    print(f"Período: {df_output['date'].min()} a {df_output['date'].max()}")

    # Verificar dados em falta
    missing = df_output.isnull().sum()
    if missing.any():
        print(f"\n⚠ Atenção: Dados em falta detectados:")
        print(missing[missing > 0])
    else:
        print("\n✓ Sem dados em falta")

if __name__ == '__main__':
    preprocess_data()
