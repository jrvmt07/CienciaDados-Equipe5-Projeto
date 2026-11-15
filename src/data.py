"""Carga e preparo dos dados da APAC (formato largo: 1 linha = posto x ano)."""
import re
import numpy as np
import pandas as pd
from meso_map import meso_map

MESES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
         'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']


def extrair_cidade(posto):
    clean = re.sub(r'\(.*?\)', '', str(posto))
    clean = re.sub(r'-\s*PCD', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'-\s*APAC', '', clean, flags=re.IGNORECASE)
    return clean.split('-')[0].strip()


def carregar(caminho="data/raw/Dataset APAC - Página1.csv"):
    """Lê o CSV bruto, cria Cidade/Mesorregiao e converte meses para número.
    Valores '-' (sem medição) viram NaN."""
    df = pd.read_csv(caminho)
    df['Cidade'] = df['Posto'].apply(extrair_cidade)
    df['Mesorregiao'] = df['Cidade'].map(meso_map)
    for m in MESES:
        df[m] = (df[m].astype(str).str.strip().str.replace(',', '.'))
        df[m] = pd.to_numeric(df[m], errors='coerce')
    return df


def X_y_grupos(df):
    """Features = 12 totais mensais; alvo = mesorregião; grupo = posto (Código).
    Agrupar por posto evita vazamento: o mesmo posto em anos diferentes
    não pode estar no treino e no teste ao mesmo tempo."""
    sem_meso = df['Mesorregiao'].isna().sum()
    if sem_meso:
        print(f"ATENÇÃO: {sem_meso} linhas sem mesorregião (cidades fora do dicionário)")
        df = df.dropna(subset=['Mesorregiao'])
    return df[MESES].copy(), df['Mesorregiao'].copy(), df['Código'].copy()
