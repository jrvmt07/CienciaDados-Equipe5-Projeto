"""Roda a grade de experimentos KNN (>= 150 combinações) e avalia o melhor modelo.

Uso:  python src/experimentos.py --csv "data/raw/Dataset APAC - Página1.csv"

Saídas em results/:
  experimentos_knn.csv      todos os experimentos, ordenados por F1 macro
  relatorio_melhor.txt      precision / recall / F1 por mesorregião do melhor modelo
  matriz_confusao.csv/.png  matriz de confusão do melhor modelo
"""
import argparse
import itertools
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (StratifiedGroupKFold, cross_val_predict,
                                     cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from data import MESES, carregar, X_y_grupos

IMPUTADORES = {
    'mediana': lambda: SimpleImputer(strategy='median'),
    'knn_imputer': lambda: KNNImputer(n_neighbors=5),
}
SCALERS = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
KS = [1, 3, 5, 7, 9]  # amplie se quiser: 11, 15, 21...
PESOS = ['uniform', 'distance']
METRICAS = ['euclidean', 'manhattan', 'chebyshev']
# Máximo de meses vazios tolerados por linha (None = não descarta nada).
# Linhas com mais meses vazios que o limite são descartadas antes de imputar.
MAX_AUSENTES = [None, 6]


def filtrar(X, y, g, max_aus):
    if max_aus is None:
        return X, y, g
    ok = X.isna().sum(axis=1) <= max_aus
    return X[ok], y[ok], g[ok]


def montar_pipe(imp, sc, k, w, met):
    return Pipeline([
        ('imputer', IMPUTADORES[imp]()),   # tudo dentro do Pipeline: sem vazamento
        ('scaler', SCALERS[sc]()),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights=w, metric=met)),
    ])


def avaliar_melhor(melhor, X0, y0, g0, cv, pasta):
    X, y, g = filtrar(X0, y0, g0, melhor['max_ausentes'])
    pipe = montar_pipe(melhor['imputador'], melhor['scaler'], int(melhor['k']),
                       melhor['pesos'], melhor['metrica'])
    pred = cross_val_predict(pipe, X, y, groups=g, cv=cv, n_jobs=-1)
    classes = sorted(y.unique())
    rel = classification_report(y, pred, labels=classes, digits=3)
    cm = confusion_matrix(y, pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(os.path.join(pasta, 'matriz_confusao.csv'))
    with open(os.path.join(pasta, 'relatorio_melhor.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Melhor configuração:\n{melhor.to_string()}\n\nInstâncias usadas: {len(X)}\n\n{rel}")
    print(f"\nInstâncias usadas no melhor modelo: {len(X)}\n{rel}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title('Matriz de confusão (contagem)')
    sns.heatmap(cm_df.div(cm_df.sum(axis=1), axis=0), annot=True, fmt='.2f',
                cmap='Blues', ax=ax[1])
    ax[1].set_title('Normalizada por classe real (recall)')
    for a in ax:
        a.set_xlabel('Previsto'); a.set_ylabel('Real')
        a.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta, 'matriz_confusao.png'), dpi=150)


def main(csv, saida, folds, seed):
    df = carregar(csv)
    X0, y0, g0 = X_y_grupos(df)
    print(f"Instâncias: {len(X0)} | Atributos: {X0.shape[1]} | Classes: {y0.nunique()}")
    print(y0.value_counts(normalize=True).round(3).to_string())
    print("Meses ausentes por mês:\n", X0.isna().sum().to_string())
    print("Linhas por nº de meses ausentes:\n",
          X0.isna().sum(axis=1).value_counts().sort_index().to_string())

    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    combos = list(itertools.product(MAX_AUSENTES, IMPUTADORES, SCALERS, KS, PESOS, METRICAS))
    print(f"\nTotal de experimentos: {len(combos)}")

    linhas, cache = [], {}
    for i, (ma, imp, sc, k, w, met) in enumerate(combos, 1):
        if ma not in cache:
            cache[ma] = filtrar(X0, y0, g0, ma)
        X, y, g = cache[ma]
        t0 = time.time()
        r = cross_validate(montar_pipe(imp, sc, k, w, met), X, y, groups=g, cv=cv,
                           n_jobs=-1, scoring=['accuracy', 'balanced_accuracy', 'f1_macro'])
        linhas.append(dict(
            id=i, max_ausentes=ma, n_instancias=len(X), imputador=imp, scaler=sc, k=k,
            pesos=w, metrica=met,
            acc=r['test_accuracy'].mean(), acc_std=r['test_accuracy'].std(),
            bal_acc=r['test_balanced_accuracy'].mean(),
            f1_macro=r['test_f1_macro'].mean(), f1_std=r['test_f1_macro'].std(),
            seg=round(time.time() - t0, 2)))
        print(f"[{i}/{len(combos)}] aus<={ma} {imp} {sc} k={k} {w} {met} -> "
              f"f1_macro={linhas[-1]['f1_macro']:.3f}", flush=True)

    res = pd.DataFrame(linhas).sort_values('f1_macro', ascending=False)
    pasta = os.path.dirname(saida) or '.'
    os.makedirs(pasta, exist_ok=True)
    res.to_csv(saida, index=False)
    print(f"\nSalvo em {saida}\nTop 5:\n{res.head(5).to_string(index=False)}")

    melhor = res.iloc[0].copy()
    melhor['max_ausentes'] = None if pd.isna(melhor['max_ausentes']) else int(melhor['max_ausentes'])
    avaliar_melhor(melhor, X0, y0, g0, cv, pasta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/raw/Dataset APAC - Página1.csv')
    ap.add_argument('--saida', default='results/experimentos_knn.csv')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    main(a.csv, a.saida, a.folds, a.seed)
