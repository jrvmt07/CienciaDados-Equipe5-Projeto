# Ciência de Dados – Equipe 5 (CIn/UFPE)

Projeto da disciplina CIN0208 (profa. Juscimara G. Avelino): **classificar a mesorregião de Pernambuco a partir da chuva mensal (dados APAC) com KNN**.

**Equipe:** Mateus Henrique, Gabriel Bueno, Randell Almeida, José Ricardo Torres, Eric Nunes

## Dados
- `data/raw/Dataset APAC - Página1.csv` — 5.104 registros (posto × ano, 2010–2025), 12 colunas de chuva mensal (mm).
- Alvo: `Mesorregiao` (5 classes), derivada do município do posto (`src/meso_map.py`).
- Features: os 12 totais mensais (**12 atributos**). Valores "-" = sem medição (NaN), tratados dentro do Pipeline.

## Estrutura
```
data/raw/        dados brutos (não editar)
docs/            decisões, divisão de temas e issues detalhadas
notebooks/       01_EDA_Checkpoint (EDA)
                 02_Modelagem_KNN (protocolo principal, 360 experimentos)
                 03_Experimentos_Extras (imputar 0, outliers, escalas, seleção, PCA, balanceamento)
src/             data.py, meso_map.py, experimentos.py
results/         experimentos_knn.csv, matriz_confusao, instancias_incorretas.csv, relatorio_melhor.txt
```
## Como rodar
```bash
pip install -r requirements.txt
python src/experimentos.py            # 360 experimentos + matriz de confusão -> results/
```

## Decisões de método
- **Pipeline** (imputação → normalização → KNN): nada é ajustado fora do treino.
- **StratifiedGroupKFold (5) agrupando por posto**: o mesmo posto em anos diferentes não aparece em treino e teste (evita vazamento).
- Classes desbalanceadas (Sertão ~32%, São Francisco ~8%) → métrica principal **F1 macro** + balanced accuracy.
- Grade: 2 imputações × 3 scalers × 5 valores de k × 2 pesos × 3 distâncias × 2 limites de ausentes (sem limite ou ≤ 6 meses vazios) = **360 experimentos**.

## Resultados

Validação por posto (StratifiedGroupKFold, 5 folds, seed 42). Baseline (Dummy): acurácia 0,33 e F1 macro 0,10.

| Modelo | F1 macro | Acurácia | F1 São Francisco |
|---|---|---|---|
| Melhor KNN (KNNImputer, ≤6 meses ausentes, StandardScaler, k=3, distance, manhattan) | 0,726 | 0,748 | 0,587 |

Extras testados no notebook 03: imputar "-" como zero (F1 0,745, hipótese ainda sem confirmação da APAC), indicador de ausência (0,680), seleção de atributos e PCA (sem ganho) e balanceamento (aumenta o recall do São Francisco, mas reduz o F1 macro).

Os erros se concentram entre regiões vizinhas: Agreste e Mata, Agreste e Sertão, Sertão e São Francisco.

## Documentação
- `docs/decisoes.md` — decisões de método, resultados registrados e pendências.
- Tarefas: aba **Issues** do repositório (cada uma com responsável e prazo).

## Como contribuir
1. Pegar uma issue e trabalhar em branch própria: `git checkout -b nome/tarefa`.
2. `git pull` antes de começar; commits pequenos e Pull Request para `main`, revisado por outra pessoa.
3. Notebooks: rodar "Restart & run all" antes de commitar.
4. Seed fixa em 42 em tudo que for aleatório; não editar `data/raw/`.
5. Resultados novos em `results/` com nome claro; decisões em `docs/decisoes.md`.
6. Sem Git instalado: dá para rodar tudo no Colab (`!git clone <url>`, `!pip install -r requirements.txt`, `!python src/experimentos.py`).
