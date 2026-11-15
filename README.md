# Ciência de Dados – Equipe 5 (CIn/UFPE)

Projeto da disciplina CIN0208 (profa. Juscimara G. Avelino): **classificar a mesorregião de Pernambuco a partir da chuva mensal (dados APAC) com KNN**.

**Equipe:** Mateus Henrique, Gabriel Bueno, Randell Almeida, José Ricardo Torres, Eric Nunes

## Dados
- `data/raw/Dataset APAC - Página1.csv` — 5.104 registros (posto × ano, 2010–2025), 12 colunas de chuva mensal (mm).
- Alvo: `Mesorregiao` (5 classes), derivada do município do posto (`src/meso_map.py`).
- Features: os 12 totais mensais (**12 atributos**). Valores "-" = sem medição (NaN), tratados dentro do Pipeline.

## Estrutura
```
data/raw/         dados brutos (não editar)
notebooks/        01_EDA_Checkpoint.ipynb (EDA)
src/              data.py (carga/preparo), meso_map.py, experimentos.py
results/          experimentos_knn.csv (saída)
relatorio/        relatório e slides
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
