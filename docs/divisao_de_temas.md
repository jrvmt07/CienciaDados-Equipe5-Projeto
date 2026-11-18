# Divisão por temas da disciplina

## 0. O que a professora pediu

A professora disse para **seguir o que vimos nas aulas e o que está no documento do trabalho** ("Projeto - Ciência de Dados"). Checklist tirado do documento:

| Pedido do documento | Onde está no projeto |
|---|---|
| 10 atributos ou mais | 12 atributos (os 12 meses) |
| 1000 instâncias ou mais | 5.104 (base completa), 3.337 (melhor modelo) |
| 150 experimentos | 360 (grade: limite de ausentes, imputação, scaler, k, peso, distância) |
| Aprender o KNN e as bibliotecas (Pandas, NumPy, Matplotlib, seaborn, scikit-learn) | notebook `02_Modelagem_KNN` |
| Limpar, uniformizar e prever da melhor forma possível | seções 1 a 4 do notebook 02 |
| Analisar os dados para ver os melhores métodos | seções 5 e 6 (grade e resultados) |
| Roteiro de inspeção: `info()`, `describe()`, `isnull().sum()` | seção 1 do notebook 02 |
| Busca em grade de `n_neighbors`, `weights`, `metric` (o exemplo usa `GridSearchCV`) | nossa grade faz o mesmo com `cross_validate` (360 combinações) |

## 1. Visão geral

| Tema | Aula | Responsável | Issues | Notebook | Pitch (3 min cada) |
|---|---|---|---|---|---|
| Problema, dados e EDA; análise dos 360 experimentos | 1, 4 | Eric | #6 | `01_EDA_Checkpoint` (base de Mateus) e `06_Analise_Experimentos` | 1º: problema, dados e EDA |
| Ausentes (o que significa "-") e slides | 5 | Ricardo | #5, #11 | `05_Ausentes` | 2º: limpeza |
| Mapeamento, valores incorretos, duplicatas, outliers | 5 | Eric | #4, #13 | `05_Limpeza` | (com o 1º) |
| Encoding e balanceamento | 6, 8 | Bueno | #7 | `03_Balanceamento` (seção de encoding dentro) | 3º: encoding e balanceamento |
| Normalização e seleção de features / PCA | 6, 7 | Mateus Henrique | #8 | `04_Selecao_Features` (seção de normalização dentro) | 4º: normalização e seleção |
| KNN, pipeline, validação, resultados, relatório | 2, 3 | Randell | #9, #10, #12 | `02_Modelagem_KNN` | 5º: modelo, resultados e conclusão |

Total: 15 minutos, todos falam.

## 2. O que cada tema precisa mostrar

### A. KNN, pipeline e avaliação (Aulas 2 e 3) · Randell
- **Já existe:** Pipeline (imputar → normalizar → KNN), validação por posto (`StratifiedGroupKFold`), F1 macro, matriz de confusão, grade de 360 experimentos.
- **Falta:** baseline da classe majoritária (`DummyClassifier`; Aula 3: "compare sempre com o baseline"). Na base de 3.337 linhas, sempre dizer "Agreste" dá ~33% de acurácia. Reportar sempre média ± desvio.
- **Pergunta provável:** por que não avaliar no treino? O que acontece com k = 1? Por que F1 macro e não acurácia?

### B. EDA e dados (Aulas 1 e 4) · Eric
- **Já existe:** EDA do Mateus (sazonalidade, ausências, boxplots) e o `main.py` do Bueno (gráficos melhores).
- **Falta:** distribuição das classes, boxplots por mesorregião (não só julho × dezembro), resumo estatístico, e uma frase para cada gráfico: o que a EDA mudou nas decisões (31% de linhas vazias; sazonalidade; desbalanceamento).
- **Pergunta provável:** o que a EDA revelou que você não esperava?

### C. Limpeza (Aula 5) · Ricardo (ausentes) e Eric (incorretos, duplicatas, outliers)
- **Ausentes:** analisar o padrão (por mês, por posto, por ano) e discutir o mecanismo (MCAR, MAR ou MNAR). Confirmar se "-" é "sem medição".
  **Indício importante (verificado nos dados):** (a) não existe nenhum valor igual a 0 em todo o dataset (o mínimo é 0,1 mm); (b) nas linhas parcialmente vazias, os "-" se concentram nos meses secos e seguem o clima de cada região: em setembro, 76% no Sertão contra 19% no Agreste; em novembro, o contrário (Agreste 54% contra Sertão 41%); (c) as 1.562 linhas totalmente vazias são ~100 por ano, ou seja, postos sem relatório. Hipótese: nas linhas parciais, "-" provavelmente significa **chuva zero** (ausência que depende do próprio valor, MNAR, Aula 5), e nas linhas 100% vazias é posto sem dados. Testar imputar **zero** nas linhas parciais (`SimpleImputer(strategy="constant", fill_value=0)`) e comparar com mediana e KNNImputer. Confirmar com a APAC. Testar limites de 4, 6, 8 e 10 meses vazios. Testar a contagem de meses vazios como atributo (Aula 5: "a ausência pode ser o sinal").
  **Cuidado:** imputar usando a mesorregião é **vazamento** (o rótulo não entra na imputação).
- **Valores incorretos:** negativos ou valores absurdos (por exemplo, acima de ~1.000 mm em um mês).
- **Duplicatas e redundância:** linhas repetidas (`Código` + `Ano`); vários postos na mesma cidade.
- **Outliers e ruído:** IQR por mês e mesorregião, LOF ou Isolation Forest. Os picos acima de 600 mm parecem reais; **só remover do treino, nunca do teste**.
- **Pergunta provável:** por que descartar linhas vazias em vez de imputar? Outlier é sempre erro?

### D. Encoding (Aula 6) · Bueno
- **Situação:** os atributos (X) são todos numéricos; só o alvo (y) é categórico, e o scikit-learn trata o y. Explicar isso com clareza.
- **Sprint 1 planejava** One-Hot para o ano e Label Encoding para o mês. Na versão final o mês virou coluna e o ano ficou de fora. Explicar o porquê: codificar mês como 1 a 12 criaria **falsa ordenação** (Aula 6).
- **Experimento opcional:** acrescentar o ano em One-Hot e comparar com o baseline.
- **Pergunta provável:** por que não codificou nada? O que seria falsa ordenação?

### E. Normalização (Aula 6) · Mateus Henrique
- **Já existe:** 3 scalers (Standard, MinMax, Robust).
- **Falta:** a aula apresenta **cinco** (acrescenta MaxAbs e Quantile Transformer). Incluir os dois na grade e tratar o scaler como hiperparâmetro.
- **Explicar:** o efeito é pequeno porque os 12 atributos já estão em mm; e cada scaler reage de um jeito aos outliers.
- **Pergunta provável:** por que normalizar no KNN? Qual scaler é mais sensível a outliers?

### F. Balanceamento (Aula 8) · Bueno
- Classes: Sertão ~32%, Agreste ~30%, Mata ~21%, Metropolitana ~10%, São Francisco ~8%.
- `KNeighborsClassifier` **não tem `class_weight`**: usar reamostragem (SMOTE, oversampling, undersampling) com `imblearn.pipeline.Pipeline`, **só no treino** de cada fold.
- Sempre comparar com "sem reamostrar" (baseline sério, segundo a Aula 8). Mostrar a troca **precisão × recall** e o F1 do São Francisco.
- **Pergunta provável:** por que reamostrar só o treino? Por que a acurácia pode cair e mesmo assim melhorar?

### G. Seleção de features e PCA (Aula 7) · Mateus Henrique
- Filter: `SelectKBest` (k = 4, 6, 8, 10). Wrapper: `SequentialFeatureSelector` (o `RFE` não serve para KNN, porque exige coeficientes ou importâncias). PCA com padronização, k escolhido por variância (90%, 95%) ou como hiperparâmetro.
- Tudo **dentro do Pipeline**, depois do imputer.
- Discutir: quais meses a seleção escolheu faz sentido? O PCA perde a interpretação.
- **Pergunta provável:** por que PCA não conhece o alvo? Seleção ou extração?

## 3. Regras comuns
1. Todo notebook começa copiando as seções 0 a 4 do `02_Modelagem_KNN` (mesma base de 3.337 linhas e mesma validação).
2. Comparar sempre com a mesma base v1 (veja docs/issues_detalhadas.md) e terminar com uma tabela **antes × depois** (F1 macro, acurácia, F1 do São Francisco).
3. Tudo que aprende dos dados (imputação, scaler, seleção, PCA, reamostragem) fica no Pipeline, só no treino.
4. Subir o notebook em `notebooks/` e comentar na issue; registrar decisões em `docs/decisoes.md`.
