# Issues detalhadas

Prazos:

| Issue | Tema | Responsável (sugestão) | Prazo |
|---|---|---|---|
| #2 | Preparar ambiente | todos | 18/11 |
| #3 | Convenções | todos | 18/11 |
| #4 | Conferir mapeamento de mesorregiões | Eric | 18/11 |
| #5 | Ausentes (o que significa "-") | Ricardo | 19/11 |
| #6 | Analisar os 360 experimentos + EDA | Eric | 19/11 |
| #7 | Balanceamento e encoding | Bueno | 20/11 |
| #8 | Normalização e seleção de features/PCA | Mateus Henrique | 20/11 |
| #13 (nova) | Valores incorretos, duplicatas e outliers | Eric | 20/11 |
| #12 (nova) | Pipeline final combinado | Randell | 21/11 |
| #9 | Análise de erros e baseline | Randell | 21/11 |
| #10 | Relatório | Randell | 22/11 |
| #11 | Slides e ensaio | Ricardo + todos | 24/11 |

Provas: 23 e 25/11. O trabalho precisa estar fechado até 24/11 (a apresentação é dia 26).

---

## 1. Base v1 e código comum (todos usam)

**Por quê:** para os testes de cada um serem comparáveis e feitos em paralelo, todos comparam contra a **mesma base**, chamada **base v1**:

- linhas com no máximo **6 meses vazios** (3.337 linhas)
- imputação por **mediana**, **StandardScaler**, **k = 5**, peso por **distância**, distância **Manhattan**
- validação: `StratifiedGroupKFold` (5 folds), grupo = posto, seed 42
- resultado esperado: **F1 macro ≈ 0,72**, acurácia ≈ 0,75, F1 do São Francisco ≈ 0,52

**Como usar:** em cada notebook novo, copie do `02_Modelagem_KNN` as células das seções 0 a 4 (imports, dados, mesorregião, X/y/validação, `filtrar` e `montar_pipe`) e acrescente esta célula, que avalia qualquer pipeline e devolve uma linha de tabela:

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

BASE_V1 = dict(max_aus=6, imp='mediana', sc='standard', k=5, w='distance', met='manhattan')

def avaliar(pipe, max_aus=6, rotulo=''):
    """Valida por posto e devolve F1 macro e o F1/recall do São Francisco."""
    Xf, yf, gf = filtrar(X, y, g, max_aus)
    r = cross_validate(pipe, Xf, yf, groups=gf, cv=cv, n_jobs=-1,
                       scoring=['accuracy', 'balanced_accuracy', 'f1_macro'])
    pred = cross_val_predict(pipe, Xf, yf, groups=gf, cv=cv, n_jobs=-1)
    sf = classification_report(yf, pred, output_dict=True, zero_division=0)['São Francisco Pernambucano']
    return {'config': rotulo, 'n_instancias': len(Xf),
            'acc': round(r['test_accuracy'].mean(), 3),
            'bal_acc': round(r['test_balanced_accuracy'].mean(), 3),
            'f1_macro': round(r['test_f1_macro'].mean(), 3),
            'f1_std': round(r['test_f1_macro'].std(), 3),
            'f1_SF': round(sf['f1-score'], 3), 'recall_SF': round(sf['recall'], 3)}

def pipe_base():
    b = BASE_V1
    return montar_pipe(b['imp'], b['sc'], b['k'], b['w'], b['met'])

linhas = [avaliar(pipe_base(), rotulo='base v1')]   # primeira linha da tabela
# ... depois: linhas.append(avaliar(seu_pipeline, rotulo='o que você testou'))
pd.DataFrame(linhas)
```

**Regras:** (1) tudo que aprende dos dados (imputação, scaler, seleção, PCA, reamostragem) fica **dentro do Pipeline**; (2) a base v1 fica numa **célula só** (se o grupo decidir mudar a base, troca-se uma linha); (3) a tabela final de cada notebook é "base v1 × o que você testou", com `f1_macro`, `f1_SF` e `n_instancias`.

---

## 2. Issues

### #2 `[até 18/11 · todos] Preparar ambiente e rodar o projeto`
```
Objetivo: cada pessoa rodar o projeto e confirmar que chega ao mesmo resultado.

Opção A: Colab (sem instalar nada)
- [ ] Abrir o notebook notebooks/02_Modelagem_KNN.ipynb no Colab (File > Open notebook > GitHub, ou upload)
- [ ] Executar tudo (~6 min)
- [ ] Conferir: 5.104 linhas, 12 atributos, 3.337 instâncias no melhor modelo, F1 macro ≈ 0,72, "OK" nos 3 requisitos do final

Opção B: no computador
- [ ] git clone https://github.com/mateuscresp/CienciaDados-Equipe5-Projeto.git
- [ ] pip install -r requirements.txt
- [ ] python src/experimentos.py (≈6 min)

Para contribuir
- [ ] Comentar aqui seu usuário do GitHub (para o Mateus convidar)
- [ ] Ler docs/decisoes.md e docs/divisao_de_temas.md

Pronto quando: você comenta "feito" com o F1 macro que apareceu para você.
Se falhar, comente com a mensagem de erro.
```

### #3 `[até 18/11 · todos] Convenções do grupo`
```
- [ ] Uma branch por tarefa: nome/tarefa (ex.: ricardo/balanceamento)
- [ ] Nada de commit direto na main: abrir Pull Request e pedir revisão de uma pessoa
- [ ] Cada tema tem seu notebook (nomes em docs/divisao_de_temas.md); não editar o notebook de outra pessoa
- [ ] Todo notebook começa com as seções 0 a 4 do 02_Modelagem_KNN e a célula "avaliar" de docs/issues_detalhadas.md
- [ ] Tudo que aprende dos dados fica DENTRO do Pipeline (só no treino)
- [ ] Notebooks: Restart & run all antes de subir; subir com as saídas
- [ ] Seed fixa em 42; não editar data/raw/
- [ ] Decisões novas vão em docs/decisoes.md (data, decisão, motivo)
- [ ] Dúvida ou bloqueio: comentar na própria issue

Pronto quando: todos comentam "li".
```

### #4 `[até 18/11 · Eric] Conferir mapeamento de mesorregiões`
```
Objetivo: garantir que o alvo (mesorregião) está correto, porque tudo depende dele.
Notebook de saída: notebooks/05_Limpeza.ipynb (seção "Mesorregiões")

- [ ] Listar as cidades distintas do dataset (df['Cidade'].nunique()) e conferir se alguma não está em src/meso_map.py (hoje nenhuma linha ficou sem mesorregião)
- [ ] Conferir as variantes de nome (Itaíba/Itaiba, Calçado/Calçados, Lagoa do Itaenga/de Itaenga, São Vicente Férrer/Ferrer): todas na mesma mesorregião?
- [ ] Conferir 15 municípios sorteados contra a fonte oficial (IBGE: mesorregiões de Pernambuco)
- [ ] Verificar quantos postos há por mesorregião e por município; destacar municípios com vários postos
- [ ] Confirmar as contagens: Sertão 1.616, Agreste 1.536, Mata 1.056, Metropolitana 496, São Francisco 400

Pronto quando: seção no notebook 05 e linha em docs/decisoes.md ("mapeamento conferido, N ajustes").
Depende de: nada. Bloqueia: nada (mas se achar erro, avise todos).
```

### #5 `[até 19/11 · Ricardo] Ausentes: o que significa "-" e como tratar`
```
Objetivo: decidir como tratar os meses vazios, com evidência. É o tema de Limpeza (Aula 5).
Notebook de saída: notebooks/05_Ausentes.ipynb
Base: base v1 (docs/issues_detalhadas.md)

Diagnóstico
- [ ] Tabela de % de "-" por mês × mesorregião, só nas linhas parcialmente vazias (1 a 11 meses vazios)
- [ ] Contar linhas totalmente vazias (12 meses) por ano e por posto: há postos que nunca reportam?
- [ ] Confirmar que não existe nenhum valor 0 no dataset (mínimo = 0,1 mm)
- [ ] Texto: mecanismo MCAR, MAR ou MNAR? Justificar com a tabela. (Indício inicial: os "-" se concentram nos meses secos de cada região, então podem significar "chuva zero")
- [ ] Tentar confirmar com a APAC o significado de "-" (site, e-mail ou professora) e registrar

Experimentos (tudo dentro do Pipeline, comparando com a base v1)
- [ ] Imputar mediana (base v1) × KNNImputer × constante 0 (SimpleImputer(strategy='constant', fill_value=0)) nas linhas parciais
- [ ] Linhas 100% vazias: sempre descartadas (explicar por quê)
- [ ] Limite de meses vazios: 4, 6, 8, 10 (e sem limite, só para comparar)
- [ ] Contagem de meses vazios como atributo extra (Aula 5: "a ausência pode ser o sinal")
- [ ] ATENÇÃO: não imputar usando a mesorregião (é o alvo: seria vazamento)

Entrega
- [ ] Tabela base v1 × cada tratamento: n_instancias, f1_macro, f1_std, f1_SF
- [ ] Recomendação clara: "para o pipeline final usar X, com limite Y" (o Randell usa isso na #12)
- [ ] Linha em docs/decisoes.md

Pronto quando: notebook subido, recomendação comentada aqui.
Depende de: base v1. Bloqueia: #12.
```

### #6 `[até 19/11 · Eric] Analisar os 360 experimentos e fechar a EDA`
```
Objetivo: transformar results/experimentos_knn.csv e a EDA em conclusões prontas para relatório e slides.
Notebook de saída: notebooks/06_Analise_Experimentos.ipynb. Gráficos salvos em results/figuras/ (plt.savefig(..., dpi=150)).

Experimentos
- [ ] Ler o CSV (pd.read_csv do results/experimentos_knn.csv) e mostrar os 10 melhores
- [ ] F1 macro médio e máximo por fator: limite de ausentes, imputação, scaler, distância, peso, k (barplot com desvio)
- [ ] Comparar com e sem limite de ausentes, avisando que as bases têm tamanhos diferentes (5.104 × 3.337)
- [ ] Os 5 primeiros diferem menos que o desvio entre folds? Dizer isso com os números

EDA (Aula 4)
- [ ] Distribuição das classes (barras, em %)
- [ ] Boxplots de chuva por mês e por mesorregião (não só julho × dezembro)
- [ ] Correlação entre os meses (heatmap)
- [ ] Mover o main.py do PR #1 para src/eda_graficos.py, lendo o CSV de data/raw/ e salvando figuras em results/figuras/
- [ ] Uma frase por gráfico: o que mostra e o que mudou nas decisões

Entrega: 1 parágrafo para o relatório + 3 a 5 tópicos para o pitch.
Depende de: nada. Bloqueia: #10 e #11.
```

### #7 `[até 20/11 · Bueno] Balanceamento e encoding`
```
Objetivo: (1) melhorar o São Francisco (F1 ~0,52, só ~8% dos dados) e (2) explicar o encoding. Aulas 8 e 6.
Notebook de saída: notebooks/03_Balanceamento.ipynb
Base: base v1 (docs/issues_detalhadas.md)

Balanceamento
- [ ] Instalar: !pip install imbalanced-learn (se o Colab não tiver)
- [ ] ATENÇÃO: usar imblearn.pipeline.Pipeline. Ordem: imputer -> scaler -> reamostragem -> KNN (SMOTE não aceita valores vazios). Reamostrar só o treino de cada fold
- [ ] Testar RandomOverSampler, RandomUnderSampler e SMOTE (k_neighbors = 3 e 5)
- [ ] KNeighborsClassifier NÃO tem class_weight; por isso usamos reamostragem
- [ ] Sempre incluir "sem reamostrar" (a base v1) na tabela (Aula 8: é um baseline sério)
- [ ] Mostrar a troca precisão × recall e o F1 do São Francisco

Encoding (seção dentro do notebook)
- [ ] Mostrar que X é 100% numérico (X.dtypes) e que só o alvo é categórico (y.value_counts())
- [ ] LabelEncoder só para o y; explicar por que não se usa no X
- [ ] Explicar a falsa ordenação: por que codificar o mês como 1 a 12 criaria ordem que não existe (Sprint 1 planejava Label Encoding de mês e One-Hot de ano; hoje o mês virou coluna e o ano ficou de fora)
- [ ] Experimento: acrescentar o Ano em One-Hot (ColumnTransformer + OneHotEncoder(handle_unknown='ignore')) e comparar com a base v1

Entrega
- [ ] Tabela base v1 × cada variação: n_instancias, f1_macro, f1_SF, recall_SF
- [ ] Matriz de confusão do melhor caso
- [ ] 3 a 5 linhas: valeu a pena? o que piorou?
Depende de: base v1. Bloqueia: #12.
```

### #8 `[até 20/11 · Mateus Henrique] Normalização e seleção de features/PCA`
```
Objetivo: (1) comparar os 5 scalers da Aula 6; (2) ver se menos atributos ajudam (Aula 7).
Notebook de saída: notebooks/04_Selecao_Features.ipynb
Base: base v1 (docs/issues_detalhadas.md)

Normalização
- [ ] Testar Standard, MinMax, MaxAbs, Robust e Quantile Transformer (n_quantiles=200), mais "sem normalizar"
- [ ] Tabela com f1_macro e f1_SF de cada um
- [ ] Explicar por que a diferença é pequena (os 12 atributos já estão em mm) e quais scalers são sensíveis a outliers (Aula 6)

Seleção de features (tudo DENTRO do Pipeline, depois do imputer)
- [ ] Remover meses na mão: sem ago a dez; só jan a jul; só os 6 mais chuvosos
- [ ] Filter: SelectKBest(f_classif) com k = 4, 6, 8, 10
- [ ] Wrapper: SequentialFeatureSelector (RFE NÃO serve com KNN, que não tem coeficientes). Pode demorar: usar cv=3 e n_features_to_select = 6
- [ ] Quais meses foram escolhidos? Faz sentido climaticamente?

PCA
- [ ] StandardScaler antes do PCA (obrigatório)
- [ ] PCA com n_components = 0,90 e 0,95 de variância e com 3, 5 e 8 componentes
- [ ] Gráfico da variância explicada acumulada
- [ ] Explicar: PCA não conhece o alvo e perde a interpretação

Entrega
- [ ] Tabela base v1 × cada variação: n_atributos, f1_macro, f1_SF
- [ ] 3 a 5 linhas de conclusão
Depende de: base v1. Bloqueia: #12.
```

### #13 (nova) `[até 20/11 · Eric] Valores incorretos, duplicatas e outliers`
```
Objetivo: completar a limpeza (Aula 5 e os "7 problemas" da Aula 3).
Notebook de saída: notebooks/05_Limpeza.ipynb (notebook 05_Limpeza.ipynb, junto com a #4)
Base: base v1 (docs/issues_detalhadas.md)

Valores incorretos e duplicatas
- [ ] Há valores negativos ou não numéricos? Quais os maiores (máximo hoje: ~1.036 mm em junho)? Algum é fisicamente impossível?
- [ ] Linhas duplicadas: df.duplicated() e df.duplicated(['Código', 'Ano'])
- [ ] Postos diferentes na mesma cidade são redundantes? (correlação entre eles) Só descrever, sem remover

Outliers e ruído
- [ ] IQR (1,5 × IQR) por mês e por mesorregião: quantos outliers? Tabela
- [ ] Opcional: LocalOutlierFactor ou IsolationForest nas linhas completas
- [ ] Decidir: os picos acima de 600 mm parecem reais (slides do Sprint 1 dizem que não prejudicam)
- [ ] Experimento: limitar valores a um teto fixo (ex.: 600 mm) com FunctionTransformer(np.clip, kw_args=dict(a_min=0, a_max=600)) dentro do Pipeline, e comparar com a base v1
- [ ] ATENÇÃO: remover ou limitar outlier só no treino; NUNCA remover do teste (Aula 5)

Entrega: tabela base v1 × tratamento (n_instancias, f1_macro, f1_SF) e uma recomendação.
Depende de: base v1. Bloqueia: #12.
```

### #12 (nova) `[até 21/11 · Randell] Pipeline final combinado`
```
Objetivo: juntar as melhores escolhas de cada tema e rodar de novo, porque cada tema foi testado separado.
Notebook de saída: nova seção "Pipeline final" no notebook 02.

Depende de: #5 (Ricardo), #13 (Eric), #7 (Bueno), #8 (Mateus Henrique).
- [ ] Ler as recomendações das 4 issues e montar o pipeline final (imputação, limite de ausentes, scaler, balanceamento se ajudou, seleção se ajudou)
- [ ] Rodar a grade reduzida (k, peso e distância) nesse pipeline
- [ ] Comparar: base v1 × pipeline final (n_instancias, f1_macro, f1_std, f1_SF)
- [ ] Salvar results/experimentos_finais.csv
- [ ] Atualizar docs/decisoes.md
Se alguma issue atrasar, usar a base v1 para essa parte e avisar no grupo.
```

### #9 `[até 21/11 · Randell] Análise de erros e baseline`
```
- [ ] Acrescentar ao notebook 02 o baseline da classe majoritária (DummyClassifier(strategy='most_frequent')): acurácia ≈ 0,33 e F1 macro ≈ 0,10
- [ ] Interpretar a matriz de confusão do modelo final: São Francisco × Sertão, Agreste × Mata
- [ ] Listar os postos mais mal classificados (por município): são cidades de fronteira entre mesorregiões?
- [ ] Reportar média ± desvio do F1 macro
- [ ] Texto para o relatório: precision, recall e F1 por classe
Depende de: #12.
```

### #10 `[até 22/11 · Randell] Relatório`
```
Seções e quem alimenta cada uma (um parágrafo + 1 tabela ou figura, entregue até 21/11 nos comentários desta issue):
- [ ] Introdução e problema (Eric, #6)
- [ ] Dados e limpeza: ausentes (Ricardo, #5); incorretos, duplicatas, outliers (Eric, #13)
- [ ] Pré-processamento: normalização, encoding e balanceamento (Mateus Henrique #8 e Bueno #7)
- [ ] Método: KNN, Pipeline, validação por posto, métricas (Randell)
- [ ] Resultados: 360 experimentos e pipeline final (Randell #12, Eric #6)
- [ ] Conclusão e limitações (Randell)
- [ ] Revisão final: os números batem com results/?
Randell junta, padroniza e revisa. Se alguém atrasar, usar o que existe e marcar a lacuna.
```

### #11 `[até 24/11 · Ricardo + todos] Slides e ensaio`
```
Pitch de ~15 min: cada um fala 3 min.
- [ ] 1º Eric: problema, dados e EDA
- [ ] 2º Ricardo: limpeza (ausentes, "-" = zero?, outliers)
- [ ] 3º Bueno: encoding e balanceamento
- [ ] 4º Mateus Henrique: normalização e seleção de features
- [ ] 5º Randell: KNN, validação por posto, 360 experimentos, resultado final e limitações
- [ ] Esqueleto dos slides até 20/11 (Ricardo); cada um preenche o seu até 22/11
- [ ] Usar as figuras de results/figuras/ e a tabela "base v1 × final"
- [ ] Ensaio completo com cronômetro em 23/11 ou 24/11
- [ ] Confirmar com a professora o formato do pitch (slides? Colab?)
```
