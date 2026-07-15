# Projeto de Machine Learning — Risco de Crédito

## 1. Contexto do Projeto

Este projeto foi desenvolvido como atividade acadêmica do módulo de **Machine Learning e Visão Computacional**.

A base escolhida foi a **Opção A — Risco de Crédito**, cujo objetivo é construir um pipeline preditivo capaz de classificar clientes de um banco entre:

- **Classe 0:** cliente pagador;
- **Classe 1:** cliente inadimplente.

O problema de negócio consiste em prever se um cliente poderá se tornar inadimplente, auxiliando a instituição financeira na tomada de decisão sobre concessão de crédito.

---

## 2. Objetivo

O objetivo principal do projeto é desenvolver um pipeline completo de Machine Learning, passando pelas seguintes etapas:

1. Análise Exploratória de Dados;
2. Tratamento e limpeza da base;
3. Criação de nova variável por Feature Engineering;
4. Separação entre treino e teste;
5. Balanceamento de classes;
6. Escalonamento seguro para o modelo KNN;
7. Treinamento e comparação entre KNN e Árvore de Decisão;
8. Diagnóstico de overfitting;
9. Avaliação por métricas e matriz de confusão;
10. Veredito final de negócio.

---

## 3. Base de Dados

A base utilizada foi `credit_risk_dataset.csv`.

Ela contém informações relacionadas ao perfil do cliente, características do empréstimo e histórico de crédito.

A variável alvo do projeto é:

| Coluna | Descrição |
|---|---|
| `loan_status` | Indica o status do empréstimo. `0` representa cliente pagador e `1` representa cliente inadimplente. |

---

## 4. Dicionário de Dados

| Coluna | Descrição |
|---|---|
| `person_age` | Idade da pessoa. |
| `person_income` | Renda anual da pessoa. |
| `person_home_ownership` | Situação de moradia da pessoa. |
| `person_emp_length` | Tempo de emprego em anos. |
| `loan_intent` | Finalidade do empréstimo. |
| `loan_grade` | Classificação de risco do empréstimo. |
| `loan_amnt` | Valor solicitado no empréstimo. |
| `loan_int_rate` | Taxa de juros do empréstimo. |
| `loan_status` | Variável alvo. Indica se o cliente pagou ou ficou inadimplente. |
| `loan_percent_income` | Percentual da renda comprometido com o empréstimo, presente na base original. |
| `cb_person_default_on_file` | Indica se há histórico anterior de inadimplência. |
| `cb_person_cred_hist_length` | Tempo de histórico de crédito da pessoa. |
| `comprometimento_renda` | Nova coluna criada no projeto. Representa o percentual da renda comprometido pelo valor solicitado no empréstimo. |

---

## 5. Estrutura do Projeto

```text
projeto-risco-credito-machine-learning/
│
├── data/
│   └── credit_risk_dataset.csv
│
├── imagens/
│   ├── graficos_eda/
│   ├── matriz_confusao_knn.png
│   └── matriz_confusao_arvore.png
│
├── notebooks/
│   └── pipeline_risco_credito.ipynb
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

## 6. Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook
- UV
- Git e GitHub

---

## 7. Análise Exploratória de Dados

Na etapa de EDA foram realizadas análises estatísticas e visuais para compreender a estrutura da base.

Foram avaliados:

- tamanho da base;
- tipos de dados;
- estatísticas descritivas;
- valores nulos;
- linhas duplicadas;
- distribuição da variável alvo;
- histogramas;
- boxplots;
- mapa de calor de correlação de Pearson.

A análise mostrou que a variável alvo `loan_status` apresenta desbalanceamento, com maior quantidade de clientes pagadores do que inadimplentes.

Também foram identificados valores ausentes nas colunas `person_emp_length` e `loan_int_rate`, além de possíveis outliers em variáveis como idade, renda, tempo de emprego, valor do empréstimo e histórico de crédito.

---

## 8. Tratamento e Limpeza dos Dados

Na etapa de tratamento foram realizadas as seguintes ações:

### Remoção de duplicadas

As linhas duplicadas foram removidas para evitar redundância na base e reduzir o risco de enviesar o treinamento dos modelos.

### Tratamento de valores nulos

Os valores ausentes foram tratados com a **mediana**.

A mediana foi escolhida porque a análise exploratória indicou assimetria e presença de valores extremos em algumas variáveis numéricas. Nesses casos, a média pode ser influenciada por outliers, enquanto a mediana tende a representar melhor o centro da distribuição.

### Tratamento de outliers

Os outliers foram tratados por **clipping** com base no método IQR.

Essa decisão foi tomada para reduzir o impacto de valores extremos sem remover registros da base, já que em dados financeiros valores altos de renda, empréstimo ou histórico de crédito podem representar clientes reais.

---

## 9. Feature Engineering

Foi criada a coluna `comprometimento_renda`, conforme a fórmula:

```python
comprometimento_renda = (loan_amnt / person_income) * 100
```

Essa variável representa o percentual da renda anual comprometido pelo valor do empréstimo.

No contexto de risco de crédito, essa informação é importante porque clientes que solicitam empréstimos proporcionalmente altos em relação à renda podem apresentar maior risco de inadimplência.

---

## 10. Preparação para Modelagem

A preparação dos dados foi feita com cuidado para evitar vazamento de dados.

As principais etapas foram:

1. Separação entre variáveis preditoras `X` e alvo `y`;
2. Divisão em treino e teste com `test_size=0.20`;
3. Uso de `stratify=y` para preservar a proporção das classes;
4. One-Hot Encoding nas variáveis categóricas;
5. Aplicação de SMOTE somente nos dados de treino;
6. Escalonamento com StandardScaler somente para o KNN;
7. Treinamento da Árvore de Decisão sem escalonamento.

O conjunto de teste não foi balanceado, pois ele deve representar a realidade original dos dados.

---

## 11. Modelagem

Foram treinados dois modelos:

- KNN;
- Árvore de Decisão.

### KNN

O KNN foi testado com os seguintes valores de `n_neighbors`:

```text
3, 5, 7 e 9
```

### Árvore de Decisão

A Árvore de Decisão foi testada com os seguintes valores de `max_depth`:

```text
3, 5, 7 e None
```

A comparação entre treino e teste foi utilizada para avaliar possíveis sinais de overfitting.

---

## 12. Resultados Finais

Após os testes, os melhores modelos encontrados foram:

| Modelo | Melhor parâmetro | Acurácia | Precisão Classe 1 | Recall Classe 1 | F1 Classe 1 |
|---|---:|---:|---:|---:|---:|
| KNN | K=9 | 84,11% | 61,04% | 75,67% | 67,57% |
| Árvore de Decisão | max_depth=7 | 92,60% | 97,28% | 68,05% | 80,08% |

---

## 13. Análise da Matriz de Confusão

| Modelo | Verdadeiro Negativo | Falso Positivo | Falso Negativo | Verdadeiro Positivo |
|---|---:|---:|---:|---:|
| KNN | 4381 | 685 | 345 | 1073 |
| Árvore de Decisão | 5039 | 27 | 453 | 965 |

No contexto de risco de crédito:

- **Falso Positivo:** cliente bom pagador classificado como inadimplente;
- **Falso Negativo:** cliente inadimplente classificado como bom pagador.

O Falso Positivo gera perda de oportunidade comercial, pois o banco pode negar crédito a um bom cliente.

O Falso Negativo tende a ser mais grave, pois pode levar o banco a aprovar crédito para um cliente com maior risco de não pagamento.

---

## 14. Veredito de Negócio

A Árvore de Decisão apresentou melhor desempenho geral em acurácia, precisão e F1-score da classe inadimplente. O modelo atingiu acurácia de 92,60%, precisão de 97,28% para a classe 1 e F1-score de 80,08%.

No entanto, o KNN apresentou melhor recall para a classe 1, com 75,67%, contra 68,05% da Árvore de Decisão. Isso significa que o KNN identificou uma proporção maior de clientes inadimplentes.

A matriz de confusão também mostra que o KNN gerou 345 falsos negativos, enquanto a Árvore de Decisão gerou 453 falsos negativos.

Como o Falso Negativo representa um cliente inadimplente sendo classificado como seguro, esse erro é especialmente crítico para o banco. Ele pode gerar aprovação indevida de crédito, perda financeira direta e aumento da inadimplência da carteira.

Dessa forma, apesar de a Árvore de Decisão apresentar melhor desempenho geral, o modelo recomendado para uma política conservadora de risco é o **KNN com K=9**, pois ele reduziu a quantidade de falsos negativos e apresentou maior capacidade de identificar clientes inadimplentes.

Como recomendação prática, o banco poderia utilizar o KNN como modelo inicial de triagem de risco. Clientes classificados como alto risco poderiam passar por uma análise complementar antes da recusa definitiva do crédito, reduzindo o impacto dos falsos positivos.

---

## 15. Como Executar o Projeto

### Opção 1 — Usando UV

Clone o repositório:

```bash
git clone https://github.com/SEU-USUARIO/projeto-risco-credito-machine-learning.git
```

Acesse a pasta do projeto:

```bash
cd projeto-risco-credito-machine-learning
```

Instale as dependências com UV:

```bash
uv sync
```

Abra o notebook:

```bash
jupyter notebook notebooks/pipeline_risco_credito.ipynb
```

---

### Opção 2 — Usando pip

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual no Windows:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra o notebook:

```bash
jupyter notebook notebooks/pipeline_risco_credito.ipynb
```

---

## 16. Conclusão

O projeto demonstrou a importância de construir um pipeline de Machine Learning com rigor metodológico.

A análise mostrou que a escolha do melhor modelo não deve ser feita apenas pela acurácia. Em problemas de risco de crédito, a interpretação dos erros é fundamental, principalmente porque falsos negativos podem gerar prejuízo financeiro direto para a instituição.

O modelo KNN com K=9 foi recomendado para uma abordagem mais conservadora, pois apresentou melhor capacidade de identificar clientes inadimplentes.
