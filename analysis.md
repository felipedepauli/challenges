# Método

## 1. Problem Definition and Strategy
1.1 Clearly define the problem and objective.
1.2 Understand what the data represents and its characteristics.
1.3 Propose an initial solution.

## 2. Gather the Data
2.1 Seek out the data (datasets)
2.2 Define the data type
2.3 Prepare them for import into notebooks

## 3. Data Loading, Initial Visualization, and Transformation
3.1 Load the data and visualize the first few rows.
3.2 Perform individual transformations.
3.3 Identify and handle null values.

## 4. Exploratory Data Analysis (EDA)
4.1 Visualize direct correlation between each Feature and Target.
4.2 Analyze descriptive statistics and distributions.
4.3 Identify and handle inconsistencies and outliers.
4.4 Visualize relationships and correlations between variables.

## 5. Hypotheses and Statistical Testing
5.1 Formulate and document hypotheses.
5.2 Perform statistical tests to validate or refute hypotheses.

## 6. Pre-processing and Feature Engineering
6.1 Create new features based on EDA insights.
6.2 Normalize or standardize features, if necessary.

## 7. Modeling
7.1 Split the data into training and testing sets.
7.2 Select and train initial models.
7.3 Evaluate and compare the performance of the models.
7.4 Hyperparameter tuning and optimization.

## 8. Final Evaluation and Implementation
8.1 Evaluate the final model on a testing or validation set.
8.2 Interpret results and metrics.
8.3 Implement the model to make predictions on new data.

## 9. Documentation and Iteration
9.1 Document the entire process, decisions, results, and metrics.
9.2 Review the methodology and identify areas for improvement.
9.3 Iterate: return to previous steps if necessary to refine the analysis and model.


---

**Observações:**
- Mantenha consistência entre transformações nos conjuntos de treino e teste.
- Modularize o código, especialmente funções de pré-processamento, para reutilização.
- Documente insights, decisões e justificativas ao longo do notebook para referência futura.



Observações:
- Todas as alterações realizadas no dataset de treinamento devem ser realizadas também no dataset de teste
- É ideal que cada ação de transformação seja criada em uma função a parte chamada preprocesing, para que possa ser usada em um programa posteriormente, e não só para análise
- 



# Comandos
## df.head()

- **Quando usar:** No início da análise para visualizar as primeiras linhas do DataFrame.
- **Por que usar:** Para ter uma visão inicial da estrutura dos dados.
- **Como usar:** 
  ```python
  df.head(n)
  ```
  onde `n` é o número de linhas a serem exibidas.

## df.describe()

- **Quando usar:** Para obter um resumo estatístico das variáveis numéricas.
- **Por que usar:** Para analisar a tendência central, dispersão e forma da distribuição.
- **Como usar:** 
  ```python
  df.describe()
  ```

## df.info()

- **Quando usar:** Para obter informações sobre o tipo de dados e valores nulos.
- **Por que usar:** Para verificar se os tipos de dados estão corretos e identificar colunas com valores nulos.
- **Como usar:** 
  ```python
  df.info()
  ```

## sns.pairplot(df)

- **Quando usar:** Para visualizar as relações entre variáveis numéricas.
- **Por que usar:** Para identificar correlações e padrões nos dados.
- **Como usar:** 
  ```python
  sns.pairplot(df)
  ```
  onde `df` é o DataFrame.

## df['column'].value_counts().plot.bar()

- **Quando usar:** Para visualizar a distribuição de variáveis categóricas.
- **Por que usar:** Para identificar a frequência de cada categoria.
- **Como usar:** 
  ```python
  df['column'].value_counts().plot.bar()
  ```
  Substitua `'column'` pelo nome da coluna categórica.

## sns.heatmap(df.corr(), annot=True)

- **Quando usar:** Para visualizar a matriz de correlação entre variáveis numéricas.
- **Por que usar:** Para identificar correlações fortes ou fracas entre as variáveis.
- **Como usar:** 
  ```python
  sns.heatmap(df.corr(), annot=True)
  ```

## df.isnull().sum()

- **Quando usar:** Para verificar a quantidade de valores nulos em cada coluna.
- **Por que usar:** Para identificar colunas que necessitam de tratamento de dados ausentes.
- **Como usar:** 
  ```python
  df.isnull().sum()
  ```

## sns.boxplot(x='column', y='target', data=df)

- **Quando usar:** Para visualizar a distribuição de uma variável em relação a diferentes categorias.
- **Por que usar:** Para identificar outliers e comparar distribuições.
- **Como usar:** 
  ```python
  sns.boxplot(x='column', y='target', data=df)
  ```
  Substitua `'column'` pela variável categórica e `'target'` pela variável numérica.

## sns.histplot(df['column'], bins=30)

- **Quando usar:** Para visualizar a distribuição de uma variável numérica.
- **Por que usar:** Para analisar a forma da distribuição e identificar anomalias.
- **Como usar:** 
  ```python
  sns.histplot(df['column'], bins=30)
  ```
  Substitua `'column'` pelo nome da coluna numérica.

## sns.scatterplot(x='column1', y='column2', data=df)

- **Quando usar:** Para visualizar a relação entre duas variáveis numéricas.
- **Por que usar:** Para identificar tendências, padrões e outliers.
- **Como usar:** 
  ```python
  sns.scatterplot(x='column1', y='column2', data=df)
  ```
  Substitua `'column1'` e `'column2'` pelos nomes das colunas numéricas.


Corr-> 
http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
