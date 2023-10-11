# Método

# Metodologia de Machine Learning

## 1. Definição do Problema e Estratégia
1.1 Definir claramente o problema e o objetivo.
1.2 Entender o que os dados representam e suas características.
1.3 Propor uma solução inicial.

## 2. Gather the Data
2.1 Procurar os dados (datasets)
2.2 Definir o tipo de dados
2.3 Prepará-los para importação nos notebooks

## 3. Carregamento, Visualização Inicial e Transformação dos Dados
3.1 Carregar os dados e visualizar as primeiras linhas.
3.2 Realizar as transformações individuais.
3.3 Identificar e tratar valores nulos.

## 4. Análise Exploratória de Dados (EDA)
4.1 Visualizar correlação direta entre cada Feature e Target.
4.2 Analisar estatísticas descritivas e distribuições.
4.3 Identificar e tratar inconsistências e outliers.
4.4 Visualizar relações e correlações entre variáveis.

## 5. Hipóteses e Testes Estatísticos
5.1 Formular e documentar hipóteses.
5.2 Realizar testes estatísticos para validar ou refutar hipóteses.

## 6. Pré-processamento e Engenharia de Features
6.1 Criação de novas features baseadas em insights da EDA.
6.2 Normalização ou padronização de features, se necessário.

## 7. Modelagem
7.1 Dividir os dados em conjuntos de treino e teste.
7.2 Selecionar e treinar modelos iniciais.
7.3 Avaliar e comparar o desempenho dos modelos.
7.4 Ajuste de hiperparâmetros e otimização.

## 8. Avaliação Final e Implementação
8.1 Avaliar o modelo final em um conjunto de teste ou validação.
8.2 Interpretar os resultados e métricas.
8.3 Implementar o modelo para fazer previsões em novos dados.

## 8. Documentação e Iteração
8.1 Documentar todo o processo, decisões, resultados e métricas.
8.2 Revisar a metodologia e identificar áreas de melhoria.
8.3 Iterar: voltar a etapas anteriores se necessário para refinar a análise e o modelo.

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
