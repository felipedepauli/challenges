# Método

### 1. Estratégia
- 1.1 Ler e entender o que os dados representam e suas características
- 1.2 Entender o problema e o objetivo
- 1.3 Desenhar uma proposta de solução

### 2. Visualização e Exploração dos Dados
- 2.1 Executar comandos para visualizar os dados.
- 2.2 Documentar observações e características dos dados (análise qualitativa).
- 2.3 Identificar valores nulos, outliers e inconsistências (tipos e conteúdo).
- 2.4 Confirmar unicidade em dados que devem ser únicos
  
### 3. Pré-processamento e Limpeza dos Dados
- 3.1 Definir ações de tratamento de dados.
- 3.2 Remover colunas desnecessárias.
- 3.3 Mudar tipos de variáveis, buscando int, float, categóricos, nesta ordem.
- 3.4 Tratar valores nulos e outliers (poda).
- 3.5 Criar novas features óbvias.
  
### 4. Criação de Hipóteses e Análises
- 4.1 Formular hipóteses sobre os dados (ex: pessoas nascidas em 1980 têm maior probabilidade de…).
- 4.2 Testar as hipóteses através de análises estatísticas e exploração de dados.
- 4.3 Verificar correlações entre diferentes variáveis.
- 4.4 Criar novas features por correlação.
  
### 5. Visualização Avançada
- 5.1 Criar gráficos e visualizações para entender padrões e relações finais.
- 5.2 Use as visualizações para presumir possíveis modelos a serem utilizados. Elencá-los.
- 5.3 Se necessário, realizar ajustes na estratégia e voltar para ações anteriores.

### 6. Preparação de dados
- 6.1 Criar as estruturas para treinamento
  
### 7. Seleção e Treinamento de Modelos
- 7.1 Executar modelos com parâmetros padrões e compará-los.
- 7.2 Escolhe os três modelos que parecem ser os mais adequados.
- 7.3 Treinar modelos com parâmetros ajustados e validar o desempenho de cada um.
- 7.4 Escolher um e realizar ajuste de hiperparâmetros e otimizar o modelo escolhido.
  
### 8. Avaliação e Implementação
- 8.1 Avalie o modelo final com base em métricas de desempenho.
- 8.2 Implemente o modelo e faça previsões.
- 8.3 Prepare e submeta os resultados, se necessário.
  
### 9. Documentação e Revisão
- 9.1 Documente todo o processo, decisões tomadas e resultados obtidos (dados históricos).
- 9.2 Revise o workflow e faça ajustes conforme necessário para futuros projetos.


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