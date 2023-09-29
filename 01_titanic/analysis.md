# Método

### 1. Visualização e Exploração dos Dados
- Execute comandos para visualizar os dados.
- Documente observações e características dos dados.
- Identifique valores nulos, outliers e inconsistências.
  
### 2. Pré-processamento e Limpeza dos Dados
- Defina ações de tratamento dos dados baseadas nas observações.
- Trate valores nulos e outliers.
- Realize a codificação de variáveis categóricas.
  
### 3. Criação de Hipóteses e Análises
- Formule hipóteses sobre os dados (ex: pessoas nascidas em 1980 têm maior probabilidade de…).
- Teste as hipóteses através de análises estatísticas e exploração de dados.
- Verifique correlações entre diferentes variáveis.
  
### 4. Visualização Avançada
- Crie gráficos e visualizações para entender padrões e relações.
- Use as visualizações para identificar possíveis modelos a serem utilizados.
  
### 5. Seleção e Treinamento de Modelos
- Escolha o modelo que parece ser o mais adequado.
- Treine diversos modelos e valide o desempenho de cada um.
- Ajuste hiperparâmetros e otimize o modelo escolhido.
  
### 6. Avaliação e Implementação
- Avalie o modelo final com base em métricas de desempenho.
- Implemente o modelo e faça previsões.
- Prepare e submeta os resultados, se necessário.
  
### 7. Documentação e Revisão
- Documente todo o processo, decisões tomadas e resultados obtidos.
- Revise o workflow e faça ajustes conforme necessário para futuros projetos.






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




### Numéricos
- PassengerId é um número, mas à princípio não tem importância. É único, por se tratar de identificação.
  - Verificar se todos os IDs são únicos
  - Procurar alguma correlação.
- Survived indica se a pessoa morreu ou não. Já está como número, sendo 0 para Morreu, e 1 para Sobreviveu. Este é o nosso target.
  - Remover o target para treinamento.
- Pclass é a classe da pessoa. 1a, 2a ou 3a classe. Já está como número e na ordem certa. Com certeza tem alguma coisa a ver. Provavelmente quem tem classes superiores teve acesso aos botes de forma mais facilitada.
  - Verificar quantos mortos em cada classe, em números abs e proporcinais.
- Age é a idade. É um float, pois pega os meses também. Não acredito que seja relevante. Pode ser arredondado para cima e virar um int.
  - Verificar se a idade influenciou. Provavelmente pessoas idosas morreram mais.
- SibSp é o número de irmãos e esposos/as que o passageiro tinha no navio.
- Parch é o número de pais e filhos que o passageiro tinha.
- Fare é o preço da passagem. Provavelmente tem correlação assim como a classe.
- 

### Não numéricos
- Name. Não acredito que tenha correlação. Mas talvez algumas famílias tenham tido melhor sorte que outras.
  - Separar os sobrenomes e fazer um dummy com eles
- Sex provavelmente tenha correlação.
  - Analisar mortes por gênero
  - Mudar para categórico
- Ticket
- Cabin
- Embarked









----------------------
1. Análise de dados. Quais são as colunas e o que elas apresentam?
   
> head()
> info()
> ------
> 

Colunas: É possível verificar que existem 7 colunas numéricas e 5 colunas não numéricas.
.. 
Amostras: São 891 amostras disponíveis.

### Numéricas:

**PassengerId**
- [H] Não se repete
- [H] Não é relevante

**Survived**
- [F] Nosso target

**Pclass**
- Correlações

**Age** toInt
- Correlações
- [F] Média de idade é de 30 anos
- [F] 75% das pessoas têm menos de 38 anos

**SibSp**
- Correlações
- [F] 50% das pessoas não têm SibSp
  
**Parch**
- Correlações
- [F] 75% não tem Pais ou filhos
- [H] Essa feature não é relevante
- [F] A variância é muito baixa, e média 0.4

**Fare** toInt
- Correlações
- [F] Pelo menos 75% pagou menos de 31 na passagem
- [F] A média da passagem é 32
  
### Categóricas:

**Name** Processo
- Talvez os títulos e/ou sobrenomes sejam importantes

**Sex** Dummy
- Correlações

**Ticket** ?
- Correlações

**Cabin** ?
- Correlações
  
**Embarked** Dummy
- Correlações

## Todo
### Análise inicial
- Verificar se alguma coluna pode obviamente ser removida
- Converter float para int
- Formular hipóteses individuais
- Formular hipóteses de correlação