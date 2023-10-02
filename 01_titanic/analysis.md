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
1. Análise de dados.
   
1.1 Quais são as colunas e o que elas apresentam?
   
> head()
> info()

Colunas: É possível verificar que existem 7 colunas numéricas e 5 colunas não numéricas.
.. 
Amostras: São 891 amostras disponíveis.

### Numéricas:

**PassengerId**
Identificação dos passageiros. Cada passageiro tem uma identificação única.
- [F] Não se repete
- [H] Não é relevante

**Survived**
Quem sobreviveu? 0 foi de vala, 1 se salvou.
- [F] Nosso target

**Pclass**
Classe dos magrões. Pode ser 1a, 2a ou 3a classe.
- Correlações

**Age** toInt
- Não se correlaciona com nenhuma var numérica
- Problema com NULL ->> pegar a média da classe, sexo e se morreu
- [F] Média de idade é de 30 anos
- [F] 75% das pessoas têm menos de 38 anos

**SibSp**
- Não se correlaciona com nenhuma var numérica
- [F] 50% das pessoas não têm SibSp
  
**Parch**
- Não se correlaciona com nenhuma var numérica
- [F] 75% não tem Pais ou filhos
- [H] Essa feature não é relevante
- [F] A variância é muito baixa, e média 0.4

**Fare** toInt
- Não se correlaciona com nenhuma var numérica
- [F] Pelo menos 75% pagou menos de 31 na passagem
- [F] A média da passagem é 32
  
### Categóricas:

**Name** Processo
- Talvez os títulos e/ou sobrenomes sejam importantes

**Sex** toInt
- [OK] Transformação direta!
- [F] Tem mais homens do que mulheres
- [F] Relação alta entre quem sobrevive e sexo

**Ticket** ?
- Correlações

**Cabin** ?
- Correlações
  
**Embarked** Dummy
- Correlações


## Statements

Para identificar correlações fortes ou fracas entre as variáveis.
Para identificar a frequência de cada categoria.
Para identificar tendências, padrões e outliers.


- Predicting survival status in the Titanic ship disaster. Its a Binary Classification problem
- Training dataset have minimal feature hence we need to create feature cross by understanding business. Refer https://developers.google.com/machine-learning/crash-course/feature-crosses/video-lecture for more details
- Due to minimal samples in training dataset generate synthetic data for more accuracy but we are not generating here to show originiality
- Check each feature correlation with survival




## Correlações numéricas
> sns.pairplot(df)
> sns.heatmap(df.corr(), annot=True)

- [H] Não há correlação alguma entre quaiquer variáveis numéricas
- [F] A maior correlação é entre Parch e SibSp
- [F] Correlaçao alta entre Survived e Fare
- [F] A menor correlação é entre Pclass e Fare
- [F] Correlação baixa entre Pclass e Age
- [F] Correlação baixa entre Survived e Pclass

## Análise Categórica

- [F] Muito mais homens morrem do que mulheres
- [F] A cada 1 homem vivo, há 4 mortos
- [F] A cada 1 mulher morta, há 3 vivas
- [F] Pessoas da terceira classe morrem mais
- [F] O embarque C tem relação com valor de ticket
- [F] O embarque Q e S não têm correlação com valor de ticket
- [F] O embarque Q tem relação com Pclass


## Todo
### Análise inicial
- Verificar se alguma coluna pode obviamente ser removida
- Converter float para int
- Formular hipóteses individuais
- Formular hipóteses de correlação