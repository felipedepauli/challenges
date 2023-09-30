import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action="ignore")

# Load data
train_data = pd.read_csv('./train.csv')
test_data  = pd.read_csv('./test.csv')

def preprocessing(train_data):
    # Action 01: Change the string column to a numerical one
    train_data['Sex'] = train_data['Sex'].replace({'male': 0, 'female': 1})
    
    # Action 02: Embarked is spread into three columns (C, S and Q)
    dummy_embarked = pd.get_dummies(train_data['Embarked'], 'Emb')
    train_data = pd.concat([train_data, dummy_embarked], axis=1)
    train_data = train_data.drop(['Embarked'], axis=1)
    
    # Action 03: Splitting the name column into surname and title columns
    # Changed -> surname is not relevant
    # train_data['Surname'] = train_data['Name'].str.split(',').str[0]
    train_data['Title']   = train_data['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    train_data = train_data.drop('Name', axis=1)
    
    # Action 04: Title to dummy, with Mr, Miss, Mrs, Master and Others
    threshold = 30
    title_counts = train_data['Title'].value_counts()
    train_data['Title'] = train_data['Title'].apply(lambda x: x if title_counts[x] >= threshold else 'Others')
    dummy_title = pd.get_dummies(train_data['Title'])
    train_data = pd.concat([train_data, dummy_title], axis=1)
    train_data = train_data.drop('Title', axis=1)

    # Action 05: Fill NULL age values
    # Preencher valores NaN com a mediana da coluna 'Age'
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

    # Aplicar a função ceil e converter para inteiros
    train_data['Age'] = train_data['Age'].apply(np.ceil).astype(int)

        # # Calcular a média da idade para cada grupo (classe, sexo, sobreviveu)
        # age_medians = train_data.groupby(['Pclass', 'Sex', 'Survived'])['Age'].median()

        # # Função para preencher os valores NaN da idade
        # def fill_age(row):
        #     if pd.isnull(row['Age']):
        #         return age_medians[row['Pclass'], row['Sex'], row['Survived']]
        #     else:
        #         return row['Age']

        # # Aplicar a função para preencher os valores NaN
        # train_data['Age'] = train_data.apply(fill_age, axis=1)
        # train_data['Age'] = train_data['Age'].apply(np.ceil).astype(int)


    # Action 06: PassengerId, Cabin and Ticket seem to be irrelevant
    train_data.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)   
    
    # Action 07: Family size of each passenger
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
 

