# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore")

# %%
# Load data
train_data = pd.read_csv('./train.csv')
test_data  = pd.read_csv('./test.csv')

# %% [markdown]
# Ok. Let's take a look at the challenge page and learn about the data.
# <img src="images/01.png" width="800"/>

# %%
# Look at the first 20 samples of the training dataset
train_data.head(20)
# ... for create a first idea about the problem.

# %%
# Quality of training data
print(f'Total samples: {len(train_data)}')
train_data.isnull().sum()

# %%
# Quality of testing data
print(f'Total samples: {len(test_data)}')
test_data.isnull().sum()

# %% [markdown]
# We have problem with Age, Cabin and Embarked in training data, and with Age, Fare and Cabin in the testing data. We have to handle Age, Cabin, Fare and Embarked nan values in some moment.

# %% [markdown]
# After analyse the date, we're going to perform some transformations. They are:
# 
# - Action 01: Change the string column to a numerical one;
# - Action 02: Embarked will be spread into three columns: (C, S and Q);
# - Action 03: Splitting the name column into surname and title columns;
# - Action 04: Title to dummy, with Mr, Miss, Mrs, Master and Others;
# - Action 05: Fill null age values (Age, Cabin and Embarked);
# - Action 06: Check it all the values are unique;
# - Action 07: Remove unnecessary columns
# 

# %%
# Action 01: Change the string column to a numerical one
train_data['Sex'] = train_data['Sex'].replace({'male': 0, 'female': 1})
test_data['Sex']  = test_data['Sex'] .replace({'male': 0, 'female': 1})


# %%
# Action 02: Embarked is spread into three columns (C, S and Q)
dummy_embarked_train = pd.get_dummies(train_data['Embarked'], prefix='Emb')
train_data = pd.concat([train_data, dummy_embarked_train], axis=1)
train_data.drop(['Embarked'], axis=1, inplace=True)

dummy_embarked_test = pd.get_dummies(test_data['Embarked'], prefix='Emb')
test_data = pd.concat([test_data, dummy_embarked_test], axis=1)
test_data.drop(['Embarked'], axis=1, inplace=True)

# Como as colunas dummy representam a coluna 'Embarked', não devem haver valores ausentes.
# No entanto, se houver algum valor ausente, você pode preencher com 0, assumindo que a ausência de 1 indica que o passageiro não embarcou naquele porto específico.

train_data['Emb_C'].fillna(0, inplace=True)
train_data['Emb_Q'].fillna(0, inplace=True)
train_data['Emb_S'].fillna(0, inplace=True)

test_data['Emb_C'].fillna(0, inplace=True)
test_data['Emb_Q'].fillna(0, inplace=True)
test_data['Emb_S'].fillna(0, inplace=True)



# %%
# Action 03: Splitting the name column into surname and title columns
# Split the Name column into Surname and Title.
train_data['Surname'] = train_data['Name'].str.split(',').str[0]
train_data['Title']   = train_data['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
test_data['Surname']  = test_data ['Name'].str.split(',').str[0]
test_data['Title']    = test_data ['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()

# Drop the unnecessary column.
train_data = train_data.drop('Name', axis=1)
test_data  = test_data .drop('Name', axis=1)

train_data.head()


# %%
train_data['Surname'].value_counts()

# ... it doesn't seem to matter. We will drop the surname column

# %%
train_data.drop('Surname', axis=1, inplace=True)
test_data .drop('Surname', axis=1, inplace=True)

# %%
# Count the unique values of title column
title_counts = train_data['Title'].value_counts()

print(title_counts)
# ... it seems relevant. We will keep it. But...

# %%
# But it could be better if we group all the titles are not Mr, Miss, Mrs, Master, Dr and Rev.
titles_to_keep = [ "Mr", "MIss", "Mrs", "Master"]

train_data["Title"] = train_data["Title"].apply(lambda x: x if x in titles_to_keep else "Others")
test_data["Title"]  = test_data["Title"] .apply(lambda x: x if x in titles_to_keep else "Others")

# %%
train_data.head()

# %%
# Action 04: Title to dummy, with Mr, Miss, Mrs, Master and Others
print("--------------------")
# Checking the results
print(train_data['Title'].value_counts())

dummy_title = pd.get_dummies(train_data['Title'], prefix="Title")
train_data = pd.concat([train_data, dummy_title], axis=1)
train_data = train_data.drop('Title', axis=1)

dummy_title_test = pd.get_dummies(test_data['Title'], prefix="Title")
test_data = pd.concat([test_data, dummy_title_test], axis=1)
test_data = test_data.drop('Title', axis=1)

# %%
train_data.head()

# %%
train_data.info()

# %%
test_data.info()

# %%
# Action 05: Fill null age values (Age, Cabin and Embarked);
def fill_age(row, age_medians):
    if pd.isnull(row['Age']):
        # Check if the key exists in the age_medians Series
        if (row['Pclass'], row['Sex']) in age_medians.index:
            return age_medians[row['Pclass'], row['Sex']]
        else:
            # If the key doesn't exist, you can return a default value or handle it as needed
            # Here we return the median age of all passengers as a default
            return train_data['Age'].median()
    else:
        return row['Age']

# Applying the function to train_data
age_medians = train_data.groupby(['Pclass', 'Sex'])['Age'].median()
train_data['Age'] = train_data.apply(lambda row: fill_age(row, age_medians), axis=1)

# Applying the function to test_data
test_data['Age'] = test_data.apply(lambda row: fill_age(row, age_medians), axis=1)




# %%
# Calcular a média da tarifa para cada classe no conjunto de treinamento
fare_means = train_data.groupby('Pclass')['Fare'].mean()

# Definir uma função para preencher os valores ausentes de Fare com base na média da classe correspondente
def fill_fare(row):
    if pd.isnull(row['Fare']):
        return fare_means[row['Pclass']]
    else:
        return row['Fare']

# Aplicar a função para preencher os valores ausentes de Fare no conjunto de treinamento e de teste
train_data['Fare'] = train_data.apply(lambda row: fill_fare(row), axis=1)
test_data['Fare'] = test_data.apply(lambda row: fill_fare(row), axis=1)


# %%
train_data.head()

# %%
# Action 06: Check it all the values are unique
print("IDs únicos:", train_data['PassengerId'].nunique() == len(train_data))


# %%
# Action 07: Remove unnecessary columns
train_data.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)
test_data.drop(['Cabin', 'Ticket'], axis=1, inplace=True)


# %%
# Countplot about the family size
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize']  = test_data['SibSp']  + test_data['Parch']  + 1

# %% [markdown]
# There is a hypothesis. If a person is Adult, Class C and Man, he did probably not survived. And if is a child and woman, probably survived.

# %%
# Define age groups
def age_group(age):
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teen'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

# Apply the function to create a new age group column
train_data['AgeGroup'] = train_data['Age'].apply(age_group)
test_data['AgeGroup'] = test_data['Age'].apply(age_group)

# Map the gender to strings for better readability
train_data['Sex'] = train_data['Sex'].map({0: 'Male', 1: 'Female'})
test_data['Sex'] = test_data['Sex'].map({0: 'Male', 1: 'Female'})

# Fill NaN values with the mode (most frequent value) of the column
train_data['Pclass'].fillna(train_data['Pclass'].mode()[0], inplace=True)
test_data['Pclass'].fillna(test_data['Pclass'].mode()[0], inplace=True)

# Now convert 'Pclass' to integers and then to strings
train_data['Pclass'] = train_data['Pclass'].astype(int).astype(str)
test_data['Pclass'] = test_data['Pclass'].astype(int).astype(str)

# Create the new combined feature
train_data['Class_Gender'] = train_data['Pclass'] + "_" + train_data['Sex']
test_data['Class_Gender'] = test_data['Pclass'] + "_" + test_data['Sex']

# Drop the original columns
train_data.drop(["Pclass", "Sex"], axis=1, inplace=True)
test_data.drop(["Pclass", "Sex"], axis=1, inplace=True)

# Check the result
print(train_data[['AgeGroup', 'Class_Gender']].head())
print(test_data[['AgeGroup', 'Class_Gender']].head())

# %%
# Use pd.get_dummies to convert the 'Class_Gender_AgeGroup' categorical variable into dummy/indicator variables
age_group_dummies = pd.get_dummies(train_data['AgeGroup'], prefix="AG")
class_gender_dummies = pd.get_dummies(train_data['Class_Gender'], prefix='CGA')
age_group_dummies_test = pd.get_dummies(test_data['AgeGroup'], prefix="AG")
class_gender_dummies_test = pd.get_dummies(test_data['Class_Gender'], prefix='CGA')

# Now, you can concatenate these new columns (dummy variables) back to your original DataFrame
train_data = pd.concat([train_data, age_group_dummies], axis=1)
train_data = pd.concat([train_data, class_gender_dummies], axis=1)
test_data = pd.concat([test_data, age_group_dummies_test], axis=1)
test_data = pd.concat([test_data, class_gender_dummies_test], axis=1)


train_data.drop(['AgeGroup', 'Class_Gender'], axis=1, inplace=True)
test_data.drop(['AgeGroup', 'Class_Gender'], axis=1, inplace=True)


# %%
train_data.head()

# %%
train_data.info()

# %%
test_data.info()

# %%
# Verificar valores nulos
print(train_data.isnull().sum(), "\n----\n")
print(test_data.isna().sum())

# %% [markdown]
# # Training

# %%
X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']

# %%
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# %%
train_data.head()

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Modelos
models = {
    'Logistic Regression'       : LogisticRegression(),
    'Decision Tree'             : DecisionTreeClassifier(),
    'Random Forest'             : RandomForestClassifier(),
    'Gradient Boosting'         : GradientBoostingClassifier(),
    'Support Vector Classifier' : SVC(),
}

# Treinar e avaliar modelos
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    print(f'{name}: {accuracy}')


# %%
# RANDOM FOREST
from sklearn.model_selection import GridSearchCV
# Define the hyperparameters and their values to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 11),
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
model_accuracies={}
# Create a RandomForestClassifier
model = RandomForestClassifier(random_state=420)

# Initialize GridSearchCV with the model, hyperparameters, and cross-validation strategy
rf_model = GridSearchCV(model, param_grid, n_jobs=-1)

# Fit the GridSearchCV to the data
print('Getting HP - Random Forest...')
rf_model.fit(X_train, y_train)
score=rf_model.score(X_train, y_train)*100
print("Model Score:", score)
# Print the best hyperparameters and corresponding accuracy score
print("Best Hyperparameters:", rf_model.best_params_)
print("Best Accuracy:", rf_model.best_score_)
model_accuracies['Random Forest']= score

# %%
# SVM
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Definir os hiperparâmetros e seus valores para a busca
param_grid_svm = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Type of kernel
    'degree': [2, 3, 4],  # Degree of the polynomial kernel function (‘poly’)
    'gamma': ['scale', 'auto'],  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
}

# Criar o classificador SVM
svm = SVC(random_state=42)

# Inicializar o GridSearchCV com o modelo, hiperparâmetros e estratégia de validação cruzada
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar o GridSearchCV aos dados de treino
print('Getting HP - SVC...')
grid_search_svm.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros e a pontuação de acurácia correspondente
print("Best Hyperparameters:", grid_search_svm.best_params_)
print("Best Accuracy:", grid_search_svm.best_score_)

# Salvar a acurácia do melhor modelo encontrado
model_accuracies['SVM'] = grid_search_svm.best_score_ * 100


# %%
# LOGISTIC REGRESSION
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Definir os hiperparâmetros e seus valores para a busca
param_grid_lr = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Criar o classificador de Regressão Logística
logreg = LogisticRegression(max_iter=1000)  # Aumentar o número de iterações para garantir a convergência

# Inicializar o GridSearchCV com o modelo, hiperparâmetros e estratégia de validação cruzada
grid_search_lr = GridSearchCV(logreg, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar o GridSearchCV aos dados de treino
print('Getting HP - Logistic Regression...')
grid_search_lr.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros e a pontuação de acurácia correspondente
print("Best Hyperparameters:", grid_search_lr.best_params_)
print("Best Accuracy:", grid_search_lr.best_score_)

# Salvar a acurácia do melhor modelo encontrado
model_accuracies['Logistic Regression'] = grid_search_lr.best_score_ * 100


# %%
# GRADIENT BOOSTING
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

# Definir os hiperparâmetros e seus valores para a busca
param_grid_gb = {
    'n_estimators': [100, 200, 300],  # Number of boosting stages to perform
    'learning_rate': [0.01, 0.1, 0.2],  # Learning rate shrinks the contribution of each tree
    'max_depth': [3, 4, 5],  # Maximum depth of the individual regression estimators
    'min_samples_split': [2, 4],  # The minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2],  # The minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],  # The number of features to consider when looking for the best split
}

# Criar o classificador Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)

# Inicializar o GridSearchCV com o modelo, hiperparâmetros e estratégia de validação cruzada
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar o GridSearchCV aos dados de treino
print('Getting HP - Gradient Boosting...')
grid_search_gb.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros e a pontuação de acurácia correspondente
print("Best Hyperparameters:", grid_search_gb.best_params_)
print("Best Accuracy:", grid_search_gb.best_score_)

# Salvar a acurácia do melhor modelo encontrado
model_accuracies['GradientBoosting'] = grid_search_gb.best_score_ * 100


# %%
# NAIVE BAYES
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

# Definir os hiperparâmetros e seus valores para a busca
# GaussianNB não tem muitos parâmetros para ajustar, mas você pode tentar ajustar 'var_smoothing'
param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

# Criar o classificador Gaussian Naive Bayes
nb = GaussianNB()

# Inicializar o GridSearchCV com o modelo, hiperparâmetros e estratégia de validação cruzada
grid_search_nb = GridSearchCV(nb, param_grid_nb, cv=5, scoring='accuracy', n_jobs=-1)

# Ajustar o GridSearchCV aos dados de treino
print('Getting HP - Naive Bayes...')
grid_search_nb.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros e a pontuação de acurácia correspondente
print("Best Hyperparameters:", grid_search_nb.best_params_)
print("Best Accuracy:", grid_search_nb.best_score_)

# Salvar a acurácia do melhor modelo encontrado
model_accuracies['Naive Bayes'] = grid_search_nb.best_score_ * 100


# %%
passengerId = test_data["PassengerId"]
test_data.drop("PassengerId", axis=1, inplace=True)

# Supondo que você tenha os melhores modelos já treinados e armazenados nas variáveis abaixo
best_models = {
    'RandomForest': rf_model.best_estimator_,
    'SVM': grid_search_svm.best_estimator_,
    'LogisticRegression': grid_search_lr.best_estimator_,
    'GradientBoosting': grid_search_gb.best_estimator_,
    'NaiveBayes': grid_search_nb.best_estimator_
}

for name, model in best_models.items():
    # Fazer previsões no conjunto de teste
    test_predictions = model.predict(test_data)

    # Preparar arquivo de submissão
    submission = pd.DataFrame({'PassengerId': passengerId, 'Survived': test_predictions})
    submission.to_csv(f'submission_{name}.csv', index=False)