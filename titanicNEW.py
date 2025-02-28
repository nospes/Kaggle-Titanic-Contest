import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Carregar os dados
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Criar uma cópia para evitar alterações nos dados originais
df = train_df.copy()

# Preenchendo valores nulos da coluna "Age" baseado na mediana da classe e do sexo
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
test_df["Age"] = test_df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))

# Criando uma nova coluna binária indicando se o passageiro tinha cabine ou não
df["HasCabin"] = df["Cabin"].notnull().astype(int)
test_df["HasCabin"] = test_df["Cabin"].notnull().astype(int)
df.drop(columns=["Cabin"], inplace=True)
test_df.drop(columns=["Cabin"], inplace=True)

# Preenchendo valores nulos da coluna "Embarked" com a moda
df = df.assign(Embarked=df["Embarked"].fillna(df["Embarked"].mode()[0]))
test_df = test_df.assign(Embarked=test_df["Embarked"].fillna(test_df["Embarked"].mode()[0]))

# Criando uma nova coluna "Title" extraída do nome
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
test_df["Title"] = test_df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# Agrupando títulos raros
title_replacements = {
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare", "Don": "Rare",
    "Dr": "Rare", "Major": "Rare", "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"
}
df["Title"] = df["Title"].replace(title_replacements)
test_df["Title"] = test_df["Title"].replace(title_replacements)

# Criando uma nova feature "FareGroup" agrupando os valores das passagens
df["FareGroup"] = pd.qcut(df["Fare"], 4, labels=[0, 1, 2, 3])
test_df["FareGroup"] = pd.qcut(test_df["Fare"].fillna(test_df["Fare"].median()), 4, labels=[0, 1, 2, 3])

# Criando nova feature "FamilySize"
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# Criando uma nova feature "IsAlone"
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
test_df["IsAlone"] = (test_df["FamilySize"] == 1).astype(int)

# Mapeando Sex para valores numéricos
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})

# Usar LabelEncoder para converter variáveis categóricas
categorical_features = ["Embarked", "Title", "FareGroup"]
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Removendo colunas desnecessárias
df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
test_df.drop(columns=["Name", "Ticket"], inplace=True)

# Separar features e variável alvo
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Dividir em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criar modelos base
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_xgb = XGBClassifier(
    objective="binary:logistic", 
    eval_metric="logloss", 
    random_state=42,
    n_estimators=50, 
    max_depth=3, 
    learning_rate=0.1,
    scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1]  # Ajuste de balanceamento
)
model_lr = LogisticRegression(max_iter=5000, random_state=42)

# Criar ensemble (Voting Classifier) usando votação por maioria
ensemble_model = VotingClassifier(
    estimators=[('lr', model_lr), ('rf', model_rf), ('xgb', model_xgb)],
    voting='hard'  
)

# Treinar o ensemble
ensemble_model.fit(X_train, y_train)

# Garantir que não há valores nulos no conjunto de teste
test_df.fillna({
    "Age": test_df["Age"].median(), 
    "Fare": test_df["Fare"].median()
}, inplace=True)

# Separar PassengerId antes de remover
passenger_ids = test_df["PassengerId"]
test_df.drop(columns=["PassengerId"], inplace=True)

# Fazer previsões no conjunto de teste
test_predictions = ensemble_model.predict(test_df)

# Criar DataFrame para submissão
submission = pd.DataFrame({"PassengerId": passenger_ids, "Survived": test_predictions})

# Salvar arquivo para submissão no Kaggle
submission.to_csv("titanic_submission.csv", index=False)

print("Submissão gerada com sucesso!")
