import pandas as pd
import numpy as np


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("\nDados Base")
print(train.head())
print(train.isnull().sum())

#Lidando com valores nulos
#Idade
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

#Cabine - Cria uma coluna só para saber se tem cabine ou não, deleta o valor de cabines
train["HasCabin"] = train["Cabin"].notnull().astype(int)
test["HasCabin"] = test["Cabin"].notnull().astype(int)
train.drop(columns=["Cabin"], inplace=True)
test.drop(columns=["Cabin"], inplace=True)

#Porto de Embarque
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
test["Embarked"] = test["Embarked"].fillna(test["Embarked"].mode()[0])

#Tarifa
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

#Convertendo variaveis
#Genero
train["Sex"] = train["Sex"].map({"male":0,"female":1})
test["Sex"] = test["Sex"].map({"male":0,"female":1})

#Porto de embarque
train = pd.get_dummies(train, columns=["Embarked"],drop_first=True)
test = pd.get_dummies(test, columns=["Embarked"],drop_first=True)

#Deletando colunas inuteis
#ID de passageiro, Nome, Numero do Ticket
train.drop(columns=["PassengerId","Name","Ticket"], inplace= True)
test.drop(columns=["PassengerId","Name","Ticket"], inplace= True)

print("\nDados Tratados")
print(train.head())
print(train.isnull().sum()) 

#Treino e Teste
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV



X = train.drop("Survived", axis=1)
Y = train["Survived"]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#MODELO REGRESSÃO LOGISTICA
print("\nRegressão Logistica")
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression(random_state=42)
param_LR = [
    {"solver": ["liblinear"], "penalty": ["l1", "l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100], "max_iter": [5000, 10000]},
    {"solver": ["lbfgs"], "penalty": ["l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100], "max_iter": [5000, 10000]},
]

gridsearch_LR = GridSearchCV(model_LR, param_LR, cv=3, n_jobs=-1, verbose=0)
gridsearch_LR.fit(X_train, Y_train)

print("Melhores parametros para Regressão Logistica:")
print(gridsearch_LR.best_params_)

top_LR = gridsearch_LR.best_estimator_
LR_Y_pred = top_LR.predict(X_val)
acc_LR = accuracy_score(Y_val, LR_Y_pred)
print(f"Precisão: {acc_LR:.4f}")

#MODELO RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
print("\nClassificador Random Forest")

model_RFC = RandomForestClassifier(random_state=42)

param_RFC = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

gridsearch_RFC = GridSearchCV(model_RFC,param_RFC,cv=5,n_jobs=-1, verbose=0)

gridsearch_RFC.fit(X_train,Y_train)

print("Melhores parametros para Random Forest:")
print(gridsearch_RFC.best_params_)

top_RFC = gridsearch_RFC.best_estimator_
RFC_Y_pred = top_RFC.predict(X_val)
acc_RFC = accuracy_score(Y_val,RFC_Y_pred)
print(f"Precisão:{acc_RFC:.4f}")


#MODELO XGBOOST
from xgboost import XGBClassifier
print("\nClassificador XGBoost")
model_XGB = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
param_XGB = {
    "n_estimators": [300,500,700],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3,5,7],
    "subsample": [0.5, 0.7, 1.0],
    "colsample_bytree": [0.5,0.7,1.0]
}

gridsearch_XGB = GridSearchCV(model_XGB, param_XGB, cv=3, n_jobs=-1, verbose=0)

gridsearch_XGB.fit(X_train,Y_train)
print("Melhores parametros XGBoost:")
print(gridsearch_XGB.best_params_)

top_XGB = gridsearch_XGB.best_estimator_
XGB_Y_pred = top_XGB.predict(X_val)
acc_XGB = accuracy_score(Y_val,XGB_Y_pred)
print(f"Precisão:{acc_XGB:.4f}")

