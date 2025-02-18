import pandas as pd
from xgboost import XGBClassifier

# Data Import
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Data Cleaning
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

train["HasCabin"] = train["Cabin"].notnull().astype(int)
test["HasCabin"] = test["Cabin"].notnull().astype(int)
train.drop(columns=["Cabin"], inplace=True)
test.drop(columns=["Cabin"], inplace=True)

train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
test["Embarked"] = test["Embarked"].fillna(test["Embarked"].mode()[0])
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})

train = pd.get_dummies(train, columns=["Embarked"], drop_first=True)
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)

train.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)
passenger_ids = test["PassengerId"]
test.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True)

# Target Column 
X = train.drop("Survived", axis=1)
y = train["Survived"]

# XGBoost best Parameter model
final_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=1,
    reg_lambda=1,
    reg_alpha=0.5,
    random_state=42
)

final_model.fit(X, y)

y_pred_test = final_model.predict(test)

#Submission
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": y_pred_test
})

submission.to_csv("submission.csv", index=False)
print("Arquivo submission.csv criado.")
