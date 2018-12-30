import utils
import numpy as np
import pandas as pd 
from sklearn import ensemble, model_selection

train = pd.read_csv("datas/train.csv")
test = pd.read_csv("datas/test.csv")

print("\nCleaning some data")

utils.clean_data(train)
utils.clean_data(test)

print("\nExtracting target and features")

print(train.shape)
target = train["Survived"].values
features_forest = train[["Pclass", "Age", "Sex", "SibSp", "Fare", "Parch", "Embarked"]].values

print("\nRandom Forest Classifier")

forest = ensemble.RandomForestClassifier(
    max_depth = 7,
    min_samples_split = 4,
    n_estimators = 1000,
    random_state = 1,
    n_jobs = -1
)

forest = forest.fit(features_forest, target)

print(forest.feature_importances_)
print(forest.score(features_forest, target))

scores = model_selection.cross_val_score(forest, features_forest, target, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

test_features_forest = test[["Pclass", "Age", "Sex", "SibSp", "Fare", "Parch", "Embarked"]].values
prediction_forest = forest.predict(test_features_forest)
utils.write_prediction(prediction_forest, "titanic_survival_prediction.csv")
