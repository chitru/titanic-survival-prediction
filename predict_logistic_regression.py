import pandas as pd
import utils
from sklearn import linear_model, preprocessing

train = pd.read_csv("datas/train.csv")
utils.clean_data(train)

feature_names = ["Pclass", "Fare", "Embarked", "Age", "Sex", "Sibsp", "Parch"]

target = train["Survived"].values
features = train[feature_names].values

#train these objects
classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)

print(classifier_.score(features, target))

poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)

classifier = classifier.fit(poly_features, target)
print(classifier_.score(poly_features, target))