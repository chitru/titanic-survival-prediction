import pandas as pd 
import statsmodels.api as sm 
from patsy import dmatrices
import matplotlib.pyplot as plt 
from sklearn import model_selection 

#loading data
print("Reading file")
df = pd.read_csv("datas/train.csv")

#cleaning data
print("Cleaning Data")
df = df.drop(['Ticket', 'Cabin'], axis=1)
df.Age = df.Age.interpolate()
df.Embarked = df.Embarked.fillna('S')

#running logistic regression
print("Running Logistic Regression")
formula = "Survived ~ C(Pclass) + C(Sex) + Age + SibSp + C(Embarked)"
results = {}

y,x = dmatrices(formula, data=df, return_type="dataframe")

model = sm.Logit(y,x)
res = model.fit()

results["Logit"] = [res, formula]
print(res.summary())

#printing stats
print("\nPrinting some stats.")
plt.figure(figsize=(18,4))

plt.subplot(121)
ypred = res.predict(x)
plt.plot(x.index, ypred, 'bo', x.index, y, 'mo', alpha=.25)
# plt.grid(color='b', linesytle='dashed')
plt.title('Logit Predictions')

plt.subplot(122)
plt.plot(res.resid_dev, 'r-')
plt.grid(color='b', linestyle='dashed')
plt.title('Logit Rediduals')

plt.show()