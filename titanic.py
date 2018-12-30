#data analysis
import pandas as pd
import numpy as np
#data visualization
import matplotlib.pyplot as plt
# %matplotlib inline

train_df = pd.read_csv('datas/train.csv')
test_df = pd.read_csv('datas/test.csv')

#subplot2grid helps to plot graph multiple times
plt.subplot2grid((2,4), (0,0))
train_df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

#looking with age
plt.subplot2grid((2,4), (0,1))
plt.scatter(train_df.Survived, train_df.Age, alpha=0.1)
plt.title("Age wrt Survived")

#looking with class
plt.subplot2grid((2,4), (0,2))
train_df.Pclass.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Class")

#looking with Sex
plt.subplot2grid((2,4), (0,3))
train_df.Sex.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Sex")

plt.subplot2grid((2,4), (1,0), colspan=2)
for x in [1,2,3]:
    train_df.Age[train_df.Pclass == x].plot(kind="kde")
plt.title("Class with Age")
plt.legend(("First", "Second", "Third"))

#looking with Embarked
plt.subplot2grid((2,4), (1,2))
train_df.Embarked.value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Embarked")

plt.show()