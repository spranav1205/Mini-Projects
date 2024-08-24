import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("soil_2.csv")

X = df.iloc[:, :100]  # All rows, first 100 columns

# Select the 'soil type' column as the target (y)
Y = df['Soil type']

print(X.shape)
print(len(Y))

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=74,  test_size=0.40,  shuffle=True)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))