import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing

data = pd.read_excel("Asssignment4_data.xlsx").drop('House ID', axis=1)
test = pd.read_excel("Asssignment4_data.xlsx", sheet_name=1).drop('House ID', axis=1)

inputsData = data.drop('Construction type', axis=1)
targetsData = data['Construction type']

inputsTest = test.drop('Construction type', axis=1)
targetsTest = test['Construction type']

le_conType = preprocessing.LabelEncoder()
targetsDataEncoded = le_conType.fit_transform(targetsData)
targetsTestEncoded = le_conType.fit_transform(targetsTest)

model = tree.DecisionTreeClassifier()

model.fit(inputsData, targetsDataEncoded)

print("\nmodel depth")
print(model.get_depth())
print("\nmodel leaves")
print(model.get_n_leaves())

print("\n",model.score(inputsData, targetsDataEncoded))
print(model.score(inputsTest, targetsTestEncoded))