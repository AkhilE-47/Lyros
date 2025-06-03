import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Iris.csv')
df.drop('Id',axis=1,inplace=True)


## separate target and independent variables
X = df.drop('Species',axis=1)
y = df['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# print(X_train.shape,X_test.shape)
# print(y_train.shape,y_test.shape)

## Let us build simple random forest classifier model
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

## Let us check the accuracy to see hows our model is performing

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

accuracy_train = accuracy_score(y_train,y_train_pred)
accuracy_test = accuracy_score(y_test,y_test_pred)

# print('accuracy train:',accuracy_train)
# print('accuarcy test',accuracy_test)

import pickle
# Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))
