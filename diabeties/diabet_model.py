from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv(r"C:\Users\eakhi\OneDrive\Desktop\lyros_project\diabeties\Diabetes.csv")

# df.isnull().sum()

# df['Insulin'].unique()


X = df[['Pregnancies','BloodPressure', 'Insulin','BMI', 'Age']]
y = df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

import pickle 
pickle.dump(model, open('dia.pkl', 'wb'))