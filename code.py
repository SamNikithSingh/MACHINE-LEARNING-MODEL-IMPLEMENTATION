import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("MACHINE LEARNING MODEL IMPLEMENTATION")
print("DATASET USED IS TITANIC DATASET")

df = sns.load_dataset('titanic')
print(df.head())


df = df.drop(['deck', 'embark_town', 'alive', 'who', 'class'], axis=1)
df = df.dropna(subset=['age', 'embarked', 'embarked'])


le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])           # male=1, female=0
df['embarked'] = le.fit_transform(df['embarked']) # C=0, Q=1, S=2
df['alone'] = df['alone'].astype(int)


X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']]
y = df['survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


importances = model.feature_importances_
feat_names = X.columns
sns.barplot(x=importances, y=feat_names)
plt.title("Feature Importance")
plt.show()
