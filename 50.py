5. Naive Bayes
import pandas as pd
from sklearn.model_selection import train_test_split
form sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder #text to num

df = pd.read_csv("data.csv")
df.head()
# conver text into numbers
le = LabelEncoder()
df['Outlook'] = le.fit_transform(df['Outlook'])
df['Temp'] = le.fit_transform(df['Temp'])
df['Humidity'] = le.fit_transform(df['Humidity'])
df['Wind'] = le.fit_transform(df['Wind'])
df['Play'] = le.fit_transform(df['Play'])

# seperate the output
X = df[['Outlook', 'Temp', 'Humidity', 'Wind']]
y = df['Play']

# split the data
X_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

## create model
bnb = BernoulliNB()

## train model
bnb.fit(X_train, y_train)

## predict output
y_pred = bnb.predict(x_test)
print("Predictions: ", y_pred)

## check accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
