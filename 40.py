import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# DATASET
data = {
    'Weather': ['Sunny', 'Sunny', 'Rainy', 'Rainy'],
    'Temp': ['Hot', 'Mild', 'Hot', 'Cold'],
    'Play': ['No', 'Yes', 'Yes', 'No']
}

# CREATE DATAFRAME
df = pd.DataFrame(data)

# CONVERT TEXT TO NUMBERS
df['Weather'] = df['Weather'].map({
    'Sunny': 0,
    'Rainy': 1
})

df['Temp'] = df['Temp'].map({
    'Hot': 0,
    'Mild': 1,
    'Cold': 2
})

df['Play'] = df['Play'].map({
    'No': 0,
    'Yes': 1
})

# INPUTS
X = df[['Weather', 'Temp']]

# OUTPUT
y = df['Play']

# CREATE MODEL
model = DecisionTreeClassifier(criterion='entropy')

# TRAIN MODEL
model.fit(X, y)

# NEW SAMPLE
sample = [[0, 1]]   # Sunny, Mild

# PREDICTION
prediction = model.predict(sample)

# OUTPUT RESULT
if prediction[0] == 1:
    print("Play = Yes")
else:
    print("Play = No")
