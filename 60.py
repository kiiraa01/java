from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# DOCUMENTS
docs = [
    "I love cricket",
    "Cricket is amazing",
    "I hate rain",
    "Rain is terrible"
]

# LABELS
# sports = 1
# weather = 0
y = [1, 1, 0, 0]

# CONVERT TEXT TO NUMBERS
cv = CountVectorizer()
X = cv.fit_transform(docs)

# MODEL
model = MultinomialNB()

# TRAIN MODEL
model.fit(X, y)

# TEST DATA
test_docs = [
    "I love rain",
    "cricket is good"
]

# CONVERT TEST DATA
X_test = cv.transform(test_docs)

# PREDICTION
pred = model.predict(X_test)

print("Predictions:", pred)

# ACTUAL OUTPUT
y_true = [0, 1]

# ACCURACY
print("Accuracy:", accuracy_score(y_true, pred))

# PRECISION
print("Precision:", precision_score(y_true, pred))

# RECALL
print("Recall:", recall_score(y_true, pred))
