# Adaline
import numpy as np

class Adaline:

    def __init__(self, lr=0.1, epochs=100):

        # LEARNING RATE
        self.lr = lr

        # NUMBER OF ITERATIONS
        self.epochs = epochs

        # RANDOM WEIGHTS
        self.w = np.random.rand(2)

        # RANDOM BIAS
        self.b = np.random.rand()

    def fit(self, X, y):

        # TRAINING
        for epoch in range(self.epochs):

            for i in range(len(X)):

                # PREDICTION
                output = np.dot(X[i], self.w) + self.b

                # ERROR
                error = y[i] - output

                # UPDATE WEIGHTS
                self.w = self.w + self.lr * error * X[i]

                # UPDATE BIAS
                self.b = self.b + self.lr * error

    def predict(self, X):

        output = np.dot(X, self.w) + self.b

        # STEP FUNCTION
        if output >= 0.5:
            return 1
        else:
            return 0


# INPUTS
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

# AND GATE OUTPUT
y = np.array([0,0,0,1])

# CREATE OBJECT
model = Adaline()

# TRAIN MODEL
model.fit(X, y)

# TEST
print(model.predict([0,0]))
print(model.predict([0,1]))
print(model.predict([1,0]))
print(model.predict([1,1]))

# FINAL WEIGHTS
print("Weights:", model.w)
print("Bias:", model.b)
