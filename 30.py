candidate- elimination algorithm
import pandas as pd
data = pd.read_csv("training_data.csv")

# NUMBER OF ATTRIBUTES
n = len(data.columns) - 1

# INITIAL SPECIFIC HYPOTHESIS
S = ['0'] * n

# INITIAL GENERAL HYPOTHESIS
G = ['?'] * n

# TRAINING
for i in range(len(data)):

    # ATTRIBUTES
    x = data.iloc[i, :-1].tolist()

    # TARGET
    target = data.iloc[i, -1]

    # POSITIVE EXAMPLE
    if target == 'Yes':

        for j in range(n):

            # FIRST POSITIVE EXAMPLE
            if S[j] == '0':
                S[j] = x[j]

            # GENERALIZE S
            elif S[j] != x[j]:
                S[j] = '?'

    # NEGATIVE EXAMPLE
    else:

        for j in range(n):

            # SPECIALIZE G
            if x[j] != S[j]:
                G[j] = S[j]
            else:
                G[j] = '?'

# FINAL OUTPUT
print("Specific Hypothesis (S):", S)
print("General Hypothesis (G):", G)
