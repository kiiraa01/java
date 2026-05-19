#find s
import pandas as pd

# READ CSV FILE
data = pd.read_csv("training.csv")

# INITIAL HYPOTHESIS
hypothesis = ['0'] * (len(data.columns) - 1)

# FIND-S ALGORITHM
for i in range(len(data)):

    # CHECK POSITIVE EXAMPLE
    if data.iloc[i, -1] == 'Yes':

        for j in range(len(hypothesis)):

            value = data.iloc[i, j]

            # FIRST POSITIVE EXAMPLE
            if hypothesis[j] == '0':
                hypothesis[j] = value

            # IF VALUES DIFFER
            elif hypothesis[j] != value:
                hypothesis[j] = '?'

# FINAL HYPOTHESIS
print("Final Hypothesis:")
print(hypothesis)
