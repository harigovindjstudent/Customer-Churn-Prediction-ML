import pandas as pd

df = pd.read_csv("data/Churn_Modelling.csv")
print(df["Exited"].value_counts(normalize=True) * 100)