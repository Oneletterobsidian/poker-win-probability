import pandas as pd
df = pd.read_csv("poker_dataset.csv")
print(df["win_probability"].describe().round(3))