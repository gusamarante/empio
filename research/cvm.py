import pandas as pd

df = pd.read_csv("cda_fi_BLC_4_202505.csv", sep=';', encoding='cp860')

print(df)
df.to_clipboard()