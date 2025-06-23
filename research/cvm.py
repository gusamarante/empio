import pandas as pd

df = pd.read_csv("cda_fi_202505/cda_fi_BLC_2_202505.csv", sep=';', encoding='cp860')

print(df)