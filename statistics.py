import pandas as pd

df = pandas.read_csv('train.csv')

df_count = df.groupby(['emotion']).count()

print(df_count)