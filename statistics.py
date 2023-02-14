import pandas as pd
df = pd.read_csv('train.csv')
df_count = df.groupby(['emotion']).count()
print(df_count)