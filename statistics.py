import pandas as pd
import os
df = pd.read_csv(os.path.join('common-voice-12', 'train.csv'))

df_emotion_count = df.groupby(['emotion']).count()
print(df_emotion_count)

df_age_count = df.groupby(['age']).count()
print(df_age_count)

df_age_count = df.groupby(['gender']).count()
print(df_age_count)