import pandas as pd
import os
pd.set_option('display.max_colwidth', None)
df = pd.read_csv(os.path.join('common-voice-12', 'validated.tsv'), sep="\t")
print(df.head)

#df_emotion_count = df.groupby(['emotion']).count()
#print(df_emotion_count)

df_age_count = df.groupby(['age']).count()
print(df_age_count)

df_gender_count = df.groupby(['gender']).count()
print(df_gender_count)

df_accents_count = df.groupby(['accents']).count()
print(df_accents_count)

df_locale_count = df.groupby(['locale']).count()
print(df_locale_count)