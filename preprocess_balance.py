import pandas as pd
import os

TRAIN_FILE = os.path.join("common-voice-12", "train.csv")
BALANCED_TRAIN_FILE = os.path.join("common-voice-12", "train_balanced.csv")

if __name__ == '__main__':
	df = pd.read_csv(TRAIN_FILE)
	age_group = df.groupby(['age'])
	df_age_count = age_group.count()
	
	min_age = df_age_count['file'].min()
	print("Min group samples:", min_age)

	balanced = age_group.sample(min_age).reset_index(drop=True)
	print("Balanced groups (head):", balanced.head(20))
	
	balanced.to_csv(BALANCED_TRAIN_FILE, index=False)
	
	