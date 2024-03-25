import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv('AI_Human.csv')

X = df['text'].values
y = df['generated'].values

# discarding 90% of the data
X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)

X = [word.lower() for word in tqdm(X)]

# add more preprocessing here

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train.reshape(-1, 1), y_train.reshape(-1, 1))