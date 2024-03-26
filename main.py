import pandas as pd
import numpy as np
import nltk
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

df = pd.read_csv('AI_Human.csv')

X = df['text'].values
y = df['generated'].values

# discarding 90% of the data
X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)

# lowercase all words
X = [word.lower() for word in tqdm(X)]

# remove newline characters
X = [word.replace("\n", "") for word in tqdm(X)]

# remove punctuation and numbers
X = [re.sub(r'[^a-zA-Z\s]', '', word) for word in tqdm(X)]

class Lemmatizer:
    """Removes stopwords and performs lemmatization."""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        
    def lemmatize(self, sentence: str) -> str:
        """Given a sentence, removes stopwords and performs lemmatization."""
        words = nltk.word_tokenize(sentence)
        filtered_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)

lemmatizer = Lemmatizer()
X = [lemmatizer.lemmatize(word) for word in tqdm(X)]

X = np.asarray(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train.reshape(-1, 1), y_train.reshape(-1, 1))