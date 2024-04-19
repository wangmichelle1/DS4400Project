import pandas as pd
import numpy as np
import nltk
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Read in original dataset as a pandas dataframe
df = pd.read_csv('AI_Human.csv')

# Split data into X and y 
X = df['text'].values
y = df['generated'].values

# Discard 90% of the data
X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)

# Lowercase all words
X = [word.lower() for word in tqdm(X)]

# Remove newline characters
X = [word.replace("\n", "") for word in tqdm(X)]

# Remove punctuation and numbers
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

# Perform lemmazation
lemmatizer = Lemmatizer()
X = [lemmatizer.lemmatize(word) for word in tqdm(X)]
# Cast x into an array
X = np.asarray(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Undersample the data to account for class imbalance
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

np.save('X_train.npy', X_resampled)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_resampled)
np.savee('y_test.npy', y_test)
