import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

# Read in file
df = pd.read_csv('data/sample.csv')

# Define x and y
X = df['text'].values
y = df['generated'].values

# Text cleaning
# Convert all words to lower case
X = [string.lower() for string in X]

# Remove \n
X = [string.replace("\n", "") for string in X]

# How should we deal with words with apostrophes like "don't"
# Remove special characters - do we want to or not?
# Tokenization
# Remove stop words?
# Lemmatization
# Vectorization
# Possible creation of new features

# Split data into training, validation, and testing sets - 80/10/10
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_train = X_train.reshape((-1, 1))
X_val = X_val.reshape((-1, 1))
X_test = X_test.reshape((-1, 1))

# Perform undersampling to address class imbalance
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)


# Model Training and Evaluation