from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

test_vectorizer = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
X_test_tfidf = test_vectorizer.fit_transform(X_test)