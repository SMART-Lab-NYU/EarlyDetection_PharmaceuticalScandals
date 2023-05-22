# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unicodedata as unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import re
import string
import time

class DenseTransformer():
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return X.todense()

# Stemmers remove morphological affixes from words, leaving only the word stem.
ps = PorterStemmer()

# Execution start here
dir = "data/" # ie: /Users/test/Documents/Leevothyrox/data/
n = 10000   # tweets subset size (testing)
n2 = 100    # df_doctissimo subset size (testing)

t0 = time.process_time()

tweets = pd.read_csv('french_tweets_updated.csv')

Ntotal = len(tweets)
print()
print('Tweets file size', Ntotal)

t1 = time.process_time()
elapsed_time10 =t1 - t0

# Limit the size for the test phase
tweets = tweets.sample(n=n)

#print(tweets)    
#List to hold cleaned tweets and labels
X = [word for word in tweets['text']]
y = list(tweets['sentiment'].values)
#print(X)

t2 = time.process_time()
elapsed_time21 =t2 - t1

print()
print('Time to load', n, 'rows', elapsed_time10, 's')
print('Time to clean', n, 'rows', elapsed_time21, 's')
print('Time to clean full dataset', Ntotal, 'rows', Ntotal*elapsed_time21/n/60, 'mn')

# First you split to train/split and then you train all the steps of your model.
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Use Pipeline as your classifier, this way you don't need to keep calling a transform and fit all the time.
classifier = Pipeline([('cv', CountVectorizer(max_features=300)), ('to_dense', DenseTransformer()), ('n_b', GaussianNB())])

# Here you train all steps of your Pipeline in one go.
classifier.fit(X_train, y_train)

t3 = time.process_time()
elapsed_time32 =t3 - t2
print('Time to fit model', elapsed_time32, 'size', n)

y_pred  = classifier.predict(X_test)

combined = np.vstack((y_test, y_pred)).T
comb = pd.DataFrame(data=combined, columns = ['test', 'predicted'])
x1 = len(comb[comb['test'] == comb['predicted']])

print()
print('Sample size', len(comb))
print('Prediction accuracy on sample', 100*x1/len(comb), '%')

# Load blog messages
df = pd.read_csv(dir + 'dataset_doctissimo_updated.csv')

# Cast date to datetime
df['date'] = pd.to_datetime(df['date'])

# Cast text column to string
df['text'] = df['text'].astype(str)

# Set date column as index
df = df.set_index('date')

# Extract subset for testing purpose only
df = df.head(n2)

# Predict sentiment
to_predict = [word for word in df['text']]
predicted = classifier.predict(to_predict)

df = df.assign(sentiment=predicted)

print()
print(df[['text', 'sentiment']])

# Save result
df.to_csv(dir+'with_polarity.csv')