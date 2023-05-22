# -*- coding: utf-8 -*-
from fasttext import train_supervised
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn 
from gensim.models.fasttext import FastText
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import string
import nltk
from nltk.corpus import stopwords
from io import StringIO
import csv
import time

# Execution start here
dir = "data/" # ie: /Users/test/Documents/Leevothyrox/data/
vector_size = 60
window = 40
min_count = 3
sample = 1e-2
test_size = 0.2                             # test/(test+train)
# Limiting datasets size during test phase
max_tweets = None                          # Or None / 10000
max_texts = None                             # Or None / 100

df = pd.read_csv(dir + 'dataset_doctissimo_updated.csv')
# Limit size for initial testing
if not max_texts is None:
    df = df.sample(n=max_texts)
   
# Fasttext (Word2Vec) machine learning model
word_tokenized_corpus = df['text'].str.split()
start = time.time()
ft_model = FastText(word_tokenized_corpus,
                    vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    sample=sample,
                    sg=1,
                    epochs=20)    
                    
end = time.time()
print()
print('Elapsed time', end - start, 's')
print()
print('Keyed vectors for levotyrox')
print(ft_model.wv['levothyrox'])
semantically_similar_words = {words: [item[0] for item in ft_model.wv.most_similar([words], topn=5)] for words in ['levothyrox', 'formul', 'secondair', 'sang', 't4', 'hormon']}
print()
print("Semantically similar words to ['levothyrox','formul', 'secondair', 'sang', 't4', 'hormon']" )
for k, v in semantically_similar_words.items():
    print(k + ':' + str(v))
print()
print('Similarity levothyrox / secondair', ft_model.wv.similarity(w1='levothyrox', w2='secondair'))
print()
print('Similarity levothyrox / formul', ft_model.wv.similarity(w1='levothyrox', w2='formul'))
all_similar_words = sum([[k] + v for k, v in semantically_similar_words.items()], [])
print()
print('Similar words')
print(all_similar_words)

word_vectors = ft_model.wv[all_similar_words]

pca = PCA(n_components=2)

p_comps = pca.fit_transform(word_vectors)
word_names = all_similar_words

plt.figure(figsize=(20, 10))
plt.scatter(p_comps[:, 0], p_comps[:, 1], c='red')

for word_names, x, y in zip(word_names, p_comps[:, 0], p_comps[:, 1]):
    plt.annotate(word_names, xy=(x+0.06, y+0.03), xytext=(0, 0), textcoords='offset points')

plt.show()

# French_tweets
french_tweets = pd.read_csv(dir + 'french_tweets_updated.csv')

# Limit size for initial testing
if not max_tweets is None:
    french_tweets = french_tweets.sample(n=max_tweets)

# Split train/test samples : option 2
train, test = train_test_split(french_tweets, test_size=test_size)

# Save train to csv
train.to_csv(dir + 'train.csv', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")

model = train_supervised(dir + 'train.csv', epoch=10)

predictions = []
for line in test['text']:
    pred_label = model.predict(line, k=-1, threshold=0.5)[0][0]
    predictions.append(pred_label)

# you add the list to the dataframe, then save the datframe to new csv
test['prediction'] = predictions

s_positive = len(test[test['sentiment']=='__label__positive'].index)
s_negative = len(test[test['sentiment']=='__label__negative'].index)
p_positive = len(test[test['prediction']=='__label__positive'].index)
p_negative = len(test[test['prediction']=='__label__negative'].index)
success = len(test[test['prediction']==test['sentiment']].index)
total = len(test.index)
print(s_positive, s_negative, p_positive, p_negative)
print('Success rate: ', success/total*100, '%')


# Drop all columns except the 'text' column
df = df[['text']]

predictions=[]
for line in df['text']:
    pred_label = model.predict(line, k=-1, threshold=0.5)[0][0]
    predictions.append(pred_label)

df['prediction'] = predictions
print()
print('All sentiments')
print(df.head())
print()
print('Positive sentiment')
print(df[df['prediction']=='__label__positive'].head())
print()
print('Negative sentiment')
print(df[df['prediction']=='__label__negative'].head())
n_positive = len(df[df['prediction']=='__label__positive'].index)
n_negative = len(df[df['prediction']=='__label__negative'].index)
print('Count positive', n_positive)
print('Count negative', n_negative)

df.to_csv(dir + 'sentiment_prediction.csv')
