# -*- coding: utf-8 -*-
import pandas as pd
import itertools
from itertools import chain
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def word_occurence(df):
    # df['text'] is an instance of class 'pandas.core.series.Series'
    words = df['text'].str.split()  
    # words is an instance of class 'pandas.core.series.Series'
    full_list = list(itertools.chain(*words))
    counts = Counter(full_list)
    index = []
    values = []
    for key, item in counts.items():
        index.append(key)
        values.append(item)
    return pd.DataFrame(data={'occurence':values}, columns=['occurence'], index=index)
        
def n_gram(df, n_gram_size):
    # An n-gram is a contiguous sequence of n items from a given sample of text
    
    # df['text'] is an instance of class 'pandas.core.series.Series'
    tokens = ' '.join([text for text in df['text']])
    tokens = tokens.split()
    ngrams = zip(*[tokens[i:] for i in range(n_gram_size)])
    list = [' '.join(ngram) for ngram in ngrams]
    counts = Counter(list)
    index = []
    values = []
    for key, item in counts.items():
        index.append(key)
        values.append(item)
            
    return pd.DataFrame(data={'occurence':values}, columns=['occurence'], index=index)

def words_cloud(df, show=False):
    # df['text'] is an instance of class 'pandas.core.series.Series'
    all_text = ' '.join([text for text in df['text']])
    wc = WordCloud(width=800, height=500,random_state=21, max_font_size=110, collocations=False).generate(all_text)
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if show:
        plt.show()
    return fig    
    
def n_gram_history(df, n_gram_size, period):
    # n-gram : yearly analysis
    df2 = None

    df3 = df.groupby(pd.Grouper(freq=period))

    for period, group in df3:
        df1 = n_gram(group, n_gram_size)
        df1.rename(columns={'occurence': period}, inplace=True)
        if df2 is None :
            df2 = df1
        else:
            df2 = pd.concat([df2, df1], axis=1, join='outer', sort=True)  

    df2 = df2.fillna(0)
    return df2

def top_n_gram_history(df, n_gram_size, period, limit):
    # n-gram : yearly analysis
    df2 = None
    
    df = n_gram_history(df, n_gram_size, period)

    # To convert to percent of column total, uncomment next line
    #df = df.div(df.sum(axis=0), axis=1).multiply(100)
    
    for period in list(df.columns):
        df1 = df.nlargest(limit, [period])                      # Period top rows
        if df2 is None:
            df2 = df1
        else:
            df2 = pd.concat([df1, df2], join='inner')           # Concatenate period top 10
                
    df2 = df2.groupby(level=0).last()                           # Clean duplicate rows
        
    return df2

def top_n_gram_history_2(df, n_gram_size, period, limit):
    # n-gram : yearly analysis
    df = n_gram_history(df, n_gram_size, period)
    df2 = pd.DataFrame()
    periods = list(df.columns)
    for period in periods:
        top_index_list = list(df.nlargest(limit, [period]).index)
        df2[period] = top_index_list
    return df2    