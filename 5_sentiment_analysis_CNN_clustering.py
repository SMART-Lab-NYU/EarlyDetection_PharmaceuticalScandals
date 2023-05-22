# -*- coding: utf-8 -*-
import time
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import itertools
from collections import Counter
from wordcloud import WordCloud
import os, shutil
import csv

# Execution start here
dir = "data/" # ie: /Users/test/Documents/Leevothyrox/data/
df = pd.read_csv(dir + 'dataset_doctissimo_updated.csv', encoding='utf8')
print('\n*****************\nFile <dataset_doctissimo_updated.csv> has been loaded') 
label = pd.read_csv(dir + 'sentiment_prediction.csv', encoding='utf8')
print('File <sentiment_prediction.csv> has been loaded\n*****************\n') 

# Define frequencies
frequencies = {}
frequencies['Y'] = {'label': 'yearly', 'format': '%Y'}
frequencies['M'] = {'label': 'monthly', 'format': '%Y_%m'}
frequencies['W'] = {'label': 'weekly', 'format': '%Y_%U'}
frequencies['D'] = {'label': 'daily', 'format': '%Y_%m_%d'}

def word_occurence(df):
    words = df['text'].str.split()  
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

def words_cloud(period, df, show=False):
    #fig = None
    wc = None
    all_text = ' '.join([text for text in df['text']])
    all_text = all_text.strip()
    try:    
        wc = WordCloud(width=800, height=500,random_state=21, max_font_size=110, collocations=False).generate(all_text)
    except ValueError:
        print('Value Error: ', period.strftime('%Y_%m_%d'), 'text: [', all_text,']')
        return None
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    if show:
        plt.show() 
    plt.close(fig) # Close the window displaying WC (too much memory used)   
    return fig    
    
def sentiment_number(row):
   if row['sentiment'] == '__label__positive':
      return 1
   if row['sentiment'] == '__label__negative':
      return 0

def save_csv(df):
    for key, frequency in frequencies.items():
        df_grouped = df.groupby(pd.Grouper(freq=key))
        index = 0
        f = open(dir+'clustering/' + frequency['label'] + '.csv', 'w')
        for period, group in df_grouped:
            if len(group) > 0:
                df1 = word_occurence(group).nlargest(10, ['occurence'])
                lst_1 = []
                if not df1.empty:
                    lst_1 = [text for text in df1.index]
                if len(lst_1) < 10:
                    lst_1.extend(['']*(10-len(lst_1)))
                fragment_1 = ','.join(lst_1)             
                df2 = n_gram(group, 2).nlargest(10, ['occurence'])
                lst_2 = []
                if not df2.empty:
                    lst_2 = [text for text in df2.index]
                if len(lst_2) < 10:
                    lst_2.extend(['']*(10-len(lst_2)))
                fragment_2 = ','.join(lst_2)
                xs_p = 100*len(group[group['sentiment'] == '__label__positive'])/len(group)
                xs_n = 100*len(group[group['sentiment'] == '__label__negative'])/len(group)
                line = str(index) + ',' + period.strftime(frequencies[key]['format']) + ',' + fragment_1 + ',' + fragment_2 + ',' + str(xs_p) +',' + str(xs_n) + '\n'
                f.write(line)
            index+=1
        f.close() 

# Save WC in folders : yearly - monthly - weekly - daily and sort by normal_0 or abnormal_1 tag // 30 min of execution time
def save_word_clouds(df, start, end):
    # Define intervals
    intervals = []
    # Normal 
    intervals.append({'index':0, 'name':'normal', 'mask':(df.index < start) | (df.index > end)})
    # Abnormal
    intervals.append({'index':1, 'name':'abnormal', 'mask':(df.index >= start) & (df.index <= end)})
    for item in intervals:  
        for key, frequency in frequencies.items():
            df_grouped = df[item['mask']].groupby(pd.Grouper(freq=key))
            for period, group in df_grouped:
                if len(group.index) > 0:
                    file_path = dir+'cnn/'+ frequency['label'] + '/' + item['name'] + '/'  + 'world_cloud_' + frequency['label'] + '_' + period.strftime(frequency['format']) + '.png'
                    #print(period, file_path)
                    fig = words_cloud(period, group)
                    if not fig is None:
                        fig.savefig(file_path)
''' 
def clean_folders():
    folders = []
    folders.append(dir+'clustering')
    folders.append(dir+'cnn/yearly/normal')
    folders.append(dir+'cnn/monthly/normal')
    folders.append(dir+'cnn/weekly/normal')
    folders.append(dir+'cnn/daily/normal')
    folders.append(dir+'cnn/yearly/abnormal')
    folders.append(dir+'cnn/monthly/abnormal')
    folders.append(dir+'cnn/weekly/abnormal')
    folders.append(dir+'cnn/daily/abnormal')
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
              print('Failed to delete %s. Reason: %s' % (file_path, e))

# Execution starts here   
clean_folders()
print('All files in clustering & CNN folders are deleted\n******************************')
'''
# DF formating & CSV export for clustering
print('DF label file (sentiment_prediction.csv)\n******************************')
print(label)
print('\nDF without labeling\n******************************')
print(df)

df['sentiment'] = label['prediction'] # Add a column
df['date'] = pd.to_datetime(df['date']) # Cast date to datetime
df['text'] = df['text'].astype(str) # Cast text column to string
df['text'] = df['text'].str.replace('[{}]'.format(string.punctuation), '') # Clean text

df = df.set_index('date') # Set date column as index
print('\nDF with labeling\n******************************')
print(df)

# Add a numeric column reflecting sentiment value
df['sentiment_number'] = df.apply (lambda row: sentiment_number(row), axis=1)
print('\nDF with labeling and sentiment number\n******************************')
print(df)

# Save labeled dataframe to csv
df.to_csv(dir + 'dataset_doctissimo_updated_labeled.csv', sep=',', header=True, encoding='utf8')
# Generate and save top_10 words, top_10 bi-grams, +/- sentiments in csv files
save_csv(df)
print('\n******************************')
print('CSV files are saved in dir+clustering/ : yearly.csv, monthly.csv, weekly.csv, daily.csv')

# Charts & plots
# Distribution of messages by sentiment
# All dates
fig, ax = plt.subplots(figsize=(20, 10))
df['sentiment_number'].value_counts().plot(ax=ax, kind='bar')
print('Distribution of messages by sentiment (0: negative / 1: positive):\n-> all dates\n******************************')
plt.show()
# Selected year
fig, ax = plt.subplots(figsize=(20, 10))
df[df.index.year == 2016]['sentiment_number'].value_counts().plot(ax=ax, kind='bar')
print('\nDistribution of messages by sentiment (0: negative / 1: positive):\n-> 2017\n******************************')
plt.show()
# Selected month
fig, ax = plt.subplots(figsize=(20, 10))
df[(df.index.year == 2017) & (df.index.month == 2)]['sentiment_number'].value_counts().plot(ax=ax, kind='bar')
print('\nDistribution of messages by sentiment (0: negative / 1: positive):\n-> 2017-02\n******************************')
plt.show()
# Selected day
fig, ax = plt.subplots(figsize=(20, 10))
df[df.index == '2017-03-18']['sentiment_number'].value_counts().plot(ax=ax, kind='bar')
print('\nDistribution of messages by sentiment (0: negative / 1: positive):\n-> 2017-03-18\n******************************')
plt.show()
# Selected range
fig, ax = plt.subplots(figsize=(20, 10))
df[(df.index >= '2017-03-18') & (df.index < '2017-03-25')]['sentiment_number'].value_counts().plot(ax=ax, kind='bar')
print('\nDistribution of messages by sentiment (0: negative / 1: positive):\n-> 2017-03-18 to 2017-03-25\n******************************')
plt.show()
# Historical line chart per selected sentiment
freq = pd.offsets.Day(30)
fig, ax = plt.subplots(figsize=(20, 10))
ax = df[(df.index.year == 2017) & (df['sentiment'] == '__label__positive')]['sentiment_number'].resample(freq).sum().plot.line(ax=ax)
print('\nHistorical line chart per selected sentiment (__label__positive) in 2017\n******************************')
plt.show()
# Historical line chart of comments per user
fig, ax = plt.subplots(figsize=(20, 10))
df[(df.index.year == 2017) & (df.index.month == 2)]['user'].value_counts().plot(ax=ax, kind='bar')
print('\nComments per user in 2017-02\n******************************')
plt.show()
'''
# WC saving for CNN algorithm
start = '2017-07-01' # Start of date range qualified as abnormal
end = '2017-12-31' # End of date range qualified as abnormal
print('Start of date range qualified as abnormal : ' + start + '\nEnd of date range qualified as abnormal : ' + end + '\n*****************************\n')
print('Generating .png files...')
save_word_clouds(df, start, end) # Generate and save wordclouds as .png image files 
print('\n*****************************\n.png files have been saved in dir+cnn/\n*****************************\n')
'''