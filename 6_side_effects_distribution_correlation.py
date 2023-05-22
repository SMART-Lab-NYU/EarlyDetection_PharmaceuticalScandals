# -*- coding: utf-8 -*-
import time
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
import csv
import re
import sys
import seaborn as sns
from wordcloud import WordCloud
from common import n_gram, n_gram_history, top_n_gram_history
#nltk.download('punkt')

def get_side_effects(text, side_effects):
    se_list = []
    for regex in list(side_effects):
        if re.search(regex, str(text)):
            result = re.search(regex, str(text)).group()
            result = result.replace(' ', '')
            result = result.replace('\'', '')
            result = result.replace('[', '')
            result = result.replace(']', '')
            result = result.replace(',', ' ')
            se_list.append(result)
    return se_list
    
# Execution start here
dir = "data/" # ie: /Users/test/Documents/Leevothyrox/data/
side_effects_min_freq = 10
#pd.options.mode.chained_assignment = None

levo_side_effects_extended = ['fatigu\S*', 'astheni\S*',
               'insomni\S*',
               'ma\S* d\S* tete', 'ma\S* \S* d\S* tete' 'cephal\S*',
               'vertig\S*',
               'depressi\S*', 'deprim\S*', 'suicid\S*',
               'douleur\S* musculair\S*', 'douleur\S* \S* musculair\S*', 'myalgi\S*',
               'douleur\S* articulair\S*', 'douleur\S* \S* articulair\S*', 'douleur\S* a\S* articulation\S*', 'douleur\S* \S* a\S* articulation\S*', 'douleur\S* d\S* articulation\S*', 'douleur\S* \S* d\S* articulation\S*', 'douleur\S* articulation\S*', 'douleur\S* \S* articulation\S*', 'arthralgi\S*',
               'chut\S* d\S* cheveu\S*', 'chut\S* \S* d\S* cheveu\S*', 'chut\S* cheveu\S*', 'chut\S* \S* cheveu\S*', 'pert\S* cheveu\S*', 'pert\S* \S* cheveu\S*', 'pert\S* d\S* cheveu\S*', 'pert\S* \S* d\S* cheveu\S*',
               'pri\S* d\S* poid\S*', 'pri\S* \S* d\S* poid\S*', 'prendre poid\S*', 'prendre \S* poid\S*', 'prendre d\S* poid\S*', 'prendre \S* d\S* poid\S*',
               'pert\S* poid\S*', 'pert\S* \S* poid\S*', 'pert\S* d\S* poid\S*', 'pert\S* \S* d\S* poid\S*',
               'troubl\S* memoir\S*', 'troubl\S* \S* memoir\S*', 'troubl\S* \S* \S* memoir\S*', 'troubl\S* \S* \S* \S* memoir\S*',
               'anxie\S*',
               'nervosit\S*', 'nerveu\S*', 'irritabilit\S*', 'irritabl\S*',
               'nausee', 'nauseeu\S*',
               'diar\S*',
               'constip\S*',
               'sue', 'suee', 'suees', 'suer', 'sueur\S*', 'transpi\S*',
               'acouphen\S*',
               'tachycardi\S*','arythmi\S*', 'hyperten\S*', 'hypoten\S*', 'hyper ten\S*', 'hypo ten\S*']

# Read data
df = pd.read_csv(dir + 'dataset_doctissimo_updated_labeled.csv', encoding='utf8')
print('\n*****************\nFile <dataset_doctissimo_updated_labeled.csv> has been loaded\n*****************\n')
# Drop useless columns
df = df.drop(columns = ['user', 'url', 'year', 'words_count', 'sentiment'])
# Cast date to datetime
df['date'] = pd.to_datetime(df['date'])
# Set date column as index
df = df.set_index('date')
# Side effects listed in text
df['side_effect_count'] = df['text'].str.count(r'\b|\b'.join(levo_side_effects_extended))  

# Downsample to sample size = 1 day
ONEDAY = pd.offsets.Day(1)
df_daily = df.resample(ONEDAY)["side_effect_count"].sum()

# Daily occurences of side effects reported in messages, all dates
fig, ax = plt.subplots(figsize=(20, 10))
ax = df_daily.plot.line(ax=ax)
#plt.show()
plt.savefig(dir + 'data/SE-2016-2020.png')
print('\n*****************\nDaily occurence in 2016-2020... Check\n*****************\n')

# Daily occurences of side effects reported in messages, 2017
fig, ax = plt.subplots(figsize=(20, 10))
ax = df_daily[df_daily.index.year==2017].plot.line(ax=ax)
#plt.show()
plt.savefig(dir + 'data/SE-2017.png')
print('\n*****************\nDaily occurence in 2017... Check\n*****************\n')

# Normalize 
fig, ax = plt.subplots(figsize=(20, 10))
df_normalized = (df_daily - df_daily.mean())/df_daily.std()
ax1 = df_normalized.rolling(window=30).mean().plot.line(ax=ax)
#plt.show()
plt.savefig(dir + 'data/SE-2017-normalized.png')
print('\n*****************\nNormalize... Check\n*****************\n')

df['side_effects'] = df['text'].apply(get_side_effects, side_effects=levo_side_effects_extended)
print('\n*****************\nDataframe side effects\n*****************\n')
print(df['side_effects'].head(20))

# Build most common side effects list
a = ' '.join(np.concatenate(df['side_effects']))
words = nltk.word_tokenize(a)
word_dist = nltk.FreqDist(words)
most_common_side_effects = pd.DataFrame(list(filter(lambda x: x[1] >= side_effects_min_freq, word_dist.items())), columns = ['side_effect', 'frequency'])
mcse_list = most_common_side_effects['side_effect'].to_list()

# Restrict side effects to most_common_side_effects and store in column  
df['most_common_side_effects'] = df['side_effects'].apply(lambda l :[x for x in l if np.isin(x, mcse_list)])

# New dataframe
vector = pd.DataFrame(columns = mcse_list)
for side_effect in mcse_list:
    vector[side_effect] = df.loc[df['most_common_side_effects'].astype(bool)]['most_common_side_effects'].apply(lambda l: int(side_effect in l))
# For testing purpose 
#print(vector.astype(bool).sum(axis=0))
print('\n*****************\nVector\n*****************\n')
print(vector.head(20))

# Correlation matrix
correlations = vector.corr()
print('\n*****************\nCorrelation matrix\n*****************')
print(correlations.head(20))

cor_1 = correlations.where(np.triu(np.ones(correlations.shape)).astype(bool)).stack().sort_values(ascending=False).reset_index()
cor_1 = cor_1.loc[cor_1['level_0'] != cor_1['level_1']]
cor_1.columns = ['side_effect_1','side_effect_2','Correlation']
print(cor_1.head(20))

# plot the correlation matrix
plt.figure(figsize=(20,20))
sns.heatmap(correlations, cmap='RdBu', vmin=-1, vmax=1, square = True, cbar_kws={'label':'correlation'})
#plt.show()
plt.savefig(dir + 'data/heatmap.png')

all_text = ' '.join([text for text in df['side_effects'].apply(str).str.replace('[{}]'.format(string.punctuation), '', regex=True)])
print('\n*****************\nWC Generating...\n*****************')
print('Number of words in all_text:', len(all_text))
wordcloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110, collocations=False).generate(all_text)
plt.figure(figsize=(20, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off');
#plt.show()
plt.savefig(dir + 'data/SE-WC.png')

df1 = df.copy()
# Create a 'text' column for use in N_gram and top_N-gram functions
# This has to be improved
# Option 0 : pass a series
# Option 1 : pass the column name in the finctions parameters
# Option 2 : limite de dataframe to a single column
df1['text'] = df1['side_effects'].apply(lambda l: ' '.join(l))
df2 = n_gram_history(df1, 2, 'Y')
print('\n*****************\nn_gram_history\n*****************')
print(df2.head(20))
print('\n*****************\ntop_n_gram_history\n*****************')
df3 = top_n_gram_history(df1, 2, 'Y', 10)
print(df3.head(20))

# Save file
#df.to_csv(dir + 'dataset_doctissimo_side_effect_1.csv', index=False, sep=',', header=True, encoding='utf8')
