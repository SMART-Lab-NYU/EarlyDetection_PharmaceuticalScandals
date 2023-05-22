# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from common import word_occurence, n_gram, words_cloud

def word_occurence_history(df, period):
    df2 = None
    df3 =df.groupby(pd.Grouper(freq=period))
    for period, group in df3:
        df1 = word_occurence(group)
        df1.rename(columns={'occurence': period}, inplace=True)       
        if df2 is None :
            df2 = df1
        else:
            df2 = pd.concat([df2, df1], axis=1, join='outer', sort=True)
    df2 = df2.fillna(0) 
    return df2   
 
def top_word_occurence_history(df, period, limit):
    df2 = None
    df = word_occurence_history(df, period)
    # Convert to percent of column total
    #df = df.div(df.sum(axis=0), axis=1).multiply(100)
    periods = list(df.columns)
    for per in periods :
        df1 = df.nlargest(limit, [per])                     # Period top rows  
        if df2 is None :
            df2 = df1
        else:
            df2 = pd.concat([df1, df2], join='inner')       # Concatenate period top 10   
    df2 = df2.groupby(level=0).last()                       # Clean duplicate rows
    return df2   

def n_gram_history(df, n_gram_size, period):
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
    df = n_gram_history(df, n_gram_size, period)
    df2 = pd.DataFrame()
    periods = list(df.columns)
    for period in periods:
        top_index_list = list(df.nlargest(limit, [period]).index)
        df2[period] = top_index_list
    return df2

def getCorrelations(df):
    df_cor_data = []
    df = df.transpose()
    
    columns_list = list(df.columns)
    for i in range(0, len(columns_list)):
        col_1 = columns_list[i]
        for j in range (i+1, len(columns_list)):
            col_2 = columns_list[j]    
            df1 = pd.merge(df[col_1], df[col_2], left_index=True, right_index=True, how='inner', suffixes=('_left', '_right'))
            cor = df.corr()[col_1][col_2]
            df_cor_data.append({'Item 1':col_1, 'Item 2':col_2, 'correlation':cor})
            #cor = np.corrcoef(security_1.data.loc[start:end]['Log_Returns(AdjClose)'], security_2.data.loc[start:end]['Log_Returns(AdjClose)'])[1, 0]
    df_cor = pd.DataFrame(df_cor_data)
    df_cor = df_cor.sort_values(by='correlation', ascending=False)
    return df_cor
   
   
def getTopCorrelations(df):
    df = getCorrelations(df)
    df = df.loc[(df['correlation']>0.95)]
    #df = df.sort_values(by=['Item 1', 'Item 2'])
    df = df.sort_values(by=['correlation'], ascending=False)
    return df
 
# Added 2021/09/21
def getTopCorrelations_2(df):
    # df : historical dataframe
    df = df.transpose()
    dict = {}
    for year in range(2016, 2020):
        df0 = df.loc[df.index.year==year]
        df_cor_data = []    
        columns_list = list(df0.columns)
        for i in range(0, len(columns_list)):
            col_1 = columns_list[i]
            for j in range (i+1, len(columns_list)):
                col_2 = columns_list[j]    
                df1 = pd.merge(df0[col_1], df0[col_2], left_index=True, right_index=True, how='inner', suffixes=('_left', '_right'))
                cor = df0.corr()[col_1][col_2]
                df_cor_data.append({'Item 1':col_1, 'Item 2':col_2, 'correlation':cor})
                #cor = np.corrcoef(security_1.data.loc[start:end]['Log_Returns(AdjClose)'], security_2.data.loc[start:end]['Log_Returns(AdjClose)'])[1, 0]
        df_cor = pd.DataFrame(df_cor_data)
        df_cor = df_cor.sort_values(by='correlation', ascending=False)
        df_cor = df_cor.loc[(df_cor['correlation']>0.95)]
        df_cor = df_cor.sort_values(by=['Item 1', 'Item 2'])
        dict[year] = df_cor
    return dict
 
# Execution starts here        
dir = "data/" # ie: /Users/test/Documents/Leevothyrox/data/
top_size = 15
n_gram_size = 2
limit = 10
period = 'M'        # 'Y', 'M', 'Q'
show = True
df = pd.read_csv(dir + 'dataset_doctissimo_updated.csv')       

# Cast date to datetime
df['date'] = pd.to_datetime(df['date'])

# Cast text column to string
df['text'] = df['text'].astype(str)

# Set date column as index
df = df.set_index('date')

# Words cloud after cleaning
words_cloud(df, True)

print()
print('Top word occurence 2016-2020')
print(word_occurence(df).nlargest(10, ['occurence']))
print()

print()
print('Top word occurence history')
#print(word_occurence_history(df, 'Y').nlargest(10, [2016, 2017, 2018, 2019, 2020]))
print(top_word_occurence_history(df, 'Y', 10))
print()

print()
print('Top word occurence correlations 2016-2020')
print(getCorrelations(top_word_occurence_history(df, 'Y', 10)))
print()

print()
print('Top N-gram occurrence 2016-2020')
print(n_gram(df, n_gram_size).nlargest(10, ['occurence']))
print()

print()
print('Top n_gram history')
#print(n_gram_history(df, n_gram_size, 'Y').nlargest(10, [2016, 2017, 2018, 2019, 2020]))
print(top_n_gram_history(df, n_gram_size, 'Y', 10))
print()

print()
print('Top n_gram history 2')
print(top_n_gram_history_2(df, n_gram_size, 'Y', 10))
print()

print()
print('n_gram occurence correlations 2016-2020')
print(getCorrelations(top_n_gram_history(df, n_gram_size, 'Y', 10)))
print()

print()
print('Top n_gram occurence correlations 2016-2020')
print(getTopCorrelations(top_n_gram_history(df, n_gram_size, 'Y', 10)))
print()

#df1 = top_word_occurence_history(df, 'Y', 10).transpose()
#df1 = top_word_occurence_history(df, 'M', 5).transpose()
df1 = top_n_gram_history(df, 2, 'Q', 5).transpose()
#df1 = top_n_gram_history(df, 2, 'M', 10).transpose()

df1.plot()
plt.show()

dict = getTopCorrelations_2(top_word_occurence_history(df, 'M', 5)) # Replace 5 (tests) by 20 for best results
for year, df in dict.items():
    print()
    print(year)
    print(df)