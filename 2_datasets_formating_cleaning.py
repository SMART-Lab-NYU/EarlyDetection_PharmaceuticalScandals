# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import csv
import nltk
import ssl
import sys
import spacy
import fr_core_news_sm
from nltk.corpus import stopwords
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
stop = stopwords.words('french')
nlp_fr = fr_core_news_sm.load()

# Functions & Dictionaries
def doctissimo_sort_range(df):
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        df = df.sort_values(by=['date'], ascending=False)
        df['year'] = df['date'].dt.year
        df = df.set_index(df['date'])
        df = df[:'2015-12-31']
        df['index'] = range(0, len(df))
        df = df.set_index(df['index'])
        del df['index']
        df['text'] = df['text'].astype(str)
        return df

def doctissimo_words_improvment(df):
        df['text'] = df['text'].str.split() # Split string
        i=0
        for line in df['text']:
                new_line = []
                for word in line:
                        for imprv in words_improvment:
                                if word in imprv and word != imprv[0]:
                                        word = imprv[0]
                                        break
                        new_line.append(word)
                df.at[i, 'text'] = new_line
                i += 1
        df['text'] = df['text'].apply(' '.join) # Join string
        return df

def dataframe_preprocessing(df):
        df['text'] = df['text'].str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8') # Remove accent
        df['text'] = [re.sub(r'[^a-zA-Z0-9 ]', ' ', str(x)) for x in df['text']] # Remove special characters and punctuation
        df['text'] = df['text'].str.lower() # Convert ['text'] string to lowercase
        df['text'] = df['text'].str.replace('#034', '', regex=True) # Remove #034 pattern
        df['text'] = df['text'].str.replace('#039', '', regex=True) # Remove #039 pattern
        df['text'] = df['text'].str.replace(".*gif", "", regex=True) # Remove all gifs
        df['text'] = df['text'].str.replace("http.* ", "", regex=True) # Remove http links
        df['text'] = df['text'].str.replace("https.* ", "", regex=True) # Remove https links
        df['text'] = [re.sub(r'[A-Za-z]+\d+|\d+[A-Za-z]+','', str(x)) for x in df['text']] # Delete numbers between alphabetic chars
        df['text'] = [re.sub(r'\b(?!(\D\S*|[12][0-9]{3})\b)\S+\b','', str(x)) for x in df['text']] # Numbers except dates
        df['text'] = df['text'].str.replace('\n',' ', regex=True).replace('\t',' ', regex=True) # Remove line breaks and tabulations
        df['text'] = [re.sub(r'(^| ).( |$)', ' ', str(x)) for x in df['text']] # Remove single characters
        df['text'] = [re.sub(r'\s+', ' ', str(x)) for x in df['text']] # Delete multiple spaces
        return df

def dataframe_stopwords_wtd(df):
        # Words to delete (csv file)
        words_to_delete = []
        words_count = 0
        with open(dir + 'exclusions.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in reader:
                words_to_delete.append('; '.join(row)) 
        words_count = len(words_to_delete)
        words = df['text'].str.split()
        words = words.apply(lambda x: [item for item in x if item not in words_to_delete]) # Words in "text" of dataframe) without excluded words
        words = words.apply(lambda x: [item for item in x if item not in stop]) # From library "FR"
        words = words.apply(lambda x: [item for item in x if item not in additional_stopwords]) # From dictionary
        df['text'] = words.apply(' '.join)
        return df

def dataframe_lemmatization(df):
        # Lemmatization with nlp_fr
        df['text'] = df['text'].apply(lambda x: [y.lemma_ for y in nlp_fr(x)]).apply(' '.join) 
        return df

def dataframe_duplicata_less3words(df):
        words_count = df['text'].str.count(' ') + 1 # 'text' characters counter
        df['words_count'] = words_count # Add words_count column on the dataframe
        before_deleting = df['text'].count()
        print('\n******************\nNumber of rows BEFORE deleting the rows which contains less than 3 words : ' + str(before_deleting) + ' rows')
        df.drop(df[df['words_count'] < 3].index, inplace = True) # Remove rows which contains less than 3 words
        after_deleting = df['text'].count()
        print('Number of rows AFTER deleting the rows which contains less than 3 words : ' + str(after_deleting) + ' rows')
        diff_deleting = before_deleting - after_deleting
        print('Difference : ' + str(diff_deleting) + ' rows')
        df.drop_duplicates() # Remove duplicates rows
        df.dropna() # Drop the rows even with single NaN or single missing values.
        after_del_duplicates = df['text'].count()
        print('Number of rows after deleting duplicata : ' + str(after_del_duplicates) + ' rows')
        diff_duplicates = after_deleting - after_del_duplicates
        total_delete = diff_deleting + diff_duplicates
        print('Difference : ' + str(diff_duplicates) + ' rows')
        print('Total number of deleted rows : ' + str(total_delete) + '\n******************')
        return df

def lemmatizer(text):        
    sent = []
    doc = nlp_fr(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

# StopWords dictionary
additional_stopwords = ['a', 'abord', 'afin', 'ah', 'ai', 'ainsi', 'allaient', 'allo', 'allô', 'allons', 'alors', 'apres', 'après', 'assez', 'attendu', 'aucun', 'aucune', 'aucuns', 'aujourd', 'aujourdhui', 'auquel', 'auquelle', 'auquelles', 'auquels', 'aussi', 'autre', 'autres', 'auxquelles', 'auxquels', 'avant', 'avoir',
                        'b', 'bonjour', 'bonsoir', 'bah', 'beaucoup', 'bien', 'bigre', 'bon', 'boum', 'br', 'bravo', 'brr', 'brrr',
                        'ca', 'ça', 'car', 'ceci', 'cela', 'celle', 'celle-ci', 'celle-la', 'celle-là', 'celles', 'celles-ci', 'celles-la', 'celles-là', 'celui', 'celui-ci', 'celui-la', 'celui-là', 'cent', 'cependant', 'certain', 'certaine', 'certaines', 'certains', 'certes', 'cet', 'cette', 'ceux', 'ceux', 'ceux-ci', 'ceux-là', 'ceux-là', 'chacun', 'chaque', 'cher', 'chere', 'chère', 'cheres', 'chères', 'chers', 'chez', 'chiche', 'chut', 'ci', 'cinq', 'cinquantaine', 'cinquante', 'cinquantieme', 'cinquantième', 'cinquieme', 'cinquième', 'clac', 'clic', 'combien', 'comme', 'comment', 'compris', 'concernant', 'contre', 'couic', 'crac',
                        'da', 'debout', 'debut', 'début', 'dedans', 'dehors', 'dela', 'delà', 'depuis', 'derriere', 'derrière', 'dés', 'dès', 'desormais', 'désormais', 'desquelles', 'desquels','dessous', 'dessus', 'deux', 'deuxieme', 'deuxième', 'deuxiemement', 'deuxièmement', 'devant', 'devers', 'devra', 'devrait', 'different', 'différent', 'differente', 'différente', 'differentes', 'différentes', 'differents', 'différents', 'dire', 'divers', 'diverse', 'diverses', 'dix', 'dix-huit', 'dix-neuf', 'dix-sept', 'dixieme', 'dixième', 'doit', 'doivent', 'donc', 'dont', 'douze', 'douzieme', 'douzième', 'dring', 'droite', 'duquel', 'durant',
                        'e', 'effet', 'eh', 'elle-meme', 'elle-même', 'elles', 'elles-memes', 'elles-mêmes', 'encore', 'entre', 'envers', 'environ', 'ès', 'essai', 'etaient', 'etais', 'etait', 'etant', 'etante', 'etantes', 'etants', 'etat', 'état', 'etats', 'états', 'etc', 'ete', 'etee', 'etees', 'etes', 'etiez', 'etions', 'étions', 'etre', 'être', 'euh', 'eumes', 'eux-memes', 'eux-mêmes', 'excepte', 'excepté',
                        'f', 'facon', 'façon', 'fais', 'faisaient', 'faisant', 'fait', 'faites', 'feront', 'fi', 'flac', 'floc', 'fois', 'font', 'force', 'fumes', 'futes',
                        'g', 'gens',
                        'h', 'ha', 'haut', 'he', 'hé', 'hein', 'helas', 'hélas', 'hem', 'hep', 'hi', 'ho', 'hola', 'holà', 'hop' ,'hormis', 'hors' ,'hou', 'houp', 'hue', 'hui', 'huit', 'huitieme', 'huitième', 'hum', 'hurrah',
                        'i', 'ici', 'importe',
                        'jusqu', 'jusqua', 'jusque', 'juste',
                        'k',
                        'là', 'laquelle', 'las', 'lequel', 'lès', 'lesquelles', 'lesquels', 'leurs', 'longtemps', 'lorsque', 'lui-meme', 'lui-même',
                        'maint', 'maintenant', 'malgre', 'malgré', 'meme', 'memes', 'mêmes' ,'merci', 'mien', 'mienne', 'miennes', 'miens', 'mille', 'mince', 'mine', 'moi-meme', 'moi-même', 'moins', 'mot', 'moyennant',
                        'na', 'nai', 'nas', 'neanmoins', 'néanmoins', 'neuf', 'neuvieme', 'neuvième', 'ni', 'nombreuses', 'nombreux', 'nommes', 'nommés', 'non', 'nôtre', 'notres', 'nôtres', 'nous-meme', 'nous-memes', 'nous-memes', 'nous-mêmes', 'nouveau', 'nouveaux', 'nul',
                        'o', 'onsoir', 'onjour', 'ô', 'oh', 'ohe', 'ohé', 'ole', 'olé', 'olle', 'ollé', 'onze', 'onzieme', 'onzième', 'ore', 'où', 'ouf', 'ouias', 'oust', 'ouste', 'outre',
                        'p', 'paf', 'pan', 'parce', 'parmi', 'parmis', 'parole', 'partant', 'particulier', 'particuliere', 'particulière', 'particulierement', 'particulièrement', 'passe', 'passé', 'pendant', 'personne', 'personnes', 'peu', 'peut', 'peuvent', 'peux', 'pff', 'pfff', 'pffff', 'pfft', 'pfut', 'piece', 'pièce', 'pif', 'plein', 'pleins', 'plouf', 'plupart', 'plus', 'plusieurs', 'plutot', 'plutôt', 'pouah', 'pourquoi', 'premier', 'premiere', 'première', 'premierement', 'premièrement', 'pres','près', 'proche', 'psitt', 'puisque',
                        'q', 'quand', 'quant', 'quant-a-soi', 'quant-à-soi', 'quant-a-soit', 'quanta', 'quarante', 'quatorze', 'quatre', 'quatre-vingt', 'quatrieme', 'quatrième', 'quatriemement', 'quatrièmement', 'quel', 'quelconque', 'quell', 'quelle', 'quelle', 'quelles', 'quelles', 'quelque', 'quelques', 'quelquun', 'quels', 'quest', 'quiconque', 'quil', 'quils', 'quinze', 'quoi', 'quoique',
                        'r', 'revoici', 'revoila', 'revoilà', 'rien',
                        'sacrebleu', 'sans', 'sapristi', 'sauf', 'seize', 'selon', 'sept', 'septieme', 'seulement', 'si', 'sien', 'sienne', 'siennes', 'siens', 'sinon', 'six', 'sixieme', 'sixième', 'soi', 'soi-meme', 'soi-même', 'soient', 'sois', 'soixante', 'sous', 'suivant', 'sujet', 'surtout',
                        'tac', 'tandis', 'tant', 'té', 'tel', 'telle', 'tellement', 'telles', 'tels', 'tenant', 'tic', 'tien', 'tienne', 'tiennes', 'tiens', 'toc', 'toi-meme', 'toi-même', 'touchant', 'toujours', 'tous', 'tout', 'toute', 'toutes', 'treize', 'trente', 'tres', 'très', 'trois', 'troisieme', 'troisième', 'troisiemement', 'troisièmement', 'trop', 'tsoin', 'tsouin',
                        'u', 'unes', 'uns',
                        'v', 'va', 'vais', 'valeur', 'valeurs', 'vas', 've', 'vé', 'vers', 'via', 'vif', 'vifs', 'vingt', 'vivat', 'vive', 'vives', 'vlan', 'voici', 'voie', 'voient', 'voila', 'voilà', 'vont', 'vôtre', 'votres', 'vôtres', 'vous-memes', 'vous-mêmes', 'vu', 
                        'w',
                        'x',
                        'z', 'zut']

# Words_improvment dictionary
words_improvment = [['levothyrox', 'levo', 'levothyro', 'levotyrox'],
                    ['euthyrox', 'leuthyrox', 'eutyrox', 'leutyrox'],
                    ['lthyroxine', 'lthyroxin', 'ltyroxine', 'ltyroxin'],
                    ['hypothyroidie', 'lhypothyroidie', 'hypotyroidie', 'lhypotyroidie'],
                    ['comprime', 'comprim'],
                    ['cytomel', 'cynomel'],
                    ['controle', 'control'],
                    ['changer', 'change', 'chang', 'changement'],
                    ['allemagne', 'allemand'],
                    ['generaliste', 'generalist'],
                    ['arret', 'arreter', 'arrete'],
                    ['excipient', 'excipients'],
                    ['laboratoire', 'laboratoir'],
                    ['poids', 'poid'],
                    ['hormone', 'hormon', 'dhormone', 'dhormon', 'lhormone', 'lhormon'],
                    ['correcte', 'correct'],
                    ['courche', 'coucher', 'chouchee'],
                    ['neomercazole', 'neomercazol'],
                    ['enceinte', 'enceint'],
                    ['ancien', 'ancienne', 'lancien', 'lancienne'],
                    ['francais', 'fraincaise', 'francai'],
                    ['manque', 'manqu'],
                    ['medicament', 'medicaments', 'medoc', 'medocs'],
                    ['stress', 'stres'],
                    ['analyse', 'danalyse'],
                    ['hasimoto', 'dhashimoto', 'hasimoto'],
                    ['thyroidite', 'thyroidit', 'tyroidite', 'tyroidit'],
                    ['angoisse', 'dangoisse'],
                    ['hypo', 'lhypo', 'dhypo'],
                    ['hyper', 'lhyper', 'dhyper'],
                    ['euthyral', 'leuthyral', 'deuthyral', 'eutyral', 'leutyral', 'deutyral'],
                    ['autre', 'lautre'],
                    ['pmol', 'pmoil'],
                    ['soucis', 'souci'],
                    ['augmente', 'augment', 'augmenter', 'daugmenter', 'daugmente', 'daugment'],
                    ['sentais', 'sentai'],
                    ['pensezvous', 'pensezvou'],
                    ['echographie','echo', 'lecho', 'lechographie'],
                    ['aller', 'alle', 'allee'],
                    ['adenomegalie', 'dadenomegalie'],
                    ['jaurai', 'jaurais'],
                    ['elever', 'eleve', 'elevee'],
                    ['cheveux', 'cheveu'],
                    ['devrai', 'devrais'],
                    ['menopause', 'menopaus'],
                    ['nodule', 'nodul'],
                    ['reactive', 'reactiv'],
                    ['periode', 'period'],
                    ['epuiser', 'epuisee', 'epuise'],
                    ['arriver', 'arrive', 'narrive', 'narriv'],
                    ['memoire', 'memoir'],
                    ['parcours', 'parcour'],
                    ['message', 'messages'],
                    ['specialiste', 'specialist'],
                    ['apriori', 'priori'],
                    ['delais', 'delai'],
                    ['gonfler', 'gonfle', 'gonflee'],
                    ['ovaire', 'lorvair', 'lovair'],
                    ['precis', 'preci'],
                    ['prend', 'prends'],
                    ['fatiguer', 'fatigue', 'fatiguee'],
                    ['deprimer', 'deprime', 'deprimee', 'deprim'],
                    ['penser', 'pense', 'pensee', 'pensez'],
                    ['secretaire', 'secretair'],
                    ['quelque', 'quelques', 'quelqu'],
                    ['cause', 'caus'],
                    ['lobe', 'lob'],
                    ['t3', 't3l', 'ft3'],
                    ['t4', 't4l', 'ft4'],
                    ['norme', 'norm'],
                    ['doser', 'dosee', 'dose'],
                    ['endocrinologue', 'lendocrinologue', 'endocrino', 'lendocrino'],
                    ['continu', 'continue'],
                    ['interval', 'intervalle'],
                    ['thyroidien', 'tyroidien'],
                    ['soir', 'soiree'],
                    ['conseil', 'conseille'],
                    ['anticorps', 'anticorp'],
                    ['ablation', 'lablation'],
                    ['aider', 'aide', 'maide', 'maider'],
                    ['ordre', 'ordr', 'lordre', 'lordr'],
                    ['operation', 'loperation'],
                    ['remercier', 'remercie'],
                    ['marseille', 'marseill']]

# Execution start here
dir = "data/" # ie: /Users/test/Documents/Leevothyrox/data/
df_doctissimo = pd.read_csv(dir + 'dataset_doctissimo_22_03_2020.csv', encoding='utf8')
print('\n*****************\nFile <dataset_doctissimo_22_03_2020.csv> has been loaded') 
df_french_tweets = pd.read_csv(dir + 'french_tweets.csv', encoding='utf8')
print('File <french_tweets.csv> has been loaded\n*****************\n') 

# Doctissimo : start
start = time.time()
df = df_doctissimo.copy()
del df_doctissimo
# Doctissimo : sort by date and select range : 2016-2020
df = doctissimo_sort_range(df)
print('*****************\nSorting and selecting range : 2016-2020... check\n*****************')
# Doctissimo dataframe preprocessing
df = dataframe_preprocessing(df)
print('\n*****************\nData cleaning and formating... check\n*****************\n')
print(df.iloc[3,2])
# Doctissimo : words improvment
df = doctissimo_words_improvment(df)
print('\n*****************\nWords improvment... check\n*****************\n')
print(df.iloc[3,2])
# Doctissimo stop words removing
df = dataframe_stopwords_wtd(df)
print('\n*****************\nRemoving stop words and WTD... check\n*****************\n')
print(df.iloc[3,2])
# Doctissimo lemmatization (COMMENT TO ENHANCE TIME OF EXECUTION)
df = dataframe_lemmatization(df)
print('\n*****************\nLemmatization... check\n*****************\n')
print(df.iloc[3,2])
# Doctissimo final cleaning step
df = dataframe_duplicata_less3words(df)
print('\n*****************\nRemoving duplicates and rows which contains less than 3 words... check\n*****************\n')
df.to_csv(dir + 'dataset_doctissimo_updated.csv', index=False, sep=',', header=True, encoding='utf8')
print('\nFile <dataset_doctissimo_updated.csv> has been exported')
end = time.time()
print('Elapsed time - Doctissimo:', end - start, 's')
print()
print(df)

# French_tweets : start
start = time.time()
df = df_french_tweets.astype(str).copy()
del df_french_tweets
# French_tweets dataframe preprocessing
df = dataframe_preprocessing(df)
# French_tweets : labeling format
df['sentiment'] = df['label']
df['sentiment'] = df['sentiment'].str.replace("0", "negative", regex=True)
df['sentiment'] = df['sentiment'].str.replace("1", "positive", regex=True)
col = ['sentiment', 'text']
df = df[col]
df['sentiment']=['__label__'+ s for s in df['sentiment']]
# French_tweets stop words removing
df = dataframe_stopwords_wtd(df)
# French_tweets lemmatization (COMMENT TO ENHANCE TIME OF EXECUTION)
df = dataframe_lemmatization(df)
print('*****************\nAll formating steps... check\n*****************')
# French_tweets final cleaning step
df = dataframe_duplicata_less3words(df)
del df['words_count'] # Remove "words_count" column
df.to_csv(dir + 'french_tweets_updated.csv', index=False, sep=',', header=True, encoding='utf8')
print('\nFile <french_tweets_updated.csv> has been exported')
end = time.time()
print('Elapsed time - French_tweets: ', end - start, 's')
print()
print(df)