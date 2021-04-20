print('Chapter 05: Feature Engineering Similarity')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('setup.py')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import timeit
import sys, os
BASE_DIR = ".."
def universal_filename(f):
    return os.path.join(BASE_DIR, f)

_ABCNEWS_FILE = os.path.join('data', 'abcnews', 'abcnews-date-text.csv.gz')
ABCNEWS_FILE = universal_filename(_ABCNEWS_FILE)


def figNum():
    figNum.counter += 1
    return "{0:02d}".format(figNum.counter)
figNum.counter = 0
FIGPREFIX = 'ch05_fig'

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('settings.py')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# suppress warnings
import warnings;
warnings.filterwarnings('ignore');

# common imports
import pandas as pd
import numpy as np
import math
import re
import glob
import os
import sys
import json
import random
import pprint as pp
import textwrap
import sqlite3
import logging

import spacy
import nltk

from tqdm.auto import tqdm
# register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`
tqdm.pandas()

# pandas display options
# https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html#available-options
pd.options.display.max_columns = 30 # default 20
pd.options.display.max_rows = 60 # default 60
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.precision = 2
pd.options.display.max_colwidth = 200 # default 50; -1 = all
# otherwise text between $ signs will be interpreted as formula and printed in italic
pd.set_option('display.html.use_mathjax', False)

# np.set_printoptions(edgeitems=3) # default 3

import matplotlib
from matplotlib import pyplot as plt

plot_params = {'figure.figsize': (8, 4),
               'axes.labelsize': 'large',
               'axes.titlesize': 'large',
               'xtick.labelsize': 'large',
               'ytick.labelsize':'large',
               'figure.dpi': 100}
# adjust matplotlib defaults
matplotlib.rcParams.update(plot_params)

import seaborn as sns
sns.set_style("darkgrid")

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Data preparation')
sentences = ["It was the best of times",
             "it was the worst of times",
             "it was the age of wisdom",
             "it was the age of foolishness"]

tokenized_sentences = [[t for t in sentence.split()] for sentence in sentences]

vocabulary = set([w for s in tokenized_sentences for w in s])

import pandas as pd
[[w, i] for i,w in enumerate(vocabulary)]

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('One-hot by hand')
def onehot_encode(tokenized_sentence):
    return [1 if w in tokenized_sentence else 0 for w in vocabulary]

onehot = [onehot_encode(tokenized_sentence) for tokenized_sentence in tokenized_sentences]

for (sentence, oh) in zip(sentences, onehot):
    print("%s: %s" % (oh, sentence))

pd.DataFrame(onehot, columns=vocabulary)

sim = [onehot[0][i] & onehot[1][i] for i in range(0, len(vocabulary))]
sum(sim)

import numpy as np
np.dot(onehot[0], onehot[1])

np.dot(onehot, onehot[1])

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Out of vocabulary')
onehot_encode("the age of wisdom is the best of times".split())

onehot_encode("John likes to watch movies. Mary likes movies too.".split())

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('document term matrix')
onehot

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('similarities')
import numpy as np
np.dot(onehot, np.transpose(onehot))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('scikit learn one-hot vectorization')
from sklearn.preprocessing import MultiLabelBinarizer
lb = MultiLabelBinarizer()
lb.fit([vocabulary])
lb.transform(tokenized_sentences)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('CountVectorizer')
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

more_sentences = sentences + ["John likes to watch movies. Mary likes movies too.",
                              "Mary also likes to watch football games."]
pd.DataFrame(more_sentences)

cv.fit(more_sentences)

print(cv.get_feature_names())

dt = cv.transform(more_sentences)

print(dt)

pd.DataFrame(dt.toarray(), columns=cv.get_feature_names())

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(dt[0], dt[1])

print(len(more_sentences))

pd.DataFrame(cosine_similarity(dt, dt))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('TF/IDF')
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf_dt = tfidf.fit_transform(dt)

pd.DataFrame(tfidf_dt.toarray(), columns=cv.get_feature_names())

pd.DataFrame(cosine_similarity(tfidf_dt, tfidf_dt))

headlines = pd.read_csv(ABCNEWS_FILE, parse_dates=["publish_date"])
headlines.head()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
dt = tfidf.fit_transform(headlines["headline_text"])

print(dt)

dt.data.nbytes

#%%time
exe_time = timeit.timeit(cosine_similarity(dt[0:10000], dt[0:10000]), number=10000)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Stopwords')
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
print(len(stopwords))
tfidf = TfidfVectorizer(stop_words=stopwords)
dt = tfidf.fit_transform(headlines["headline_text"])
print(dt)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('min_df')
tfidf = TfidfVectorizer(stop_words=stopwords, min_df=2)
dt = tfidf.fit_transform(headlines["headline_text"])
print(dt)

tfidf = TfidfVectorizer(stop_words=stopwords, min_df=.0001)
dt = tfidf.fit_transform(headlines["headline_text"])
print(dt)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('max_df')
tfidf = TfidfVectorizer(stop_words=stopwords, max_df=0.1)
dt = tfidf.fit_transform(headlines["headline_text"])
print(dt)

tfidf = TfidfVectorizer(max_df=0.1)
dt = tfidf.fit_transform(headlines["headline_text"])
print(dt)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('n-grams')
tfidf = TfidfVectorizer(stop_words=stopwords, ngram_range=(1,2), min_df=2)
dt = tfidf.fit_transform(headlines["headline_text"])
print(dt.shape)
print(dt.data.nbytes)
tfidf = TfidfVectorizer(stop_words=stopwords, ngram_range=(1,3), min_df=2)
dt = tfidf.fit_transform(headlines["headline_text"])
print(dt.shape)
print(dt.data.nbytes)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Lemmas')
from tqdm.auto import tqdm
import spacy
nlp = spacy.load("en")
nouns_adjectives_verbs = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]
for i, row in tqdm(headlines.iterrows(), total=len(headlines)):
    doc = nlp(str(row["headline_text"]))
    headlines.at[i, "lemmas"] = " ".join([token.lemma_ for token in doc])
    headlines.at[i, "nav"] = " ".join([token.lemma_ for token in doc if token.pos_ in nouns_adjectives_verbs])

headlines.head()

tfidf = TfidfVectorizer(stop_words=stopwords)
dt = tfidf.fit_transform(headlines["lemmas"].map(str))
print(dt)

tfidf = TfidfVectorizer(stop_words=stopwords)
dt = tfidf.fit_transform(headlines["nav"].map(str))
print(dt)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('remove top 10,000')
top_10000 = pd.read_csv("https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt", header=None)
tfidf = TfidfVectorizer(stop_words=set(top_10000.iloc[:,0].values))
dt = tfidf.fit_transform(headlines["nav"].map(str))
print(dt)

tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words=set(top_10000.iloc[:,0].values), min_df=2)
dt = tfidf.fit_transform(headlines["nav"].map(str))
print(dt)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Finding document most similar to made-up document')
tfidf = TfidfVectorizer(stop_words=stopwords, min_df=2)
dt = tfidf.fit_transform(headlines["lemmas"].map(str))
print(dt)

made_up = tfidf.transform(["australia and new zealand discuss optimal apple size"])

sim = cosine_similarity(made_up, dt)

sim[0]

headlines.iloc[np.argsort(sim[0])[::-1][0:5]][["publish_date", "lemmas"]]

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Finding the most similar documents')
stopwords.add("test")
tfidf = TfidfVectorizer(stop_words=stopwords, ngram_range=(1,2), min_df=2, norm='l2')
dt = tfidf.fit_transform(headlines["headline_text"])

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Timing Cosine Similarity')
#%%time
exe_time = timeit.timeit(cosine_similarity(dt[0:10000], dt[0:10000], dense_output=False), number=10000)

#%%time
exe_time = timeit.timeit((cosine_similarity(dt[0:10000], dt[0:10000])), number=10000)
r = cosine_similarity(dt[0:10000], dt[0:10000])
r[r > 0.9999] = 0
print(np.argmax(r))

#%%time
exe_time = timeit.timeit(cosine_similarity(dt[0:10000], dt[0:10000], dense_output=False), number=10000)
r = cosine_similarity(dt[0:10000], dt[0:10000], dense_output=False)
r[r > 0.9999] = 0
print(np.argmax(r))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Timing Dot-Product')
#%%time
exe_time = timeit.timeit(np.dot(dt[0:10000], np.transpose(dt[0:10000])), number=10000)
r = np.dot(dt[0:10000], np.transpose(dt[0:10000]))
r[r > 0.9999] = 0
print(np.argmax(r))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Batch')
%%time
exe_time = timeit.timeit({} , number=10000)
batch = 10000
max_sim = 0.0
max_a = None
max_b = None
for a in range(0, dt.shape[0], batch):
    for b in range(0, a+batch, batch):
        print(a, b)
        #r = np.dot(dt[a:a+batch], np.transpose(dt[b:b+batch]))
        r = cosine_similarity(dt[a:a+batch], dt[b:b+batch], dense_output=False)
        # eliminate identical vectors
        # by setting their similarity to np.nan which gets sorted out
        r[r > 0.9999] = 0
        sim = r.max()
        if sim > max_sim:
            # argmax returns a single value which we have to
            # map to the two dimensions
            (max_a, max_b) = np.unravel_index(np.argmax(r), r.shape)
            # adjust offsets in corpus (this is a submatrix)
            max_a += a
            max_b += b
            max_sim = sim

print(max_a, max_b)

print(max_sim)

pd.set_option('max_colwidth', -1)
headlines.iloc[[max_a, max_b]][["publish_date", "headline_text"]]

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Finding most related words')
tfidf_word = TfidfVectorizer(stop_words=stopwords, min_df=1000)
dt_word = tfidf_word.fit_transform(headlines["headline_text"])

r = cosine_similarity(dt_word.T, dt_word.T)
np.fill_diagonal(r, 0)

voc = tfidf_word.get_feature_names()
size = r.shape[0] # quadratic
for index in np.argsort(r.flatten())[::-1][0:40]:
    a = int(index/size)
    b = index%size
    if a > b:  # avoid repetitions
        print('"%s" related to "%s"' % (voc[a], voc[b]))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('')
