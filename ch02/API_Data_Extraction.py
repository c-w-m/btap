print('Chapter 02: API Data Extraction')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('run local - not configured for colab')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('setup.py')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BASE_DIR = ".."


def figNum():
    figNum.counter += 1
    return "{0:02d}".format(figNum.counter)


figNum.counter = 0
FIGPREFIX = 'ch02_fig'

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('settings.py')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# suppress warnings
import warnings

warnings.filterwarnings('ignore')

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
pd.options.display.max_columns = 30  # default 20
pd.options.display.max_rows = 60  # default 60
pd.options.display.float_format = '{:.2f}'.format
# pd.options.display.precision = 2
pd.options.display.max_colwidth = 200  # default 50; -1 = all
# otherwise text between $ signs will be interpreted as formula and printed in italic
pd.set_option('display.html.use_mathjax', False)

# np.set_printoptions(edgeitems=3) # default 3

import matplotlib
from matplotlib import pyplot as plt

plot_params = {'figure.figsize': (8, 6),
               'axes.labelsize': 'small',
               'axes.titlesize': 'small',
               'xtick.labelsize': 'small',
               'ytick.labelsize':'small',
               'figure.dpi': 100}
# adjust matplotlib defaults
matplotlib.rcParams.update(plot_params)

import seaborn as sns

sns.set_style("darkgrid")

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Load Python Settings')

# to print output of all statements and not just the last
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

# otherwise text between $ signs will be interpreted as formula and printed in italic
pd.set_option('display.html.use_mathjax', False)

# path to import blueprints packages
sys.path.append(BASE_DIR + '/packages')

# adjust matplotlib resolution for book version
matplotlib.rcParams.update({'figure.dpi': 200})

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('How to use APIs to extract and derive insights from text data')
print('Application Programming Interface')
print('Blueprint - Extracting data from an API using the requests module')

import requests

response = requests.get('https://api.github.com/repositories',
                        headers={'Accept': 'application/vnd.github.v3+json'})
print(response.status_code)

print('encoding: {}'.format(response.encoding))
print('Content-Type: {}'.format(response.headers['Content-Type']))
print('server: {}'.format(response.headers['server']))

print(response.headers)

import json

print(json.dumps(response.json()[0], indent=2)[:200])

response = requests.get('https://api.github.com/search/repositories')
print(response.status_code)

response = requests.get('https://api.github.com/search/repositories',
                        params={'q': 'data_science+language:python'},
                        headers={'Accept': 'application/vnd.github.v3.text-match+json'})
print(response.status_code)

from IPython.display import Markdown, display  ###


def printmd(string):  ###
    display(Markdown(string))  ###


for item in response.json()['items'][:5]:
    print(item['name'] + ': repository ' +
          item['text_matches'][0]['property'] + ' - \"' +
          item['text_matches'][0]['fragment'] + '\" -- matched with ' +
          item['text_matches'][0]['matches'][0]['text'])

response = requests.get(
    'https://api.github.com/repos/pytorch/pytorch/issues/comments')
print('Response Code', response.status_code)
print('Number of comments', len(response.json()))

print(response.links)


def get_all_pages(url, params=None, headers=None):
    output_json = []
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        output_json = response.json()
        if 'next' in response.links:
            next_url = response.links['next']['url']
            if next_url is not None:
                output_json += get_all_pages(next_url, params, headers)
    return output_json


out = get_all_pages(
    "https://api.github.com/repos/pytorch/pytorch/issues/comments",
    params={
        'since': '2020-07-01T10:00:01Z',
        'sorted': 'created',
        'direction': 'desc'
    },
    headers={'Accept': 'application/vnd.github.v3+json'})
df = pd.DataFrame(out)

pd.set_option('display.max_colwidth', -1)
if 'body' in df.index:
    print(df['body'].count())
    print(df[['id', 'created_at', 'body']].sample(1, random_state=42))

response = requests.head(
    'https://api.github.com/repos/pytorch/pytorch/issues/comments')
print('X-Ratelimit-Limit', response.headers['X-Ratelimit-Limit'])
print('X-Ratelimit-Remaining', response.headers['X-Ratelimit-Remaining'])

# Converting UTC time to human-readable format
import datetime

print(
    'Rate Limits reset at',
    datetime.datetime.fromtimestamp(int(
        response.headers['X-RateLimit-Reset'])).strftime('%c'))

from datetime import datetime
import time


def handle_rate_limits(response):
    now = datetime.now()
    reset_time = datetime.fromtimestamp(
        int(response.headers['X-RateLimit-Reset']))
    remaining_requests = response.headers['X-Ratelimit-Remaining']
    remaining_time = (reset_time - now).total_seconds()
    intervals = remaining_time / (1.0 + int(remaining_requests))
    print('Sleeping for', intervals)
    time.sleep(intervals)
    return True


from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

retry_strategy = Retry(
    total=5,
    status_forcelist=[500, 503, 504],
    backoff_factor=1
)

retry_adapter = HTTPAdapter(max_retries=retry_strategy)

http = requests.Session()
http.mount("https://", retry_adapter)
http.mount("http://", retry_adapter)

response = http.get('https://api.github.com/search/repositories',
                    params={'q': 'data_science+language:python'})

for item in response.json()['items'][:5]:
    print(item['name'])

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

retry_strategy = Retry(
    total=5,
    status_forcelist=[500, 503, 504],
    backoff_factor=1
)

retry_adapter = HTTPAdapter(max_retries=retry_strategy)

http = requests.Session()
http.mount("https://", retry_adapter)
http.mount("http://", retry_adapter)


def get_all_pages(url, param=None, header=None):
    output_json = []
    response = http.get(url, params=param, headers=header)
    if response.status_code == 200:
        output_json = response.json()
        if 'next' in response.links:
            next_url = response.links['next']['url']
            if (next_url is not None) and (handle_rate_limits(response)):
                output_json += get_all_pages(next_url, param, header)
    return output_json


out = get_all_pages("https://api.github.com/repos/pytorch/pytorch/issues/comments",
                    param={'since': '2020-04-01T00:00:01Z'})
df = pd.DataFrame(out)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint - Extracting Twitter data with Tweepy')
import tweepy

app_api_key = 'YOUR_APP_KEY_HERE'
app_api_secret_key = 'YOUR_APP_SECRET_HERE'

app_api_key = 'CWIBFKPrcOU4GsdRr6J5fpaps'
app_api_secret_key = 'SghP0LINUECDj0PzIi1vmDfRtNopqJNfb5xd3fH7XpO9ZaEtme'

auth = tweepy.AppAuthHandler(app_api_key, app_api_secret_key)
api = tweepy.API(auth)

print('API Host: {}'.format(api.host))
print('API Version: {}'.format(api.api_root))

pd.set_option('display.max_colwidth', None)
search_term = 'cryptocurrency'

tweets = tweepy.Cursor(api.search,
                       q=search_term,
                       lang="en").items(100)

retrieved_tweets = [tweet._json for tweet in tweets]
df = pd.json_normalize(retrieved_tweets)

print(df[['text']].sample(3))

# Note: the following code might return 'Rate limit reached. Sleeping for: 750
api = tweepy.API(auth,
                 wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True,
                 retry_count=5,
                 retry_delay=10)

search_term = 'cryptocurrency OR crypto -filter:retweets'

tweets = tweepy.Cursor(api.search,
                       q=search_term,
                       lang="en",
                       tweet_mode='extended',
                       count=30).items(12000)

retrieved_tweets = [tweet._json for tweet in tweets]

df = pd.json_normalize(retrieved_tweets)
print('Number of retrieved tweets {}'.format(len(df)))

print(df[['created_at', 'full_text', 'entities.hashtags']].sample(2))


def extract_entities(entity_list):
    entities = set()
    if len(entity_list) != 0:
        for item in entity_list:
            for key, value in item.items():
                if key == 'text':
                    entities.add(value.lower())
    return list(entities)


df['Entities'] = df['entities.hashtags'].apply(extract_entities)
pd.Series(np.concatenate(df['Entities'])).value_counts()[:25].plot(kind='barh')
plt.tight_layout()
plt.savefig('{}{}_top25_entities_hbar.png'.format(FIGPREFIX, figNum()))

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

tweets = tweepy.Cursor(api.user_timeline,
                       screen_name='MercedesAMGF1',
                       lang="en",
                       tweet_mode='extended',
                       count=100).items(5000)

retrieved_tweets = [tweet._json for tweet in tweets]
df = pd.io.json.json_normalize(retrieved_tweets)
print('Number of retrieved tweets {}'.format(len(df)))


def get_user_timeline(screen_name):
    api = tweepy.API(auth,
                     wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    tweets = tweepy.Cursor(api.user_timeline,
                           screen_name=screen_name,
                           lang="en",
                           tweet_mode='extended',
                           count=200).items()
    retrieved_tweets = [tweet._json for tweet in tweets]
    df = pd.io.json.json_normalize(retrieved_tweets)
    df = df[~df['retweeted_status.id'].isna()]
    return df


df_mercedes = get_user_timeline('MercedesAMGF1')
print('Number of Tweets from Mercedes {}'.format(len(df_mercedes)))
df_ferrari = get_user_timeline('ScuderiaFerrari')
print('Number of Tweets from Ferrari {}'.format(len(df_ferrari)))

import regex as re
import nltk
from collections import Counter
from wordcloud import WordCloud

stopwords = set(nltk.corpus.stopwords.words('english'))
RE_LETTER = re.compile(r'\b\p{L}{2,}\b')


def tokenize(text):
    return RE_LETTER.findall(text)


def remove_stop(tokens):
    return [t for t in tokens if t.lower() not in stopwords]


pipeline = [str.lower, tokenize, remove_stop]


def prepare(text):
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens


def count_words(df, column='tokens', preprocess=None, min_freq=2):
    # process tokens and update counter
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)

    # create counter and run through all data
    counter = Counter()
    df[column].map(update)

    # transform counter into data frame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'

    return freq_df.sort_values('freq', ascending=False)


def wordcloud(word_freq, title=None, max_words=200, stopwords=None):
    wc = WordCloud(width=800, height=400,
                   background_color="black", colormap="Paired",
                   max_font_size=150, max_words=max_words)

    # convert data frame into dict
    if type(word_freq) == pd.Series:
        counter = Counter(word_freq.fillna(0).to_dict())
    else:
        counter = word_freq

    # filter stop words in frequency counter
    if stopwords is not None:
        counter = {token: freq for (token, freq) in counter.items()
                   if token not in stopwords}
    wc.generate_from_frequencies(counter)

    plt.title(title)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")


def wordcloud_blueprint(df, colName, max_words, num_stopwords):
    # Step 1: Convert input text column into tokens
    df['tokens'] = df[colName].map(prepare)

    # Step 2: Determine the frequency of each of the tokens
    freq_df = count_words(df)

    # Step 3: Generate the wordcloud using the frequencies controlling for stopwords
    wordcloud(freq_df['freq'], max_words, stopwords=freq_df.head(num_stopwords).index)


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)  ###
wordcloud_blueprint(df_mercedes, 'full_text',
                    max_words=100,
                    num_stopwords=5)

plt.subplot(1, 2, 2)  ###
wordcloud_blueprint(df_ferrari, 'full_text',
                    max_words=100,
                    num_stopwords=5)
plt.tight_layout()
plt.savefig('{}{}_mercedes_ferrari_maxwd100_numstopwd_wc.png'.format(FIGPREFIX, figNum()))

from datetime import datetime
import math


class FileStreamListener(tweepy.StreamListener):

    def __init__(self, max_tweets=math.inf):
        self.num_tweets = 0
        self.TWEETS_FILE_SIZE = 10
        self.num_files = 0
        self.tweets = []
        self.max_tweets = max_tweets

    def on_data(self, data):
        while self.num_files * self.TWEETS_FILE_SIZE < self.max_tweets:
            self.tweets.append(json.loads(data))
            self.num_tweets += 1
            if self.num_tweets < self.TWEETS_FILE_SIZE:
                return True
            else:
                filename = 'Tweets_' + str(datetime.now().time()) + '.txt'
                print(self.TWEETS_FILE_SIZE, 'Tweets saved to', filename)
                file = open(filename, "w")
                json.dump(self.tweets, file)
                file.close()
                self.num_files += 1
                self.tweets = []
                self.num_tweets = 0
                return True
        return False

    def on_error(self, status_code):
        if status_code == 420:
            print('Too many requests were made, please stagger requests')
            return False
        else:
            print('Error {}'.format(status_code))
            return False


user_access_token = 'YOUR_USER_ACCESS_TOKEN_HERE'
user_access_secret = 'YOUR_USER_ACCESS_SECRET_HERE'

auth = tweepy.OAuthHandler(app_api_key, app_api_secret_key)
auth.set_access_token(user_access_token, user_access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

fileStreamListener = FileStreamListener(20)
fileStream = tweepy.Stream(auth=api.auth,
                           listener=fileStreamListener,
                           tweet_mode='extended')
fileStream.filter(track=['cryptocurrency'])

df = pd.json_normalize(json.load(open('Tweets_01:01:36.656960.txt')))
print(df.head(2))

import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

p_wiki = wiki_wiki.page('Cryptocurrency')
print(p_wiki.text[:200], '....')

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
