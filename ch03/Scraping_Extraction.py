print('Chapter 03: Scraping Extraction')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('setup.py')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BASE_DIR = ".."
def figNum():
    figNum.counter += 1
    return "{0:02d}".format(figNum.counter)
figNum.counter = 0
FIGPREFIX = 'ch03_fig'

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
print('Blueprint: download and interpret robots.txt')
import urllib.robotparser
rp = urllib.robotparser.RobotFileParser()
rp.set_url("https://www.reuters.com/robots.txt")
rp.read()

rp.can_fetch("*", "https://www.reuters.com/sitemap.xml")

rp.can_fetch("*", "https://www.reuters.com/finance/stocks/option")

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: finding URLs from sitemap.xml')
# might need to install xmltodict
import xmltodict
import requests

sitemap = xmltodict.parse(requests.get('https://www.reuters.com/sitemap_news_index1.xml').text)

# just see some of the URLs
urls = [url["loc"] for url in sitemap["urlset"]["url"]]
print("\n".join(urls[0:3]))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: finding URLs from RSS')
# might need to install feedparser
import feedparser
feed = feedparser.parse('http://web.archive.org/web/20200613003232if_/http://feeds.reuters.com/Reuters/worldNews')

print([(e.title, e.link) for e in feed.entries])

print([e.id for e in feed.entries])

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Example: Downloading HTML pages with Python')
% % time
s = requests.Session()
for url in urls[0:10]:
    # get the part after the last / in URL and use as filename
    file = url.split("/")[-1]

    r = s.get(url)
    with open(file, "w+b") as f:
        f.write(r.text.encode('utf-8'))

with open("urls.txt", "w+b") as f:
    f.write("\n".join(urls).encode('utf-8'))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: Extraction with regular expressions')
url = 'https://www.reuters.com/article/us-health-vaping-marijuana-idUSKBN1WG4KT'

file = url.split("/")[-1] + ".html"

r = requests.get(url)

with open(file, "w+") as f:
    f.write(r.text)

import re
with open(file, "r") as f:
    html = f.read()
    g = re.search(r'<title>(.*)</title>', html, re.MULTILINE|re.DOTALL)
    if g:
        print(g.groups()[0])

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Using an HTML parser for extraction')
WA_PREFIX = "http://web.archive.org/web/20200118131624/"
html = s.get(WA_PREFIX + url).text

from bs4 import BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')
print(soup.select("h1.ArticleHeader_headline"))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: extracting the title/headline')
print(soup.h1)

print(soup.h1.text)

print(soup.title.text)

print(soup.title.text.strip())

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('')


print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: extracting the article text')
print(soup.select_one("div.StandardArticleBody_body").text)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: extracting image captions')
print(soup.select("div.StandardArticleBody_body figure"))

print(soup.select("div.StandardArticleBody_body figure img"))

print(soup.select("div.StandardArticleBody_body figcaption"))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: extracting the URL')
print(soup.find("link", {'rel': 'canonical'})['href'])

print(soup.select_one("link[rel=canonical]")['href'])

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: extracting list information (authors)')
print(soup.find("meta", {'name': 'Author'})['content'])

sel = "div.BylineBar_first-container.ArticleHeader_byline-bar div.BylineBar_byline span"
print(soup.select(sel))

print([a.text for a in soup.select(sel)])

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: Extracting text of links (section)')
print(soup.select_one("div.ArticleHeader_channel a").text)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: Extracting reading time')
print(soup.select_one("p.BylineBar_reading-time").text)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: extracting attributes (id)')
print(soup.select_one("div.StandardArticle_inner-container")['id'])

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: Extracting Attribution')
print(soup.select_one("p.Attribution_content").text)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: Extracting Timestamp')
ptime = soup.find("meta", { 'property': "og:article:published_time"})['content']
print(ptime)

from dateutil import parser
print(parser.parse(ptime))

print(parser.parse(soup.find("meta", { 'property': "og:article:modified_time"})['content']))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: Spidering')
import requests
from bs4 import BeautifulSoup
import os.path
from dateutil import parser


def download_archive_page(page):
    filename = "page-%06d.html" % page
    if not os.path.isfile(filename):
        url = "https://www.reuters.com/news/archive/" + \
              "?view=page&page=%d&pageSize=10" % page
        r = requests.get(url)
        with open(filename, "w+") as f:
            f.write(r.text)


def parse_archive_page(page_file):
    with open(page_file, "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    hrefs = ["https://www.reuters.com" + a['href']
             for a in soup.select("article.story div.story-content a")]
    return hrefs


def download_article(url):
    # check if article already there
    filename = url.split("/")[-1] + ".html"
    if not os.path.isfile(filename):
        r = requests.get(url)
        with open(filename, "w+") as f:
            f.write(r.text)


def parse_article(article_file):
    def find_obfuscated_class(soup, klass):
        return soup.find_all(lambda tag: tag.has_attr("class") and (klass in " ".join(tag["class"])))

    with open(article_file, "r") as f:
        html = f.read()
    r = {}
    soup = BeautifulSoup(html, 'html.parser')
    r['url'] = soup.find("link", {'rel': 'canonical'})['href']
    r['id'] = r['url'].split("-")[-1]
    r['headline'] = soup.h1.text
    r['section'] = find_obfuscated_class(soup, "ArticleHeader-channel")[0].text

    r['text'] = "\n".join([t.text for t in find_obfuscated_class(soup, "Paragraph-paragraph")])
    r['authors'] = find_obfuscated_class(soup, "Attribution-attribution")[0].text
    r['time'] = soup.find("meta", {'property': "og:article:published_time"})['content']
    return r

# download 10 pages of archive
for p in range(1, 10):
    download_archive_page(p)

# parse archive and add to article_urls
import glob

article_urls = []
for page_file in glob.glob("page-*.html"):
    article_urls += parse_archive_page(page_file)

# download articles
for url in article_urls:
    download_article(url)

# arrange in pandas DataFrame
import pandas as pd

df = pd.DataFrame()
for article_file in glob.glob("*-id???????????.html"):
    df = df.append(parse_article(article_file), ignore_index=True)
df['time'] = pd.to_datetime(df.time)

print(df)

print(df.sort_values("time"))

df[df["time"]>='2020-01-01'].set_index("time").resample('D').agg({'id': 'count'}).plot.bar()
plt.savefig('{}{}_2020-01-01_bar.png'.format(FIGPREFIX, figNum()))

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint Density extraction')
from readability import Document

doc = Document(html)
print(doc.title())

print(doc.short_title())

print(doc.summary())

doc.url

density_soup = BeautifulSoup(doc.summary(), 'html.parser')
print(density_soup.body.text)

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('Blueprint: Scrapy')
# might need to install scrapy
import scrapy
import logging


class ReutersArchiveSpider(scrapy.Spider):
    name = 'reuters-archive'

    custom_settings = {
        'LOG_LEVEL': logging.WARNING,
        'FEED_FORMAT': 'json',
        'FEED_URI': 'reuters-archive.json'
    }

    start_urls = [
        'https://www.reuters.com/news/archive/',
    ]

    def parse(self, response):
        for article in response.css("article.story div.story-content a"):
            yield response.follow(article.css("a::attr(href)").extract_first(), self.parse_article)

        next_page_url = response.css('a.control-nav-next::attr(href)').extract_first()
        if (next_page_url is not None) & ('page=2' not in next_page_url):
            yield response.follow(next_page_url, self.parse)

    def parse_article(self, response):
        yield {
            'title': response.css('h1::text').extract_first().strip(),
        }

# this can be run only once from a Jupyter notebook due to Twisted
from scrapy.crawler import CrawlerProcess
process = CrawlerProcess()

process.crawl(ReutersArchiveSpider)
process.start()

print(glob.glob("*.json"))

os.system("cat 'reuters-archive.json'")

print('\n')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
