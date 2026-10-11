# all preprocessing codes

import os
import json
import re

import spacy
import unicodedata
from bs4 import BeautifulSoup
from textblob import TextBlob
from textblob import Word
from textblob.sentiments import NaiveBayesAnalyzer
from googletrans import Translator

from spacy.lang.en.stop_words import STOP_WORDS as stopwords

import nltk

fpath = os.path.join(os.path.dirname(__file__), 'data/contractions.json')
contractions = json.load(open(fpath))

nlp = spacy.load('en_core_web_sm')


def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')
    nltk.download('wordnet')
    nltk.download('maxent_ne_chunker')
    nltk.download('punkt')


# -------------------------
# A. Basic text features
# -------------------------

def word_count(x):
    return len(str(x).split())


def char_count(x):
    x = str(x)
    return len(re.sub(r'\s', '', x))


def avg_word_len(x):
    x = str(x)
    if word_count(x) == 0:
        return 0
    return char_count(x) / word_count(x)


def stop_words_count(x):
    x = str(x).lower()
    return len([word for word in x.split() if word in stopwords])


def hashtags_count(x):
    return len(re.findall(r'#\w+', str(x)))


def mentions_count(x):
    return len(re.findall(r'@\w+', str(x)))


def numerics_count(x):
    return len(re.findall(r'\b\d+\b', str(x)))


def upper_case_count(x):
    return len([word for word in str(x).split() if word.isupper()])


def get_basic_features(df, text_col='tweets'):
    """
    Adds basic NLP features to a DataFrame.
    Works with: df = ps.get_basic_features(df)
    If your text column is not named 'tweets', use:
    df = ps.get_basic_features(df, 'your_column_name')
    """
    df['word_count'] = df[text_col].apply(word_count)
    df['char_count'] = df[text_col].apply(char_count)
    df['avg_word_len'] = df[text_col].apply(avg_word_len)
    df['stop_words_count'] = df[text_col].apply(stop_words_count)
    df['hashtags_count'] = df[text_col].apply(hashtags_count)
    df['mentions_count'] = df[text_col].apply(mentions_count)
    df['numerics_count'] = df[text_col].apply(numerics_count)
    df['upper_case_count'] = df[text_col].apply(upper_case_count)
    return df


# -------------------------
# B. Preprocessing and cleaning
# -------------------------

def to_lower_case(x):
    return str(x).lower()


def contraction_to_expansion(x):
    x = str(x)
    return " ".join([
        contractions.get(word.lower(), word)
        for word in x.split()
    ])


def remove_emails(x):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return re.sub(pattern, '', str(x))


def count_emails(x):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return len(re.findall(pattern, str(x)))


def count_urls(x):
    pattern = r"http\S+|www\.\S+"
    return len(re.findall(pattern, str(x)))


def remove_urls(x):
    pattern = r"http\S+|www\.\S+"
    return re.sub(pattern, '', str(x))


def count_rt(x):
    pattern = r'\bRT @\w+'
    return len(re.findall(pattern, str(x)))


def remove_rt(x):
    pattern = r'\bRT @\w+'
    return re.sub(pattern, '', str(x))


def remove_html_tags(x):
    return BeautifulSoup(str(x), 'lxml').get_text()


def remove_accented_chars(x):
    return unicodedata.normalize(
        'NFKD', str(x)
    ).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def remove_mentions(x):
    return re.sub(r'@\w+', '', str(x)).strip()


def remove_special_chars(x):
    return re.sub(r'[^\w\s]', '', str(x))


def remove_repeated_chars(x):
    return re.sub(r'(.)\1+', r'\1\1', str(x))


def remove_stop_words(x):
    return ' '.join([
        word for word in str(x).split()
        if word not in stopwords
    ])


def lemmatize_noun_verb(x):
    doc = nlp(str(x))
    tokens = []

    for token in doc:
        if token.pos_ in ['NOUN', 'VERB']:
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)

    x = ' '.join(tokens)
    return re.sub(r'\s\.', '.', x)


def lemmatize(x):
    doc = nlp(str(x))
    return ' '.join([token.lemma_ for token in doc])


def remove_common_words(x, common_words):
    return ' '.join([
        word for word in str(x).split()
        if word not in common_words
    ])


def remove_rare_words(x, rare_words):
    return ' '.join([
        word for word in str(x).split()
        if word not in rare_words
    ])


def correct_spelling(x):
    words = []
    for word in str(x).split():
        words.append(Word(word).correct())
    return ' '.join(words)


def get_noun_phrase(x):
    return TextBlob(str(x)).noun_phrases


def n_gram(x, n=2):
    return list(TextBlob(str(x)).ngrams(n))


def singularize_words(x):
    blob = TextBlob(str(x))
    return ' '.join([
        word.singularize() if tag == 'NNS' else word
        for word, tag in blob.tags
    ])


def pluralize_words(x):
    blob = TextBlob(str(x))
    return ' '.join([
        word.pluralize() if tag == 'NN' else word
        for word, tag in blob.tags
    ])


def sentiment_analysis(x):
    return TextBlob(
        str(x), analyzer=NaiveBayesAnalyzer()
    ).sentiment.classification


def detect_language(x):
    translator = Translator()
    return translator.detect(str(x)).lang


def translate(x, dest='en'):
    translator = Translator()
    return translator.translate(str(x), dest=dest).text
