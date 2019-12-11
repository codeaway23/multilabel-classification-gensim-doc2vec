from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import pickle
import re

##remove stop words from word list
def remove_stop_words(sample_words):
    stop_words = set(stopwords.words('english'))
    sample_words = [x for x in sample_words if not x in stop_words]
    return sample_words

##remove special characters from word list
def remove_special_char(sample_words):
    special_char = set(punctuation) 
    sample_words = [x for x in sample_words if not x in special_char]    
    return sample_words

##lemmatize words in word list
def lemmatizer(sample_words):
    lemmatizer = WordNetLemmatizer()
    sample_words = [lemmatizer.lemmatize(x) for x in sample_words]
    return sample_words

##all words in lower case
def lower_case(sample_words):
    sample_words = [x.lower() for x in sample_words]
    return sample_words

##normalize a word list (if document already tokenized)
def normalize_word_list(sample_words,
                        lowercase=True,
                        stopwords=True,
                        specialchar=True,
                        lemmatize=True):
    if lowercase:
        sample_words = lower_case(sample_words)
    if stopwords:
        sample_words = remove_stop_words(sample_words)
    if specialchar:
        sample_words = remove_special_char(sample_words)
    if lemmatize:
        sample_words = lemmatizer(sample_words)
    sample_words = ' '.join(sample_words)
    return sample_words

##normalize a list of sentences
def normalize_sent_list(sample_sents,
                        lowercase=True,
                        stopwords=True,
                        specialchar=True,
                        lemmatize=True):    
    print("Pre-processing text ...")
    sent_list = sample_sents
    for i in range(len(sample_sents)):
        sent_list[i] = re.findall(r"[\w']+|[.,!?;]", sent_list[i])
        sent_list[i] = normalize_word_list(sent_list[i],
                            lowercase=True,
                            stopwords=True,
                            specialchar=True,
                            lemmatize=True)
    return sent_list
