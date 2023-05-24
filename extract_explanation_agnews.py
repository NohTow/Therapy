import codecs
import pandas as pd
import numpy as np
import logging
import argparse
import re
from itertools import chain

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

import pickle
import json

import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i", 
    "--input", 
    dest="input_file", 
    required=True,
    help="input csv file")
parser.add_argument(
    "--test_split_size",
    default=0.1,
    type=float,
    help="Size of the test split"
)


args = parser.parse_args()

logger = logging.getLogger(__name__)

args.file_stop = '/nfs/nas4.irisa.fr/deepfakes/deepfakes/image_repurposing/multimodal-fake-news-detection/IMATAG/common_words.total_en.u8.txt'
s_stopwords = set( list('ABCDEFGHIJKLN+MNOPQRSTUVWXYZ0123456789%') + [l.rstrip('\r\n') for l in codecs.open(args.file_stop,'r', 'utf-8')] + ['_', '___', '_______', '________', '___________','_____________', '___________________', '________________________', '_____________________________',] )

def simple_logistic_classify(X_tr, y_tr, X_test, y_test, C=1.0):
    model = LogisticRegression(C=C, random_state=42).fit(X_tr, y_tr)
    score = model.score(X_test, y_test)
    print("Test score : ", score)
    return model

logger.info("Reading data file")


# Reading data file
df = pd.read_csv(args.input_file, sep=",", encoding="utf8")
    

train_df, test_df = train_test_split(df, test_size=args.test_split_size, random_state=42)

X_train = train_df["review"].values.tolist()
y_train = train_df["label"].values.tolist()

X_test = test_df["review"].values.tolist()
y_test = test_df["label"].values.tolist()


corpus = X_train + X_test

tfidf_vectorizer = TfidfVectorizer(binary=False, stop_words=s_stopwords, ngram_range=(1, 1)).fit(corpus)
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = simple_logistic_classify(X_train_tfidf, y_train, X_test_tfidf, y_test)
feature_names = tfidf_vectorizer.get_feature_names_out()

sorted_lists_explainer = {}
for c in range(len(model.coef_)):
    sorted_lists_explainer[c] = sorted(zip(model.coef_[c], feature_names), reverse=True, key=lambda x: x[0])

top_words = {}
for i, duple in enumerate([sorted_lists_explainer[c][:200] for c in sorted_lists_explainer]):
    top_words[i] = {}
    for coeff, word in duple:
        top_words[i][word] = coeff
json.dump(top_words, open("./top_words/ag_news.json", 'w'))

