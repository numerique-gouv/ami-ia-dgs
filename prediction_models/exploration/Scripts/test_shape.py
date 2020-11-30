import warnings
warnings.filterwarnings('ignore')

from pprint import pprint
from time import time
import logging

import pandas as pd

import numpy as np
import sklearn as sk
import seaborn as sns

import nltk
from nltk import word_tokenize
lang ='french'

import clean_text
import skmultilearn




import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score,f1_score,classification_report
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD,IncrementalPCA,SparsePCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier

import spacy
nlp =spacy.load('fr')
from spacy.lang.fr.stop_words import STOP_WORDS
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

mrv = pd.read_csv('data/data_mrv/declaration_mrv_complet.csv', usecols=["DESCRIPTION_INCIDENT","ETAT_PATIENT","FABRICANT","CLASSIFICATION" ,"TEF_ID"])

mrv = mrv[~mrv['TEF_ID'].isin(['E1213', 'E1454', 'E1210', 'E2280'])] # delete non renseigne, autre, sans effet...

mrv = mrv.dropna(subset=['TEF_ID'])

text_columns =  ["DESCRIPTION_INCIDENT","ETAT_PATIENT","FABRICANT", "CLASSIFICATION"]

mrv[text_columns] = mrv[text_columns].applymap(clean_text.preprocess_text)

mrv['DESCRIPTION_INCIDENT'] = mrv['DESCRIPTION_INCIDENT'].fillna('')

mrv['ETAT_PATIENT'] = mrv['ETAT_PATIENT'].fillna('')              
mrv['CLASSIFICATION'] = mrv['CLASSIFICATION'].fillna('')
mrv['FABRICANT'] = mrv['FABRICANT'].fillna('')
mrv["text"] = mrv['DESCRIPTION_INCIDENT']+' '+mrv['ETAT_PATIENT']



# Load mapping and encode tef_id

#mapping = pd.read_csv('referentiel_corrige.csv')

#mapping = mapping[mapping["TEF_ID"]!='E1213']

le = LabelEncoder()

le.fit(mrv["TEF_ID"].values)

mrv['TEF_ID'] = le.transform(mrv['TEF_ID'])

mrv = mrv.sort_values(by=['TEF_ID']) 

mrv = mrv.groupby('text').agg({'TEF_ID':lambda x: list(set(x)),
                               'DESCRIPTION_INCIDENT':lambda x: list(set(x)),
                               'ETAT_PATIENT':lambda x: list(set(x)),
                               'FABRICANT':lambda x: list(set(x)),
                               'CLASSIFICATION':lambda x: list(set(x))
                               })
#mrv = mrv.drop('TEF_ID',axis=1)
#mrv = pd.merge(mrv,mrv_, on = 'text')
#mrv = mrv.drop_duplicates('ETAT_PATIENT')

mrv = mrv.reset_index()


# Split data

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1029)

mlb = MultiLabelBinarizer()

for train_index, test_index in msss.split(mrv['text'], mlb.fit_transform(mrv['TEF_ID'])):

    print("TRAIN:", train_index, "TEST:", test_index)

    train, test = mrv.loc[train_index],  mrv.loc[test_index]

    
print(train.shape, test.shape)

print(train.head())