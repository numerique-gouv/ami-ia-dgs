"""
Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
Description: 
  - Lemmatise les données textuelles de la base MRVeil et les sauvegarde dans une colonne de la DataFrame

"""

import pandas as pd
import gensim
import numpy as np
import sklearn as sk
import seaborn as sns

import nltk
from nltk import word_tokenize
lang ='french'

import clean_text



from scipy.stats import randint
from scipy.sparse import csr_matrix


import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD,IncrementalPCA,SparsePCA

import spacy
nlp =spacy.load('fr')
from spacy.lang.fr.stop_words import STOP_WORDS 


df_declaration_mrv = pd.read_csv("data/data_mrv/declaration_mrv.csv",delimiter=';',encoding='ISO-8859-1')
id_to_dco = pd.read_csv("data/ref_MRV/referentiel_dispositif.csv",delimiter=';',encoding='ISO-8859-1')

df = df_declaration_mrv[['DESCRIPTION_INCIDENT','LIBELLE_COMMERCIAL','DCO_ID']]


df['Text'] = df['LIBELLE_COMMERCIAL']+ ' ' + df['DESCRIPTION_INCIDENT']

df = df.dropna()


def normalize_text(text:str)->str:
    doc = nlp(text)
    Norm  = [elt.lemma_+'_'+elt.pos_ for elt in doc]
    return(" ".join(Norm))

df.Text = df.Text.map(lambda x: clean_text.preprocess_text(x))
print("cleanning terminé, debut de la normalisation...")
df.Text = df.Text.map(lambda x: normalize_text(x))
print("Normalisation terminé..")

df.to_csv('df_text_Libelle_Descr_clean.csv')