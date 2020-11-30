
"""
    Auteur: Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
    Description:  Script utilisé pour entrainé un ClassifierChain combiné avec un XGboost sur les Classes : TYPE_EFFET
"""

import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score,f1_score,classification_report,recall_score,precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


import spacy
nlp =spacy.load('fr')
from spacy.lang.fr.stop_words import STOP_WORDS


from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

mlb = MultiLabelBinarizer()

train = pd.read_pickle('./data_split/train.pkl')
# Pour faire un modèle sans le 
#train = train[~train['TEF_ID'].map(lambda x : 106 in x)]
X_train = train[['FABRICANT','CLASSIFICATION','DESCRIPTION_INCIDENT','ETAT_PATIENT']]
y_train = mlb.fit_transform(train['TEF_ID'])
test =  pd.read_pickle('./data_split/test.pkl')
#test = test[~test['TEF_ID'].map(lambda x : k in x)]
X_test = test[['FABRICANT','CLASSIFICATION','DESCRIPTION_INCIDENT','ETAT_PATIENT']]
y_test = mlb.transform(test['TEF_ID'])



preprocess = ColumnTransformer(
    [('description_tfidf',TfidfVectorizer(sublinear_tf=True, min_df=3,
                            ngram_range=(1, 1),
                            stop_words=STOP_WORDS,
                            max_features = 10000,norm = 'l2'), 'DESCRIPTION_INCIDENT'),
     
     ('etat_pat_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=3,ngram_range=(1, 1),
                                       stop_words=STOP_WORDS,
                                       max_features = 10000,norm = 'l2'), 'ETAT_PATIENT'),
     
     ('fabricant_tfidf',TfidfVectorizer(sublinear_tf=True, min_df=3,
                            ngram_range=(1, 1),
                            stop_words=STOP_WORDS,
                            max_features = 5000,norm = 'l2'), 'FABRICANT')
     ],
    
    remainder='passthrough')



print("Preprocessing...")
X_train_, X_test_ =preprocess.fit_transform(X_train),preprocess.transform(X_test)
print("Done !")



param = {'eta': 0.1,
 'max_depth': 7,
 'n_estimators': 7,
 'objective': 'binary:hinge',
 'n_jobs':-1}
 

clf = XGBClassifier(**param)

chains = [ClassifierChain(clf, order='random', random_state=i) for i in range(10)]

for chain in chains:
    chain.fit(X_train_, y_train)
    
y_pred_chains = np.array([chain.predict(X_test_) for chain in chains])

chain_f1_scores = [f1_score(y_test, y_pred_chain, average='samples') for y_pred_chain in y_pred_chains]

y_pred_ensemble = y_pred_chains.mean(axis=0)

thresholds = [0.01,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7]
for val in thresholds:
    print("For threshold: ", val)
    pred=y_pred_ensemble.copy()
  
    pred[pred>=val]=1
    pred[pred<val]=0
  
    precision = precision_score(y_test, pred, average='samples')
    recall = recall_score(y_test, pred, average='samples')
    f1 = f1_score(y_test, pred, average='samples')
   
    print("Samples-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    
    
