"""
Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
Description: 
Utilisation de la bibliothèque sentence transformer pour encodé les déclarations
Sauvegarde des vecteurs train et test au format npy pour la variable TEF_ID 
"""


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import clean_text

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

from sentence_transformers import SentenceTransformer
from sentence_transformers import models

from numpy import savez_compressed


from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os
#On créer un répertoire pour sauvegarder les résultats
directory = './results/'
if not os.path.exists(directory):
    os.makedirs(directory)

#load data
mlb = MultiLabelBinarizer()

train = pd.read_pickle('./data_split/train.pkl')
test =  pd.read_pickle('./data_split/test.pkl')
# Pour faire un modèle sans le 

X_train = train[['FABRICANT','CLASSIFICATION','DESCRIPTION_INCIDENT','ETAT_PATIENT','ACTION_PATIENT']]
y_train = mlb.fit_transform(train['TEF_ID'])


X_test = test[['FABRICANT','CLASSIFICATION','DESCRIPTION_INCIDENT','ETAT_PATIENT','ACTION_PATIENT']]
y_test = mlb.transform(test['TEF_ID'])


for col in  ['DESCRIPTION_INCIDENT','ETAT_PATIENT','FABRICANT','ACTION_PATIENT'] :
    X_train[col] = X_train[col].map(lambda x: clean_text.preprocess_text(x))
    X_test[col] = X_test[col].map(lambda x: clean_text.preprocess_text(x))



np.save('./results/y_train.npy',y_train)
np.save("./results/y_test.npy",y_test)


#Construction du modèle d'ambeding

word_embedding_model =  models.Transformer('./Models/Models/finetune2_nli_dgs_max_len_128_fp16_dynamic_padding_smart_batching_batch_32_seed_321/')

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


#Construction du Pipeline pour traiter chaque collone de manière unique

class CamenBertVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X,y=None):
        return self

    def transform(self, X):
        if type(X)==str:       
            return np.array(self.model.encode([X],show_progress_bar=True, batch_size=10))
        elif type(X)==list :     
            return np.array(self.model.encode(X,show_progress_bar=True, batch_size=10))
        elif type(X)==pd.Series :      
            return np.array(self.model.encode(X.values.tolist(),show_progress_bar=True, batch_size=10))
        else :
            print('Please enter a list of string or a pandas Series')



def preprocess_pipeline(model =model, n =200) :
    return Pipeline([
        ('vect', CamenBertVectorTransformer(model = model)),
        ('svd', TruncatedSVD(n_components=n))
    ])

preprocess = ColumnTransformer(
    [('etat_pat_bert', CamenBertVectorTransformer(model = model) , 'ETAT_PATIENT'),
         
     ('description_bert',CamenBertVectorTransformer(model = model), 'DESCRIPTION_INCIDENT'),
     #('description_bert_svd',TruncatedSVD(n_components =200), 'description_bert')
     
     ('action_pat_bert',CamenBertVectorTransformer(model = model), 'ACTION_PATIENT'),
     
     ('fabricant_bert',CamenBertVectorTransformer(model = model), 'FABRICANT'),
          
    ],
    #
    remainder='passthrough')


X_train = X_train[['DESCRIPTION_INCIDENT','ETAT_PATIENT','FABRICANT','ACTION_PATIENT']]
X_test = X_test[['DESCRIPTION_INCIDENT','ETAT_PATIENT','FABRICANT','ACTION_PATIENT']]


#Calcul des vecteurs
X_train = preprocess.fit_transform(X_train)
X_test = preprocess.fit_transform(X_test)


np.save('./results/dgs_camenbert_train_vec.npy',X_train)
np.save('./results/dgs_camenbert_test_vec.npy',X_test)