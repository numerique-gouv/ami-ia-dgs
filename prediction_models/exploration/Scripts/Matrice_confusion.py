"""
Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
 Description: 
Affiche une matrice de confusion pour un modèle de svm sur les DCO
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
df.Text = df.Text.map(lambda x: clean_text.preprocess_text(x))

def select_raw_by_nb_obs(df:pd.DataFrame, seuil:int)->(pd.DataFrame) :
    """
    Renvoie les lignes ou le nombre d'observations est supérieur au seuil entrée
    """
    S = df.groupby('DCO_ID').count()>seuil
    liste_DCO =S[S['Text']==True].index
    df_utilisable= df[df['DCO_ID'].isin(liste_DCO)]
    #df_reduit = df_utilisale[df_utilisale['DCO_ID']>2900]
    #print(len(df_reduit))
    return(df_utilisable)

df_utilisable_10 = select_raw_by_nb_obs(df,10)

def construction_features(df_utilisable:pd.DataFrame,method='tfidf')->(np.array,np.array) :
    """
    Vectorise les données textuelles selon la methode précisé,
    """
    if method=='tfidf':
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 2),
                            stop_words=STOP_WORDS,
                            max_features = 10000)

        features = tfidf.fit_transform(df_utilisable.Text)
        #hashing = HashingVectorizer()
        #features = csr_matrix(hashing.fit_transform(df_utilisable.Text))
 
        targets = df_utilisable.DCO_ID
    return (features, targets)

features_10,targets_10 = construction_features(df_utilisable_10)

X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features_10,targets_10,df_utilisable_10.index,test_size=0.25,random_state=1)
                                                                    
model = LinearSVC(class_weight='balanced')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

id_to_dco = id_to_dco.set_index('DCO_ID')
print("ntrainement terminé")
conf_mat = confusion_matrix(y_test, y_pred)#,labels =id_to_dco.index.values)

df_conf_mat = pd.DataFrame(data=conf_mat)
df_conf_mat.to_csv('conf_mat.csv')

print('conf mat terminé')
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16)
plt.savefig('conf_mat.jpg')
print('figure terminéé')
        
