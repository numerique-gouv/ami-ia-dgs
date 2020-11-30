"""
Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
Description: 
- Construction d'un modèle SVM + TF-IDF pour la varaible tf-idf
- print du f1 sample
"""

import joblib
import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score,f1_score,classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


le = joblib.load('../data_split/TEF_ID_encodeur.sav')
k=le.transform(['E1213'])[0] #Le code de la variable Non Renseigné

mlb = MultiLabelBinarizer()

train = pd.read_pickle('../data_split/train.pkl')
# Pour faire un modèle sans le 
#train = train[~train['TEF_ID'].map(lambda x : 106 in x)]
X_train = train[['FABRICANT','CLASSIFICATION','DESCRIPTION_INCIDENT','ETAT_PATIENT']]
y_train = mlb.fit_transform(train['TEF_ID'])
test =  pd.read_pickle('../data_split/test.pkl')
#test = test[~test['TEF_ID'].map(lambda x : k in x)]
X_test = test[['FABRICANT','CLASSIFICATION','DESCRIPTION_INCIDENT','ETAT_PATIENT']]
y_test = mlb.transform(test['TEF_ID'])



##Pipeline TFIDF SVM

preprocess = ColumnTransformer(
    [('description_tfidf',TfidfVectorizer(sublinear_tf=True, min_df=3,
                            ngram_range=(1, 1),
                            max_features = 10000,norm = 'l2'), 'DESCRIPTION_INCIDENT'),
     
     ('etat_pat_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=3,ngram_range=(1, 1),
                                       
                                       max_features = 10000,norm = 'l2'), 'ETAT_PATIENT'),
     
     ('fabricant_tfidf',TfidfVectorizer(sublinear_tf=True, min_df=3,
                            ngram_range=(1, 1),
                            
                            max_features = 5000,norm = 'l2'), 'FABRICANT')
     ],
    
    remainder='passthrough')

pipeline = Pipeline([
    ('vect', preprocess),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced',max_iter=10000))),
])


pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)
f1 = f1_score(y_test , y_pred,average='samples')
print('f1_score samples : ',f1)