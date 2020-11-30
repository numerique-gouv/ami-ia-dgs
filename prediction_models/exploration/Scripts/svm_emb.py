"""
Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
Description: 
- Construction d'un modèle SVM à partir des sentence Embeding construit avec Camembert 
"""


import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score,f1_score


from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import joblib
import time

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=300)


t = time.time()

X_train = np.load('results/dgs_camenbert_train_vec.npy')
X_test =np.load('results/dgs_camenbert_test_vec.npy')

y_train = np.load('results/y_train.npy')
y_test = np.load('results/y_test.npy')

X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)

pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced', max_iter=20000))),
], )

pipeline.fit(X_train,y_train)



#pipeline = joblib.load('Effet_model_dgs_emb.sav')
y_pred = pipeline.predict(X_test)
#print(time.time())
#import joblib
#filename = 'Effet_model_dgs_emb.sav'
#joblib.dump(pipeline, filename)

print(f1_score(y_test,y_pred,average='samples'))
print(time.time()-t,'secondes')
