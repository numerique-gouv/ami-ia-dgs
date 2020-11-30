"""
Auteur: Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
Description: Script utilisé pour Finetuner un classifier XGboost avec la bibliothèque Optuna sur la variable TYPE_EFFET
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

import xgboost as xgb
from xgboost import XGBClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

import optuna
from optuna import Trial

import joblib
###### Chargement des données ########

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


########## Transformation des entrées ####################

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

X_train_, X_test_ =preprocess.fit_transform(X_train),preprocess.transform(X_test)

########## Construction du pipeline d'optimisation Optuna ####################


def objective(trial:Trial):
    """Fonction objectif pour l'optimiseur Optuna

    Args:
        trial (Trial): Un essai de paramètre

    Returns:
        f1 (float): score f1 sample pour le TEF_ID
    """
    msss = MultilabelStratifiedKFold(n_splits=2, random_state=1029)
    data,target = X_train_, y_train
    for train_index, test_index in msss.split(X_train_, y_train):
        train_x, valid_x, train_y, valid_y = X_train_[train_index], X_train_[test_index], y_train[train_index], y_train[test_index]
    
    # Liste des paramètres à optimiser
    param = {
        "n_estimators" : trial.suggest_int("n_estimators", 4, 30),
        "verbosity" :0,
        "early_stopping_rounds":15,
        "objective":trial.suggest_categorical("objective", ['binary:hinge', "binary:logistic"]),
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),#btree and dart use tree based models while gblinear uses linear functions
        "lambda": trial.suggest_loguniform("lambda", 1e-5, 1.0), #L2 regularization term on weights. Increasing this value will make model more conservative.
        "alpha": trial.suggest_loguniform("alpha", 1e-5, 1.0), #L1 regularization term on weights. Increasing this value will make model more conservative.
        "n_jobs":-1
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 2, 9) #Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
        param["eta"] = trial.suggest_loguniform("eta", 1e-4, 1.0)#alias: learning_rate
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-4, 1.0)#lias: min_split_loss
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])#Controls a way new nodes are added to the tree
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])#Type of sampling algorithm
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"]) #Type of normalization algorithm
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-5, 1.0) #Dropout rate
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-5, 1.0)#Probability of skipping the dropout procedure during a boosting iteration
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial,'eval-mlogloss')
    
    clf = OneVsRestClassifier(XGBClassifier(**param,callbacks=[pruning_callback]))
    clf.fit(train_x,train_y)
    pred = clf.predict(valid_x)
    print("f1 score on validation set...")
    f1 = f1_score(valid_y,pred, average='samples')
    print(f1)
    ################################
    print("f1 score on test set ...")
    pred_test = clf.predict(X_test_)
    f1_test = f1_score(y_test,pred_test, average='samples')
    print(f1_test)
    joblib.dump(clf,'last_xgb_effet.sav')
            
    return f1


#Optimisation
studyName = '30_h_study_effet'
maximum_time = 30*60*60#second
number_of_random_points = 1000
study = optuna.create_study(study_name = studyName,  direction="maximize")
study.optimize(objective, n_trials=number_of_random_points, timeout=maximum_time)# On créer 1000 points

#Sauvegarde du resultat
df = study.trials_dataframe()
df.to_json(studyName+'.json')
print(study.best_trial)