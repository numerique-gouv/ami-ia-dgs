"""
Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
Description: 
- Construction d'un modèle XGBoost + TF-IDF pour la variable DCO
- Optimisation avec Optuna
- Sauvegarde de l'étude au format csv
"""



#Librairies

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd



import clean_text

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score,f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
import spacy
nlp =spacy.load('fr')
from spacy.lang.fr.stop_words import STOP_WORDS


import xgboost as xgb

import optuna
from optuna import Trial

#Données

df_declaration_mrv = pd.read_csv("data/data_mrv/declaration_mrv_complet.csv")#delimiter=';',encoding='ISO-8859-1')
id_to_dco = pd.read_csv("data/ref_MRV/referentiel_dispositif.csv",delimiter=';',encoding='ISO-8859-1')

#Charegement des colonnes utiles
df = df_declaration_mrv[['DESCRIPTION_INCIDENT','TYPE_VIGILANCE','LIBELLE_COMMERCIAL',
                         'REFERENCE_COMMERCIALE','ETAT_PATIENT','FABRICANT','DCO_ID','CLASSIFICATION']]
# On complète les NaN avec du vide
df['ETAT_PATIENT'] = df['ETAT_PATIENT'].fillna("")
df['DESCRIPTION_INCIDENT'] = df['DESCRIPTION_INCIDENT'].fillna("")
df['LIBELLE_COMMERCIAL'] = df['LIBELLE_COMMERCIAL'].fillna("")
df['FABRICANT'] = df['FABRICANT'].fillna("")
df["REFERENCE_COMMERCIALE"] = df['REFERENCE_COMMERCIALE'].fillna("")
df['TYPE_VIGILANCE'] = df['TYPE_VIGILANCE'].fillna("")
df['CLASSIFICATION'] = df['CLASSIFICATION'].fillna('')


# On ajoute des collones pertinentes
df['des_lib'] = df['LIBELLE_COMMERCIAL']+ ' ' + df['DESCRIPTION_INCIDENT']
df['fab_lib'] = df['LIBELLE_COMMERCIAL']+ ' ' + df['FABRICANT']
df['com'] = df['LIBELLE_COMMERCIAL']+ ' ' + df['REFERENCE_COMMERCIALE']
df['Text'] = df['LIBELLE_COMMERCIAL']+ ' ' + df['FABRICANT'] + "" + df['DESCRIPTION_INCIDENT']

# On nettoie les données :
for col in  ['DESCRIPTION_INCIDENT','LIBELLE_COMMERCIAL','ETAT_PATIENT','Text',"des_lib","fab_lib"] :
    df[col] = df[col].map(lambda x: clean_text.preprocess_text(x))

n = 15
# On filtre pour a voir plus de n observations par classse
df_n = df.groupby("DCO_ID").filter(lambda x: len(x) > n)

# On encode les labels
le = LabelEncoder()
df_n.DCO_ID = le.fit_transform(df_n.DCO_ID.values)
#On encode le type de vigilance
df_n.TYPE_VIGILANCE = le.fit_transform(df_n.TYPE_VIGILANCE.values)
#On encode la classifcation 
df_n.CLASSIFICATION = le.fit_transform(df_n.CLASSIFICATION.values)

# On selection les variables de test en faisant attention aux doublons
train_index,test_index = next(GroupShuffleSplit(random_state=1029).split(df_n, groups=df_n['DESCRIPTION_INCIDENT']))
df_train, df_test = df_n.iloc[train_index], df_n.iloc[test_index]



# On transforme les variables
preprocess = ColumnTransformer(
    [   
     ('libelle_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=3,ngram_range=(1, 1),
                                       stop_words=STOP_WORDS,
                                       max_features = 10000,norm = 'l2'), 'LIBELLE_COMMERCIAL'),
     
     ('description_tfidf',TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 1),
                            stop_words=STOP_WORDS,
                            max_features = 10000,norm = 'l2'), 'DESCRIPTION_INCIDENT'),
     
    ('fabricant_tfidf',TfidfVectorizer(sublinear_tf=True, min_df=3,
                            ngram_range=(1, 1),
                            stop_words=STOP_WORDS,
                            max_features = 10000,norm = 'l2'), 'FABRICANT')],
    
    remainder='passthrough')

X = df_train[['DESCRIPTION_INCIDENT','FABRICANT','LIBELLE_COMMERCIAL']]  
X_prep = preprocess.fit_transform(X)

y = df_train.DCO_ID


#Fonction objectif
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial:Trial):
    data,target = X_prep, y
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.1)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)
    
    # Liste des paramètres à optimiser
    param = {
        #"silent": 1,
        "verbosity" :1,
        "objective":'multi:softmax',
        "eval_metric":'mlogloss',
        "num_class":len(y.unique()),
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),#btree and dart use tree based models while gblinear uses linear functions
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0), #L2 regularization term on weights. Increasing this value will make model more conservative.
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0), #L1 regularization term on weights. Increasing this value will make model more conservative.
        "nthread":-1
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 2, 9) #Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
        param["eta"] = trial.suggest_loguniform("eta", 1e-5, 1.0)#alias: learning_rate
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)#lias: min_split_loss
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])#Controls a way new nodes are added to the tree
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])#Type of sampling algorithm
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"]) #Type of normalization algorithm
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0) #Dropout rate
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)#Probability of skipping the dropout procedure during a boosting iteration
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial,'eval-mlogloss')
    
    bst = xgb.train(param, dtrain,20,evals=[(dvalid, "eval")],callbacks=[pruning_callback],early_stopping_rounds=15)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    f1_weighted = f1_score(valid_y, pred_labels, average='weighted')
    print(f1_weighted)
    multiclass_log_loss = bst.best_score
    #pprint()
    bst.save_model('last_xgb.model')
            
    return multiclass_log_loss


#Optimisation
studyName = '14_h_study'
maximum_time = 14*60*60#second
number_of_random_points = 100
#optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(study_name = studyName,  direction="minimize")
study.optimize(objective, n_trials=100, timeout=maximum_time)# On créer 100 points
#Alternative
#study = optuna.create_study(study_name = studyName, sampler=TPESampler(n_startup_trials=number_of_random_points),direction="maximize")

#Suvegarde du resultat
df = study.trials_dataframe()
df.to_json(studyName+'.json')
print(study.best_trial)