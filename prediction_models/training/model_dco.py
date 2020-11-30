"""
Auteur:  
    Quillivic Robin, Data Scientist chez StarClay,rquilivic@starclay.fr
Description:  
    Ce fichier contient les fonctions nécessaires pour entraîner, tester le modèle pour inférer les DCO. 
    Il y a 4 fonction principales : 
    - prépare_data: préparer les données pour l'entrainement (nettoyage, encodage etc.)
    - train_DCO: Entrainement du pipeline tfidf+svm sur les données merveilles
    - evaluate_model: évaluation du modèle par la balance accuracy
    - repro_result : entraine et test le modèle sur différents jeu de données (ensemble, les données des citoyens et des professionnels)
"""

import pandas as pd
import numpy as np

import clean_text
import sklearn as sk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from clean_text import STOP_WORDS

import joblib
import os
import yaml

from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score


# Path config
from datetime import date
today = date.today()
d4 = today.strftime("%b-%d-%Y")
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)
MODEL_PATH = os.path.join(config_data['training']['save_path'], 'model_'+str(d4))

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

import logging
import logging.config


with open(os.path.join(os.path.dirname(__file__), '../log_conf.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)

logger = logging.getLogger('inference_dco')


def prepare_data(df: pd.DataFrame, clean=True, n=3)->(pd.DataFrame,sk.preprocessing._label.LabelEncoder ) :
    """ Permet de préparer les données de la base MRveil
    avant l'entrainement des modèles. Cette préparation est spécifique
    pour l'inférence de la DCO. Elle se compose des étapes suivantes :
    - suppression des lignes contenant les DCO à ignorer
    - gestion des valeurs nulles
    - nettoyage des données textulles (ponctuation etc.)
    - encodage des colonnes classifications et  type de vigilances.
    - augmention des données rares pour permette les cross validation

    Args:
        df (pd.DataFrame): données de la base MRveill
        clean (bool, optional): Nettoyage textuelles des données. Defaults to True.
        n (int, optional): nombre de fois qu'on réplique les données rares. Les données considérées comme rare sont
        celles qui apparaissent moins de n fois. Defaults to 3.

    Returns:
        df_prep (pd.DataFrame): Données préparée et nettoyée pour l'entrainement
        le (sklearn.preprocessing._label.LabelEncoder) : Label encoder fitter sur la variable classification
    """
    logger.info('    Suppression des lignes avec des labels peu significatif')
    drop = ['TOUTES LES DCO', 'AUTRE DIV', 'INCONNU - REACTO', 'NON DIV',
            'NRV', 'INCONNU', 'NDM', 'NMV', 'PSIG', 'PSIG2', 'Tous', 'TOUS LES DIV']
    df_prep = df[~df['DCO'].map(lambda x: x in drop)]
    logger.info('    Done ! ')

    logger.info('    Gestion des valeurs nulles')
    # On complète les NaN avec du vide
    df_prep['ETAT_PATIENT'] = df_prep['ETAT_PATIENT'].fillna("")
    df_prep['DESCRIPTION_INCIDENT'] = df_prep['DESCRIPTION_INCIDENT'].fillna(
        "")
    df_prep['LIBELLE_COMMERCIAL'] = df_prep['LIBELLE_COMMERCIAL'].fillna("")
    df_prep['FABRICANT'] = df['FABRICANT'].fillna("")
    df_prep["REFERENCE_COMMERCIALE"] = df_prep['REFERENCE_COMMERCIALE'].fillna(
        "")
    df_prep['TYPE_VIGILANCE'] = df_prep['TYPE_VIGILANCE'].fillna("")
    df_prep['CLASSIFICATION'] = df_prep['CLASSIFICATION'].fillna('')
    logger.info('    Done ! ')

    # On nettoie les données :
    if clean == True:
        logger.info('    Nettoyage des données...')
        for col in ['DESCRIPTION_INCIDENT', 'LIBELLE_COMMERCIAL', 'FABRICANT']:
            df_prep[col] = df_prep[col].map(
                lambda x: clean_text.preprocess_text(x))
        logger.info('    Done ! ')

    le = LabelEncoder()

    # On encode le type de vigilance
    df_prep.TYPE_VIGILANCE = le.fit_transform(df_prep.TYPE_VIGILANCE.values)
    # On encode la classification
    df_prep.CLASSIFICATION = le.fit_transform(df_prep.CLASSIFICATION.values)
    # On encode les labels
    df_prep.DCO_ID = le.fit_transform(df_prep.DCO_ID.values)
    df_inf = df_prep.groupby("DCO_ID").filter(lambda x: len(x) < n)
    df_prep = df_prep.append([df_inf]*n, ignore_index=True)

    return (df_prep, le)


def train_DCO(df_train: pd.DataFrame, cv=3, save=False):
    """Fonction qui permet d'entraîner le modèle des DCO, celui ci est un pipeline réalisant une vectorisation  utilisant le
    tfidf finetuné avec la librairie Optuna puis utilise un  SVM linéaire pour classifier les DCO.

    Args:
        df_train (pd.DataFrame): dataframe avec les données complètes et nettoyées
        cv (int, optional): nombre de crossvalidation dans le calibratedClassifier pour probabiliser le SVM  (Ne pas dépasser 6, pour que le temps d'execution reste raisonable). Defaults to 3.
        save (bool, optional): permet de sauvegarder le pipeline entrainé si True. Defaults to False.

    Returns:
        pipeline (sk.pipeline.Pipeline):  le pipeline entrainé sur les données d'entrées pour inférer la DCO
    """
      
      
    preprocess = ColumnTransformer(
        [('reference_tfidf', TfidfVectorizer(sublinear_tf=True,
                                             analyzer='word',
                                             min_df=4,
                                             ngram_range=(1, 1),
                                             stop_words=STOP_WORDS,
                                             max_features=5804,
                                             norm='l1'), 'REFERENCE_COMMERCIALE'),

         ('libelle_tfidf', TfidfVectorizer(sublinear_tf=True,
                                           analyzer='word',
                                           min_df=1,
                                           ngram_range=(1, 1),
                                           stop_words=STOP_WORDS,
                                           max_features=8655,
                                           norm='l2'), 'LIBELLE_COMMERCIAL'),

            ('description_tfidf', TfidfVectorizer(sublinear_tf=True,
                                                  min_df=1,
                                                  ngram_range=(1, 1),
                                                  stop_words=STOP_WORDS,
                                                  max_features=18294,
                                                  norm='l2'), 'DESCRIPTION_INCIDENT'),

            ('fabricant_tfidf', TfidfVectorizer(sublinear_tf=True,
                                                analyzer='char',
                                                min_df=2,
                                                ngram_range=(1, 1),
                                                stop_words=STOP_WORDS,
                                                max_features=2387,
                                                norm='l2'), 'FABRICANT')],

        remainder='passthrough')

    pipeline = Pipeline([
        ('vect', preprocess),
        ('clf', CalibratedClassifierCV(
            LinearSVC(class_weight='balanced'), cv=cv, method='isotonic')),
    ])

    X = df_train[['DESCRIPTION_INCIDENT', 'FABRICANT',
                  'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']]
    y = df_train.DCO_ID

    pipeline.fit(X, y)

    if save == True:
        joblib.dump(pipeline, os.path.join(MODEL_PATH,'DCO_model_last.sav'))

    return pipeline


def evaluate_model(pipeline: sk.pipeline.Pipeline, df_test: pd.DataFrame):
    """
    Renvoie le score adéquate pour le pipeline chargé. Dans le Cas du DCO, la balance accuracy est le score adéquate pour notre problème
    Entrées:
    
    Args:
        pipeline (sk.pipeline.Pipeline): le pipeline entrainé sur les données de la DCO
        df_test (pd.DataFrame): les données de test
        
    Returns:
        score (float) : le score balanced_accuracy_score du modèle
    """

    X_test = df_test[['DESCRIPTION_INCIDENT', 'FABRICANT',
                      'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']]
    y_test = df_test.DCO_ID

    y_pred = pipeline.predict(X_test)

    logger.info("Justesse" +str(accuracy_score(y_test, y_pred)))
    logger.info("Balanced_accuracy : "+str (balanced_accuracy_score(y_test, y_pred)))
    logger.info("f1-weighted : "+str( f1_score(y_test, y_pred, average='weighted')))
    score = balanced_accuracy_score(y_test, y_pred)
    return(score)


def repro_result(df: pd.DataFrame, n=3, citoyen=True, pro=True, clean=False) -> (float, float, float):
    """ 
    Fonction qui permet de reproduire les résultats annoncés. Les résultats sont stockés dans le fichier excel performance.xlsx
    afin de garder une trace au fur et à mesure des entraînements.

    Args:
        df (pd.DataFrame): 
        n (int, optional): Le nombre de cross validation pour train_DCO. Defaults to 3
        citoyen (bool, optional): le score concernant les citoyens doit t-il être calculé ?. Defaults to True
        pro (bool, optional): e score concernant les professionnels doit t-il être calculé. Defaults to True
        clean (bool, optional): faut-il nettoyer les données textuelles avant de calculer les score ?. Defaults to False

    Returns:
        score (float): score globale pour les l'ensemble du Data set
        score_citoyens (float): score calculé pour un modèle entrainé seulement sur les données citoyennes
        score_pro (float) :  score calculé pour un modèle entrainé seulement sur les données professionnelles
    """
    
    df_n = df.copy()

    # Selection des index citoyen et pro
    citoyen_index = df_n[df_n.TYPE_DECLARANT == 'Citoyen'].index
    professionel_index = df_n[df_n.TYPE_DECLARANT ==
                              'Etablissements et professionnels de santé'].index

    logger.info('    Entrainement du modèle sur un sous échantillon puis test.')
    # Entrainement du modèle de base
    train_index, test_index = next(GroupShuffleSplit(
        random_state=1029).split(df_n, groups=df_n['DESCRIPTION_INCIDENT']))
    df_train, df_test = df_n.iloc[train_index], df_n.iloc[test_index]
    df_inf = df_train.groupby("DCO_ID").filter(lambda x: len(x) < n)
    df_train = df_train.append([df_inf]*n, ignore_index=True)
    score = evaluate_model(train_DCO(df_train, cv=n), df_test)
    logger.info("    Done ! ")

    score_citoyens = 0
    score_pro = 0
    logger.info('    Entrainement du modèle sur les déclarations de citoyens puis test.')
    if citoyen == True:
        df_c = df_n.loc[citoyen_index]
        # On selection les variables de test en faisant attention aux doublons
        train_index, test_index = next(GroupShuffleSplit(
            random_state=1029).split(df_c, groups=df_c['DESCRIPTION_INCIDENT']))
        df_train, df_test = df_c.iloc[train_index], df_c.iloc[test_index]
        df_inf = df_train.groupby("DCO_ID").filter(lambda x: len(x) < n)
        df_train = df_train.append([df_inf]*n, ignore_index=True)
        score_citoyens = evaluate_model(train_DCO(df_train, cv=n), df_test)
    logger.info('    Done!')

    logger.info('    Entrainement du modèle sur les déclarations de professionnels puis test.')
    if pro == True:
        df_p = df_n.loc[professionel_index]
        # On selection les variables de test en faisant attention aux doublons
        train_index, test_index = next(GroupShuffleSplit(
            random_state=1029).split(df_p, groups=df_p['DESCRIPTION_INCIDENT']))
        df_train, df_test = df_p.iloc[train_index], df_p.iloc[test_index]
        df_inf = df_train.groupby("DCO_ID").filter(lambda x: len(x) < n)
        df_train = df_train.append([df_inf]*n, ignore_index=True)
        score_pro = evaluate_model(train_DCO(df_train, cv=n), df_test)
    logger.info('    Done!')
    return (score, score_citoyens, score_pro)
