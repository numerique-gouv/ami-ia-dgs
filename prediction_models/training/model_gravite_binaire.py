"""
Auteur:  
    Quillivic Robin, Data Scientist chez StarClay, rquilivic@starclay.fr
Description:  
    Ce fichier contient les fonctions nécessaire pour entraîner, tester le modèle pour inférer le Gravité à 5 classes.
    Ce fichier fait partie du Livrable 2 et son architecture est de la même forme que les 5 autres fichiers de classification
    Il y a 4 fonctions principales : 
    - preprare_data: préparer les données pour l'entrainement (nettoyage, encodage etc.)
    - train_Gravite: Entrainement du pipeline tfidf+svm sur les données merveilles
    - evaluate_model: évaluation du modèle par la balance accuracy
    - repro_result : entraine et test le modèle sur différents jeu de données ( l'ensemble des données, les données des citoyens et des professionnels)
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

from OrdinalClassifier import OrdinalClassifier
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
import yaml

with open(os.path.join(os.path.dirname(__file__), '../log_conf.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)

logger = logging.getLogger('inference')

# On encode les labels


def GRAVITE_ENC(x):
    """ Fonction pour encoder les 2 valeurs de la gravité avec 0 pour la classe critique et 1 pour la classe. 

    Args:
        x (str): un élément de la colonne gravité

    Returns:
        int : code associé à la Gravité
    """
    if x == 'CRITI':
        return 0
    else:
        return 1


def prepare_data(df: pd.DataFrame, clean=True):
    """"cette fonction permet de préparer les données de la base MRveil
    avant de l'entrainement des modèles. Cette préparation est spécifique
    pour l'inférence de la Gravité séparé en 2 classes. Elle se compose des étapes suivantes :
    - suppression des lignes contenant les valeurs nulles
    - gestion des valeurs nulles
    - nettoyage des données textulles (ponctuation etc.)
    - encodage de la gravité avec la fonction GRAVITE_ENC
    - 

    Args:
        df (pd.DataFrame): données de la base MRveill
        clean (bool, optional): Nettoyage textuelles des données. Defaults to True.

    Returns:
        df_prep (pd.DataFrame): Données préparée et nettoyée pour l'entrainement des modèles de Gravité
    """
    logger.info('    Suppression des lignes avec peu des labels vide')
    df_prep = df[df['GRAVITE'].notna()]
    logger.info('    Done ! ')

    logger.info('    Gestion des valeures nulles')
    # On complète les NaN avec du vide
    df_prep['ETAT_PATIENT'] = df_prep['ETAT_PATIENT'].fillna("")
    df_prep['DESCRIPTION_INCIDENT'] = df_prep['DESCRIPTION_INCIDENT'].fillna(
        "")
    df_prep['ACTION_PATIENT'] = df_prep['ACTION_PATIENT'].fillna("")
    df_prep['FABRICANT'] = df['FABRICANT'].fillna("")
    df_prep['TYPE_VIGILANCE'] = df_prep['TYPE_VIGILANCE'].fillna("")
    df_prep['CLASSIFICATION'] = df_prep['CLASSIFICATION'].fillna('')
    logger.info('    Done ! ')

    # On nettoie les données :
    if clean == True:
        logger.info('    Nettoyage des données...')
        for col in ['DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'ACTION_PATIENT']:
            df_prep[col] = df_prep[col].map(
                lambda x: clean_text.preprocess_text(x))
        logger.info('    Done ! ')

    # On encode les labels
    df_prep.GRAVITE = df_prep.GRAVITE.map(lambda x: GRAVITE_ENC(x))

    return (df_prep)


def train_GRAVITE(df_train: pd.DataFrame, cv=3, save=False) :
    """Fonction qui permet d'entraîner le modèle d'inference de la GRAVITE, celui ci est un pipeline réalisant une vectorisation  utilisant le
    tfidf finetuné avec la librairie Optuna puis utilise un  SVM linéaire  combiné avec un Calibrated classifier qui permet
    de probabiliser la classification  des Gravités. 
    
    Args:
        df_train (pd.DataFrame): dataframe avec les données complètes et nettoyées
        cv (int, optional): nombre de crossvalidation dans le calibratedClassifier pour probabiliser le SVM  (Ne pas dépasser 6, pour que le temps d'execution reste raisonable). Defaults to 3.
        save (bool, optional): permet de sauvegarder le pipeline entrainé si True. Defaults to False.

    Returns:
        pipeline (sk.pipeline.Pipeline):  le pipeline entrainé sur les données d'entrées pour inférer la Gravité à 5 classes. 
    """

    preprocess = ColumnTransformer(
        [('etat_pat_tfidf_tfidf', TfidfVectorizer(sublinear_tf=True,
                                                  analyzer='word',
                                                  min_df=1,
                                                  ngram_range=(1, 1),
                                                  stop_words=STOP_WORDS,
                                                  max_features=5057,
                                                  norm='l2'), 'ETAT_PATIENT'),

         ('action_pat_tfidf_tfidf', TfidfVectorizer(sublinear_tf=True,
                                                    analyzer='word',
                                                    min_df=2,
                                                    ngram_range=(1, 1),
                                                    stop_words=STOP_WORDS,
                                                    max_features=9783,
                                                    norm='l2'), 'ACTION_PATIENT'),


            ('description_tfidf', TfidfVectorizer(sublinear_tf=True,
                                                  min_df=4,
                                                  ngram_range=(1, 1),
                                                  stop_words=STOP_WORDS,
                                                  max_features=7057,
                                                  norm='l1'), 'DESCRIPTION_INCIDENT'),

            ('fabricant_tfidf', TfidfVectorizer(sublinear_tf=True,
                                                analyzer='char_wb',
                                                min_df=3,
                                                ngram_range=(1, 1),
                                                stop_words=STOP_WORDS,
                                                max_features=2385,
                                                norm='l1'), 'FABRICANT'),
            ('classification_enc', TfidfVectorizer(sublinear_tf=True,
                                                   analyzer='word',
                                                   min_df=5,
                                                   ngram_range=(1, 1),
                                                   stop_words=STOP_WORDS,
                                                   max_features=4766,
                                                   norm='l2'), 'CLASSIFICATION')
         ],

        remainder='passthrough')

    pipeline = Pipeline([
        ('vect', preprocess),
        ('clf', CalibratedClassifierCV(
            LinearSVC(class_weight='balanced'), cv=cv, method='isotonic'))
    ])

    X = df_train[['DESCRIPTION_INCIDENT', 'ETAT_PATIENT',
                  'ACTION_PATIENT', 'FABRICANT', 'CLASSIFICATION']]
    y = df_train.GRAVITE

    pipeline.fit(X, y)

    if save == True:
        joblib.dump(pipeline, os.path.join(MODEL_PATH,'Gravite_bin_model_last.sav'))

    return pipeline


def evaluate_model(pipeline: sk.pipeline.Pipeline, df_test: pd.DataFrame) -> (float):
    """
    Renvoie le score adéquate pour le pipeline chargé. Dans le cas de la gravité binaire, la f1 score de la classe critique 
    est le score adéquate pour notre problème
    Entrées:
    
    Args:
        pipeline (sk.pipeline.Pipeline): le pipeline entrainé sur les données de la GRAVITE
        df_test (pd.DataFrame): les données de test
        
    Returns:
        score (float) : le f1 score de la classe critique pour modèle
    """

    X_test = df_test[['DESCRIPTION_INCIDENT', 'ETAT_PATIENT',
                      'ACTION_PATIENT', 'FABRICANT', 'CLASSIFICATION']]
    y_test = df_test.GRAVITE

    y_pred = pipeline.predict(X_test)

    logger.info("Justesse"+str(accuracy_score(y_test, y_pred)))
    logger.info("Balanced_accuracy : "+str(balanced_accuracy_score(y_test, y_pred)))
    logger.info("f1-weighted : "+str( f1_score(y_test, y_pred, average='weighted')))
    logger.info("f1-Binaire : "+str( f1_score(y_test, y_pred, average='binary', pos_label=0)))
    return(f1_score(y_test, y_pred, average='binary', pos_label=0))


def repro_result(df: pd.DataFrame, n=3, citoyen=True, pro=True, clean=False) -> (float, float, float):
    """" 
    Fonction qui permet de reproduire les résultats annoncés. Les résultats sont stockés dans le fichier excel performance.xlsx
    afin de garder une trace au fur et à mesure des entraînements.

    Args:
        df (pd.DataFrame): 
        n (int, optional): Le nombre de cross validation pour train_GRAVITE. Defaults to 3
        citoyen (bool, optional): le score concernant les citoyens doit t-il être calculé ?. Defaults to True
        pro (bool, optional): e score concernant les professionnels doit t-il être calculé. Defaults to True
        clean (bool, optional): faut-il netoyer les données textuelles avant de calculer les score ?. Defaults to False

    Returns:
        score (float): score globale pour les l'enseble du Data set
        score_citoyens (float): score calculé pour un modèle entrainé seulement sur les données citoyennes
        score_pro (float) :  score calculé pour un modèle entrainé seulement sur les données professionnelles
    """

    df_n = df.copy()

    # Selection des index citoyen et pro
    citoyen_index = df_n[df_n.TYPE_DECLARANT == 'Citoyen'].index
    professionel_index = df_n[df_n.TYPE_DECLARANT ==
                              'Etablissements et professionnels de santé'].index

    logger.info('    Entrainement du modèle sur un sous echantillon puis test.')
    # Entrainement du modèle de base
    train_index, test_index = next(GroupShuffleSplit(
        random_state=1029, test_size=0.2).split(df_n, groups=df_n['DESCRIPTION_INCIDENT']))
    df_train, df_test = df_n.iloc[train_index], df_n.iloc[test_index]
    score = evaluate_model(train_GRAVITE(df_train, cv=n), df_test)
    logger.info("    Done ! ")

    score_citoyens = 0
    score_pro = 0
    logger.info('    Entrainement du modèle sur les déclarations de citoyens puis test.')
    if citoyen == True:
        df_c = df_n.loc[citoyen_index]
        # On selection les variables de test en faisant attention aux doublons
        train_index, test_index = next(GroupShuffleSplit(
            random_state=1029, test_size=0.2).split(df_c, groups=df_c['DESCRIPTION_INCIDENT']))
        df_train, df_test = df_c.iloc[train_index], df_c.iloc[test_index]
        score_citoyens = evaluate_model(train_GRAVITE(df_train, cv=n), df_test)
    logger.info('    Done!')

    logger.info('    Entrainement du modèle sur les déclarations de professionels puis test.')
    if pro == True:
        df_p = df_n.loc[professionel_index]
        # On selection les variables de test en faisant attention aux doublons
        train_index, test_index = next(GroupShuffleSplit(
            random_state=1029, test_size=0.2).split(df_p, groups=df_p['DESCRIPTION_INCIDENT']))
        df_train, df_test = df_p.iloc[train_index], df_p.iloc[test_index]
        score_pro = evaluate_model(train_GRAVITE(df_train, cv=n), df_test)
    logger.info('    Done!')
    return (score, score_citoyens, score_pro)
