"""
Auteur:  
    Quillivic Robin, Data Scientist chez StarClay, rquilivic@starclay.fr
    
Description:  
    Ce fichier contient les fonctions nécessaire pour entraîner, tester le modèle pour inférer la TYPOLOGIE. La Typologie se compose d'un triplet : 
    - Type de dysfonctionnement
    - Conséquence du dysfonctionnement
    - Type d'effet
    
    Nous avons construit 3 modèles quasi-identique pour inférer ce triplet car les trois problèmes de classifications sont très proches.
    
    Ce fichier fait partie du Livrable 2 et son architecture est de la même forme que les 5 autres fichiers de classification
    Il y a 5 fonctions principales : 
    - create_multilabel_data et preprare_data qui permettent de préparer les données pour l'entrainement (gestion du multilabel,nettoyage, encodage etc.)
    - train: Entrainement du pipeline tfidf+svd+bi-LSTM pour inférer la Typologie
    - evaluate_model: évaluation du modèle par le f1-sample qui est une mesure adaptée en cas de problème multilabel
    - repro_result : entraine et test le modèle sur différents jeu de données ( l'ensemble des données, les données des citoyens et des professionnels)
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import clean_text
import sklearn as sk



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV

from clean_text import STOP_WORDS

import joblib
import keras

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score


from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Bidirectional, SimpleRNN, Input, concatenate, Reshape
import tensorflow
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from keras.layers import Concatenate, GlobalMaxPool1D, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score


from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import logging
import logging.config
import yaml

with open(os.path.join(os.path.dirname(__file__), '../log_conf.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)

logger = logging.getLogger('inference')

tensorflow.random.set_seed(1234)
tensorflow.keras.backend.set_floatx('float64')

# Path config
from datetime import date
today = date.today()
d4 = today.strftime("%b-%d-%Y")
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)
MODEL_PATH = os.path.join(config_data['training']['save_path'], 'model_'+str(d4))
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

#

def create_multilabel_data(df: pd.DataFrame):
    """Fonction qui prépare le jeu de données multilabel pour l'entrainement des modèles de Typologie. Cette fonction ne renvoie rien. 
    Elle créer un fichier pkl avec les données et elle permet de : 
        - créer un dossier model sous la racine
        - gérer les NaN pour les colonnes
        - Remplir avec le code correspondant les codes de la typologie
        - Encoder la classification de l'incident et de sauvegarder l'encodeur  dans le dossier model
        - regrouper les déclaration par texte identique pour créer les données multilabel
        - Sauvegarder la base de donnée nettoyée: multilabel_data.pkl

    Args:
        df (pd.DataFrame): Base de donnée MRveil

    Returns:
        None
    """
    if not os.path.exists('./models'):
        os.makedirs('./models')

    logger.info("    Gestion des NaN")
    # 1) On gère nan pour les colonnes contenant du text
    text_columns = ["DESCRIPTION_INCIDENT", "ETAT_PATIENT",
                    "FABRICANT", 'ACTION_PATIENT', 'LIBELLE_COMMERCIAL']
    df[text_columns] = df[text_columns].fillna('NON RENSEIGNE')

    # 2) on gère les NaN pour chaque Colonnes
    df['NUMERO_DECLARATION'] = df['NUMERO_DECLARATION'].fillna('NON RENSEIGNE')
    df['DESCRIPTION_INCIDENT'] = df['DESCRIPTION_INCIDENT'].fillna(
        'NON RENSEIGNE')
    df['ETAT_PATIENT'] = df['ETAT_PATIENT'].fillna('NON RENSEIGNE')
    df['CLASSIFICATION'] = df['CLASSIFICATION'].fillna('NON RENSEIGNE')
    df['FABRICANT'] = df['FABRICANT'].fillna('NON RENSEIGNE')
    df['LIBELLE_COMMERCIAL'] = df['LIBELLE_COMMERCIAL'].fillna('NON RENSEIGNE')
    df['REFERENCE_COMMERCIALE'] = df['REFERENCE_COMMERCIALE'].fillna(
        'NON RENSEIGNE')

    df['text'] = df['DESCRIPTION_INCIDENT']+'. '+df['ETAT_PATIENT']

    # Effet
    df['TEF_ID'] = df['TEF_ID'].fillna('E1213')
    # DYSFONCTIONNEMENT
    df['TDY_ID'] = df['TDY_ID'].fillna("D0")
    # CONSÉQUENCES
    df['CDY_ID'] = df['CDY_ID'].fillna("C0")

    logger.info('    Done ! ')

    le = LabelEncoder()
    df["CLASSIFICATION"] = le.fit_transform(df["CLASSIFICATION"])
    joblib.dump(le, os.path.join(MODEL_PATH, 'classification_Encoder.sav'))

    logger.info('    Création du jeu de données Multilabel et sauvegarde')
    # La typologie est en générale multilabel
    df_id = df.groupby('text').agg({'TEF_ID': lambda x: list(set(x)),
                                    'CDY_ID': lambda x: list(set(x)),
                                    'TDY_ID': lambda x: list(set(x))
                                    })

    # On fusionne les multilabels et on supprime les doublons
    df = df.drop(['TEF_ID', 'CDY_ID', 'TDY_ID'], axis=1)
    df = pd.merge(df, df_id, on='text')
    df = df.drop_duplicates('text')
    df = df.reset_index(drop=True)

    # On remplace les strings vide
    df.replace('', 'NON RENSEIGNE', inplace=True)
    # Sauvegarde des données nettoyées
    df.to_pickle('./multilabel_data.pkl')
    logger.info('    Done ! Sauvegardés dans multilabel.pkl')

    return ()


def prepare_data(mrv, typo:str, n:int=1000, split=True)-> (np.array, np.array) :
    """Fonction qui permet de transformer les données multilabel en des données utilisable pour feed un réseau de neurone.
    Elle permet : 
     - appliquer une vectorisation tf-idf sur les colonnes description, etat_patient, action_patient et fabricant
     - appliquer une svd à n composantes
     - effectuer une stratified split pour créer un jeu de train et de test si necessaire
     - effectuer un reshape pour correspondre à l'entrée d'un STM

    Args:
        mrv (pd.DataFrame): Jeu de données multilabel
        typo (str): 3 choix possibles TEF_ID, TDY_ID et CDY_ID pour faire le split stratifié en fonction des labels
        n (int, optional): nombre de composante à la svd. Defaults to 1000.
        split (bool, optional): Faut -il créer un jeu de train et de test ?. Defaults to True.

    Returns:
        X_train,y_train (np.array,np.array): données et label d'entrainement
        Si split==True : 
        X_train,y_train,X_test,y_test (np.array,np.array,np.array,np.array): données et label d'entrainement
        
    """
    preprocess = ColumnTransformer(
        [('description_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=3,
                                               ngram_range=(1, 1),
                                               stop_words=STOP_WORDS,
                                               max_features=10000, norm='l2'), 'DESCRIPTION_INCIDENT'),

         ('etat_pat_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 1),
                                            stop_words=STOP_WORDS,
                                            max_features=10000, norm='l2'), 'ETAT_PATIENT'),
            ('action_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=3,
                                             ngram_range=(1, 1),
                                             stop_words=STOP_WORDS,
                                             max_features=5000, norm='l2'), 'ACTION_PATIENT'),


            ('fabricant_tfidf', TfidfVectorizer(sublinear_tf=True, min_df=3,
                                                ngram_range=(1, 1),
                                                stop_words=STOP_WORDS,
                                                max_features=5000, norm='l2'), 'FABRICANT')
         ],

        remainder='passthrough')

    pipeline = Pipeline([
        ('vect', preprocess),
        ('svd', TruncatedSVD(n_components=n)),
    ])

    mlb = MultiLabelBinarizer()

    if split:
        logger.info("    Séparation du jeu de données multilabel en train et test")
        msss = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=1029)

        for train_index, test_index in msss.split(mrv['text'], mlb.fit_transform(mrv[typo])):
            train, test = mrv.loc[train_index],  mrv.loc[test_index]
        logger.info("    Done ! ")

        logger.info('    Application du pipeline de transformation ')
        X_train = train[['FABRICANT', 'CLASSIFICATION',
                         'DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'ACTION_PATIENT']]
        y_train = mlb.fit_transform(train[typo])
        X_test = test[['FABRICANT', 'CLASSIFICATION',
                       'DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'ACTION_PATIENT']]
        y_test = mlb.transform(test[typo])

        X_train_, X_test_ = pipeline.fit_transform(
            X_train), pipeline.transform(X_test)
        logger.info('    Done ! ')

        logger.info('    Sauvegarde du pipeline')
        joblib.dump(pipeline,os.path.join(MODEL_PATH, str(typo)+'_pipeline.sav'))
        logger.info('     Done ! ')

        X_train_ = np.reshape(
            X_train_, (X_train_.shape[0], 1, X_train_.shape[1]))
        X_test_ = np.reshape(X_test_, (X_test_.shape[0], 1, X_test_.shape[1]))

        return (X_train_, y_train, X_test_, y_test)

    else:
        X_train = mrv[['FABRICANT', 'CLASSIFICATION',
                       'DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'ACTION_PATIENT']]
        y_train = mlb.fit_transform(mrv[typo])

        logger.info('    Sauvegarde des encodeurs')
        joblib.dump(mlb, os.path.join(MODEL_PATH, typo+'_Encoder.sav'))
        logger.info('    Done ! ')

        logger.info('    Sauvegarde du pipeline')
        X_train_ = pipeline.fit_transform(X_train)
        joblib.dump(pipeline,os.path.join(MODEL_PATH, str(typo)+'_pipeline.sav'))
        logger.info('   Done ! ')

        X_train_ = np.reshape(
            X_train_, (X_train_.shape[0], 1, X_train_.shape[1]))

        return(X_train_, y_train)

def compute_class_weight(df: pd.DataFrame, typo='TEF_ID'):
    """
    Permet de construire le pods des classes pour une typo donnée à partir des occurences dans la base Mrveille df
    

    Args:
        df (pd.DataFrame): Base de donnée MRveille
       typo (str,optional): 3 valeures possibles : 
         - 'TEF_ID', pour entrainer un modèle sur les effets
         - 'CDY_ID', pour entrainer un modèle sur les conséquences du dysfonctionnement
         - 'TDY_ID', pour entrainer un modèle sur les type de dysfonctionnement
        Defaults to 'TEF_ID'
    Returns:
        weights (dict): numéro de la classe et son poids associé
    """
    from sklearn.utils import class_weight
    y = df[typo].values
    f = lambda x : x**(1/1.5)
    w = class_weight.compute_class_weight('balanced',np.unique(y),y)
    u = f(w)
    weights = dict(zip(np.arange(0,len(np.unique(y))),u))
    return weights


def train(X_train_: np.array, y_train: np.array, typo: str, save=False,weight=None):
    """
    Fonction qui permet d'entrainer le modèle bi-LSTM avec Keras pour chaque élement de la typologie. 
    Elle renvoie en sortie le modèle entrainé.
    

    Args:
        X_train_ (np.array): matrice d'entrainement résultant d'une SVD sur le TF-IDF
        y_train (np.array): label d'entrainement
        typo (str): 3 valeurs possibles : 
         - 'TEF_ID', pour entrainer un modèle sur les effets
         - 'CDY_ID', pour entrainer un modèle sur les conséquences du dysfonctionnement
         - 'TDY_ID', pour entrainer un modèle sur les type de dysfonctionnement
        save (bool, optional): Enregistre ou non le modèle enrainé. Defaults to False.
        weight (dict,optional): poid des classes pour l'entrainement Defauls to None
    
    Returns: 
        model (keras.model.sequential): modèle entrainé sur les données rentrées. 
    """
     
    
    model = Sequential()
    model.add(Bidirectional(LSTM(200, dropout=0.2)))
    model.add(Dense(y_train.shape[1], activation='softmax', dtype='float64'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])

    epochs = 5
    batch_size = 32

    history = model.fit(X_train_, y_train, epochs=epochs,class_weight=weight,
                        batch_size=batch_size, validation_split=0.2)

    return model


def evaluate_model(model, X_test: np.array, y_test: np.array, typo: str) -> (float):
    """ Renvoie le score adéquate pour le pipeline chargé. Dans le cas de la gravité, 
    le f1 sample est le score adéquate pour notre problème multiclass multilabel.

    Args:
        model (keras.model.Sequential): Modèles entrainé sur les données d'entrainement de la base MRveil
        X_test (np.array): Données de test
        y_test (np.array): labels associées aux données de test
        typo (str): 3 valeurs possibles : 
         - 'TEF_ID', pour entrainer un modèle sur les effets
         - 'CDY_ID', pour entrainer un modèle sur les conséquences du dysfonctionnement
         - 'TDY_ID', pour entrainer un modèle sur les type de dysfonctionnement
    
    Returns: 
        f1 (float): score f1 sample du modèle présicé en entrée. 
    """

    pred = model.predict(X_test)
    if typo == 'TEF_ID':
        val = 0.12
    if typo == 'CDY_ID':
        val = 0.2
    if typo == 'TDY_ID':
        val = 0.12

    pred[pred >= val] = 1
    pred[pred < val] = 0

    precision = precision_score(y_test, pred, average='samples')
    recall = recall_score(y_test, pred, average='samples')
    f1 = f1_score(y_test, pred, average='samples')
    logger.info(
        "    Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

    return(f1)

def repro_result(typo: str, citoyen=True, pro=True,weight=None) -> (float, float, float):
    """
    Fonction qui permet de reproduire les résultats annoncés. Les résultats sont stockés dans le fichier excel performance.xlsx
    afin de garder une trace au fur et à mesure des entrainements.

    Args:
        typo (str): 3 valeurs possibles : 
         - 'TEF_ID', pour entrainer un modèle sur les effets
         - 'CDY_ID', pour entrainer un modèle sur les conséquences du dysfonctionnement
         - 'TDY_ID', pour entrainer un modèle sur les type de dysfonctionnement
        citoyen (bool, optional): le score concernant les citoyens doit t-il être calculé ?. Defaults to True
        pro (bool, optional): e score concernant les professionnels doit t-il être calculé. Defaults to True
        clean (bool, optional): faut-il netoyer les données textuelles avant de calculer les score ?. Defaults to False
        weight (dict, optional): Le poid des class pour l'entrainement. Defaults to None

    Returns:
        score (float): score globale pour les l'enseble du Data set
        score_citoyens (float): score calculé pour un modèle entrainé seulement sur les données citoyennes
        score_pro (float) :  score calculé pour un modèle entrainé seulement sur les données professionnelles
    """
    mrv = pd.read_pickle('./multilabel_data.pkl')
    # Selection des index citoyen et pro
    mrv_c = mrv[mrv.TYPE_DECLARANT == 'Citoyen']
    mrv_p = mrv[mrv.TYPE_DECLARANT ==
                'Etablissements et professionnels de santé']

    logger.info('    Calcul du score Général.')
    # Entrainement du modèle de base
    X_train_, y_train, X_test_, y_test = prepare_data(mrv, typo, n=1000)

    score = evaluate_model(
        train(X_train_, y_train, typo, save=False,weight=weight), X_test_, y_test, typo)
    logger.info("    Done ! ")

    score_citoyens = 0
    score_pro = 0
    logger.info('    Calcul du Score citoyen.')
    if citoyen == True:
        mrv_c = mrv_c.reset_index()
        X_train_, y_train, X_test_, y_test = prepare_data(mrv_c, typo, n=1000)
        score_citoyens = evaluate_model(
            train(X_train_, y_train, typo, save=False), X_test_, y_test, typo)

    logger.info('    Done!')

    logger.info('    Calcul du Score Professionel.')
    if pro == True:
        mrv_p = mrv_p.reset_index()
        X_train_, y_train, X_test_, y_test = prepare_data(mrv_p, typo, n=1000)
        score_pro = evaluate_model(
            train(X_train_, y_train, typo, save=False), X_test_, y_test, typo)
    logger.info('    Done!')
    return (score, score_citoyens, score_pro)
