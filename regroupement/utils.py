"""
Auteur: Quillivic Robin, Data scientist chez Starclay, rquillivic@straclay.fr 
"""


import logging
import os
import ast
import yaml
import joblib
import pandas as pd

import sys
from scipy.sparse import  hstack
import numpy as np


sys.path.insert(0,os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from data_preparation.prepare_data import clean_fabricant



def loading_function(path, method,obj,logger):
    """Fonction générique pour charger un objet en Python

    Args:
        path (str): emplacement du fichier
        method (function): méthode pour charger un l'objet obj
        obj (): objet à charger
        logger (logging.logger): logger associé 

    Raises:
        ValueError: [description]
        ValueError: [description]
    """
    if type(path)==str:
        if not os.path.exists(path):
            logger.error(f'path {path} does not exists')
            raise ValueError('file does not exist')
        else : 
            logger.info(f'path {path} already exists')
    try :
        obj = method(path)
        logger.info(f'loaded object from {path}')
        return obj
        
    except Exception as e : 
        logger.error(f'Error loading {path}: {e}')
        raise ValueError(f'{path} : {e}') 


def add_col_training_cat_one_hot(df,col,X,clustermodel,save=False,svd=False,k=2):
    """Permet d'ajouter des colonnes en plus de la représentation topic

    Args:
        df (pd.DataFrame): dataframe de la base de donnée MRveille
        col (list): liste des colonnes à ajouter
        X (array): représentation thèmatique

    Returns:
        X_new (array): représentation thèmatique complétée des colonnes présentes dans col
    """

    from  sklearn.preprocessing import LabelBinarizer,OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD

    # Construction du pipeline de transformation des données et netoyage des donnée entrantes
    transformer_data = []
    for c in col :
        df[c] = df[c].map(str)
        if c == "FABRICANT":
            df[c] = df[c].map(lambda x : clean_fabricant(x))
        etape = (c, OneHotEncoder(handle_unknown='ignore'),[c])
        #etape = (c, LabelBinarizer(), [c])
        transformer_data.append(etape)

    preprocess = ColumnTransformer(transformer_data, remainder='passthrough')
    if svd :
        pipe = [('vect',preprocess), ('svd',TruncatedSVD(n_components=300))]
    else :
        pipe = [('vect',preprocess)]

    pipeline = Pipeline(pipe)
    # transformation des données
    adding_data = df[col].fillna(' ')
    adding_data_transformed = pipeline.fit_transform(adding_data)

    if save :
            joblib.dump(pipeline, os.path.join(clustermodel.save_dir,'pipeline_add_col'+'.sav'))

    # Utilisation de hstack car adding_data_transformed est une matrice sparce.
    try :
        X_new = hstack((X,k*adding_data_transformed))
    except :
         X_new = np.concatenate((X,k*adding_data_transformed),axis=1)

    print(X_new.shape)
    print(type(X_new))

    return X_new


def add_col_training_cat(df, col, X, clustermodel, save=False, svd=False):
    """Permet d'ajouter des colonnes en plus de la représentation topic

    Args:
        df (pd.DataFrame): dataframe de la base de donnée MRveille
        col (list): liste des colonnes à ajouter
        X (array): représentation thèmatique

    Returns:
        X_new (array): représentation thèmatique complétée des colonnes présentes dans col
    """

    from sklearn.preprocessing import LabelEncoder

    # Construction du pipeline de transformation des données et netoyage des donnée entrantes
    transformer_data = []
    df_used = pd.DataFrame()
    for c in col:
        df[c] = df[c].map(str)
        if c == "FABRICANT":
            df[c] = df[c].map(lambda x: clean_fabricant(x))
        le = LabelEncoder()
        le.fit(df[c].map(str).fillna(' ').values)
        # for retraining
        if hasattr(clustermodel, 'inference_pipeline'):
            le_previous_dict = clustermodel.inference_pipeline[str(c)]
            le_new_keys = list(set(le.classes_) - set(le_previous_dict.keys()))
            le.classes_ = list(le_previous_dict.keys()) + le_new_keys
        df_used[c] = le.transform(df[c].map(str).fillna(' ').values)
        transformer_data.append((c, le, [c]))
    if save:
        joblib.dump(transformer_data, os.path.join(clustermodel.save_dir, 'pipeline_add_col' + '.sav'))

    # Utilisation de hstack car adding_data_transformed est une matrice sparce.
    X_new = np.concatenate((X, df_used), axis=1)

    print(X_new.shape)
    print(type(X_new))

    return X_new


def load_inference_pipeline(clustermodel):
    """
    Charge l'ensemble des transformers pour l'inference de clustering et les stocke sous forme de pipeline
    dans clustermodel.inference_pipeline

    :param clustermodel: ClusterModel chargé
    :return: ClusterModel
    """
    # chargement du pipeline
    file_path = os.path.join(clustermodel.save_dir, 'pipeline_add_col' + '.sav')
    try:
        pipeline = joblib.load(file_path)
    except Exception as e:
        raise Exception(
            f'Le pipeline {file_path} de transformation des données pour ajouter des colonnes est in trouvables')

    if isinstance(pipeline, list):
        loaded_pipeline = {}
        for label_encoder in pipeline:
            le = label_encoder[1]
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            loaded_pipeline[str(label_encoder[0])] = le_dict
        pipeline = loaded_pipeline

    clustermodel.inference_pipeline = pipeline
    return clustermodel


def add_col_inference_cat_one_hot(df, col, X, clustermodel, log_not_found=False,k=2):
    """
    Permet d'ajouter des colonnes en plus de la représentation topic;

    Args:
        df (pd.DataFrame): dataframe de la base de donnée MRveille
        cols (list): liste des colonnes à ajouter
        X (array): représentation thèmatique
        clustermodel (ClusterModel): modèle de clustering utilisé
        log_not_found: if true, prints messages when some categorical variables have not been mapped by the corresponding one-hot encoder

    Returns:
        X (array): X + colonnes ajoutées. représentation thèmatique complétée des colonnes présentes dans col
    """
    if not hasattr(clustermodel, 'inference_pipeline'):
        raise RuntimeError('You must load clustermodel inference pipeline via load_inference_pipeline')
    # modification des données:
    for c in col:
        df[c] = df[c].map(str)
        if c == "FABRICANT":
            df[c] = df[c].map(lambda x : clean_fabricant(x))

    # Vectorisation des données
    data_to_vect = df[col].fillna(" ")
    vectorisez_data = clustermodel.inference_pipeline.transform(data_to_vect)
    if log_not_found:
        start_ind = 0
        for i, transfo in enumerate(clustermodel.inference_pipeline.steps[0][1].transformers_):
            end_ind = start_ind + (len(transfo[1].categories_[0]) - 1)
            if vectorisez_data[:,start_ind:end_ind+1].sum() != vectorisez_data.shape[0]:
                print(f'{transfo[0]} not found')
            start_ind = end_ind + 1
    try :
        X_new = hstack((X,k*vectorisez_data))
        X_new = X_new.toarray()
    except :
        X_new = np.concatenate((X,k*vectorisez_data),axis=1)

    return X_new


def add_col_inference_cat(df, col, X, clustermodel):
    """
    Permet d'ajouter des colonnes en plus de la représentation topic;

    Args:
        df (pd.DataFrame): dataframe de la base de donnée MRveille
        cols (list): liste des colonnes à ajouter
        X (array): représentation thèmatique
        clustermodel (ClusterModel): modèle de clustering utilisé
        log_not_found: if true, prints messages when some categorical variables have not been mapped by the corresponding one-hot encoder

    Returns:
        X (array): X + colonnes ajoutées. représentation thèmatique complétée des colonnes présentes dans col
    """
    if not hasattr(clustermodel, 'inference_pipeline'):
        raise RuntimeError('You must load clustermodel inference pipeline via load_inference_pipeline')
    # modification des données:
    df_used = pd.DataFrame()
    for c in col:
        try:
            df[c] = df[c].map(str)
            if c == "FABRICANT":
                df[c] = df[c].map(lambda x: clean_fabricant(x))
            le_dict = clustermodel.inference_pipeline[str(c)]
            df_used[c] = df[c].apply(lambda x: le_dict.get(str(x), -1))
        except Exception as e:
            raise RuntimeError(f'Error while adding col to data : {e}')
    X = np.concatenate((X, df_used.values), axis=1)
    return X
