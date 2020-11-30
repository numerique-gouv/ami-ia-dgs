""" 
Auteur: QUillivic Robin, rquillivic@starclay.fr

Description: Script qui permet à jour les modèles de regroupement à partir de nouvelle données au format MRveille

"""
import sys
import os 

import numpy as np
import pandas as pd 
import yaml

path_to_regroupement = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(1,path_to_regroupement)

import train_topic, train_cluster
from data_preparation.prepare_data import *
from inference.inference import add_cols
import joblib
import logging


with open(os.path.join(path_to_regroupement, 'config.yaml'), 'r') as stream:
        globale_config = yaml.load(stream, Loader=yaml.FullLoader)

with open(os.path.join(path_to_regroupement,'training','logging_config.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)

logging.config.dictConfig(log_config)


logger = logging.getLogger('entrainement')

path_mrv = globale_config['data']['mrv']['path']
data = pd.read_csv(path_mrv)


new_data = data.iloc[:10]
name = '25_11_2020_test'
delta = 0.2


if __name__=='__main__':

    # Chargement des anciens modèles de topics models et de clustering
    config_path = os.path.join(os.path.join(path_to_regroupement, 'training'), 'training_config.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    SAVE_PATH = config['path_to_save']

    
    model_config_path = os.path.join(SAVE_PATH, name, 'training_config.yaml')
    with open(model_config_path, 'r') as stream:
        model_config = yaml.load(stream, Loader=yaml.FullLoader)
    
    # Loading
    topicmodel = train_topic.TopicModel(name, model_config['topic'],save_dir=os.path.join(SAVE_PATH, name))
    topicmodel.load(name)

    clustermodel = train_cluster.ClusterModel(name, model_config['cluster'],save_dir=os.path.join(SAVE_PATH, name))
    clustermodel.topicmodel = topicmodel
    clustermodel.load(name)
    logger.info('Chargement des modèles effectué...')

    # Chargement des nouvelles  données
    new_data = new_data
    logger.info('Chargement des nouvelles données effectué...')
    # Mise en forme de la nouvelle donnée
    X_trans = prepare_data(new_data, save_dir=None, use_multiprocessing=False)
    logger.info('Mise en forme des nouvelles données effectuée...')
    # Mise à jour du topic modèle
    topicmodel.update(X_trans)
    logger.info('Mise à jour du topic model effectuée...')

    # Mise à jour des représenations thématiques des anciennes données
    topicmodel.build_doc_topic_mat(save=False,data=pd.read_pickle(os.path.join(model_config['data']['path'],model_config['data']['filename'])))
    logger.info('Mise à jour de la matrice topic document  effectuée...')
    
    
    # Mise à jour du modèles du modèle de clustering avec les nouvelles représentations
    # Il suffit de mettre à jour les centres des clusters
    clustermodel.build_cluster_centers()
    logger.info('Mise à jour des centroids du modèle de clustering effectuée...')

    # Création des données d'entrée du clustermodel
    X = topicmodel.predict(X_trans)
    cols = model_config['cluster']['model']['add_columns']
    X_new = X.iloc[:, :topicmodel.model.num_topics-1].values
    if len(cols):
        X_new = add_cols(new_data, cols, X_new, clustermodel)
    logger.info("Mise en forme des données pour l'algo de clusterisation effectuée...")

    # Incorporation des nouvelles données au modèle de clustering
    logger.info('Mise à jour du modèle de clusterisation en cours...')
    clustermodel.update(X_new,delta = delta)
    logger.info('Mise à jour du modèle de clusterisation effectuée.')

    logger.info('sauvegarde des modèle en cours...')
    #sauvegarde dans un nouveau dossier
    new_name = name+'_update'
    new_topicmodel = train_topic.TopicModel(new_name, model_config['topic'],save_dir=os.path.join(SAVE_PATH, new_name))
    new_topicmodel.model = topicmodel.model
    new_topicmodel.corpus = topicmodel.corpus
    new_topicmodel.dictionary = topicmodel.dictionary
    #new_topicmodel.build_viz()
    new_topicmodel.doc_topic_mat = topicmodel.doc_topic_mat
    new_topicmodel.global_save()

    new_clustermodel = train_cluster.ClusterModel(new_name, model_config['cluster'],save_dir=os.path.join(SAVE_PATH, new_name))
    new_clustermodel.model = clustermodel.model
    new_clustermodel.model.features = clustermodel.model.features
    X_training = new_clustermodel.model.features.iloc[:,:-1].values #tout sauf la colonne label
    clustermodel.compute_score(X_training,save=True)
    clustermodel.compute_evaluation_score(data,save=True)
    clustermodel.predict_mrv(X_training, data,save=True)
    clustermodel.save(X_training)
    logger.info(f'sauvegarde des modèle effectuée dans {os.path.join(SAVE_PATH, new_name)}')
    


    