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
import logging

from utils import add_col_training_cat, load_inference_pipeline

with open(os.path.join(path_to_regroupement, 'config.yaml'), 'r') as stream:
    globale_config = yaml.load(stream, Loader=yaml.FullLoader)

with open(os.path.join(path_to_regroupement,'training','logging_config.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)

logging.config.dictConfig(log_config)
logger = logging.getLogger('réentrainement')



if __name__=='__main__':

    ###############
    # Modèle entrainé
    ##############
    trained_name = '14_01_2021'
    trained_model_dir = os.path.join("/home/cyril/Work/Starclay/dgs-signalement-ia/clustering_results",
                                     trained_name)

    ################
    #   où sauvegarder
    ################
    retrain_name = '14_01_2021_retrain'
    SAVE_DIR = os.path.join("/home/cyril/Work/Starclay/dgs-signalement-ia/clustering_results",
                            retrain_name)

    new_mrv_data_file = '...'

    # seuil de création d'un nouveau cluster (si aucun cluster n'a un score supérieur à delta, on crée un nouveau cluster pour le document)
    # Cette valeur est très dépendante du modèle et de la façon de calculer les scores.
    # Sur le modèle actuel, le clustering d'un exemple déjà connu peut donner un score max d'appartenance de 0,003, pour un score moyen de 0,001
    # il faut à priori viser un delta de 0,002 (à affiner à partir d'exemples non connus...
    delta = 0.002

    os.makedirs(SAVE_DIR)

    # chargement config entrainée
    model_config_path = os.path.join(trained_model_dir, 'training_config.yaml')
    with open(model_config_path, 'r') as stream:
        model_config = yaml.load(stream, Loader=yaml.FullLoader)

    ####################
    # Chargement data
    ####################
    # chargement data entrainées
    mrv_data = pd.read_csv(model_config['data']['mrv'])
    prepd_mrv_data = pd.read_pickle(os.path.join(model_config['data']['path'],
                                                 model_config['data']['filename']))
    logger.info(f'Données précédentes : {len(mrv_data)} lignes...')

    # chargement des nouvelles données
    new_data = pd.read_csv(new_mrv_data_file)
    logger.info(f'Nouvelles données : {len(new_data)} lignes...')
    
    ################
    # Chargement des modèles
    ################
    topicmodel = train_topic.TopicModel(trained_name, model_config['topic'],
                                        save_dir=trained_model_dir)
    topicmodel.load(trained_name)
    topicmodel.set_data(prepd_mrv_data)

    if model_config['cluster']['model']['name'] != "kprototypes":
        raise RuntimeError('Ce script permet de réentrainer les kprototypes uniquement')

    columns = model_config['cluster']['model']['add_columns']
    categorical = list(range(topicmodel.model.num_topics - 1,
                             topicmodel.model.num_topics - 1 + len(columns)))
    clustermodel = train_cluster.ClusterModelKProto(trained_name, model_config['cluster'],
                                                    save_dir=trained_model_dir,
                                                    categorical_columns_ind=categorical)
    clustermodel.topicmodel = topicmodel
    clustermodel.load(trained_name)
    clustermodel = load_inference_pipeline(clustermodel)
    logger.info('Chargement des modèles effectué...')

    ###################
    # Préparation des nouvelles données
    ###################
    logger.info('Chargement des nouvelles données effectué...')
    # Mise en forme de la nouvelle donnée
    X_trans = prepare_data(new_data, save_dir=None, use_multiprocessing=False)
    logger.info('Mise en forme des nouvelles données effectuée...')

    ##################
    #  Mise à jour topics
    ##################
    topicmodel.update(X_trans)
    logger.info('Mise à jour du topic model effectuée...')

    # Mise à jour des représentations thématiques des anciennes données
    topicmodel.build_doc_topic_mat(save=False)
    logger.info('Mise à jour de la matrice topic document  effectuée...')

    # Mise à jour du modèles du modèle de clustering avec les nouvelles représentations
    # On met à jour les coordonnées des documents dans le modèle de cluster avec les nouvelles projections de topic
    # puis on recalcule les centres
    new_features_array = np.hstack((topicmodel.doc_topic_mat.iloc[:len(mrv_data), :topicmodel.model.num_topics - 1].values,
                                    clustermodel.model.features.values[:, categorical],
                                    np.expand_dims(clustermodel.model.labels_, axis=-1)))
    clustermodel.model.features = pd.DataFrame(data=new_features_array, columns=clustermodel.model.features.columns)
    clustermodel.model.features = clustermodel.model.features.astype({"label": int})
    clustermodel.build_cluster_centers(refresh_topic_coords=True)
    logger.info('Mise à jour des centroids du modèle de clustering effectuée...')

    # Création des données d'entrée du clustermodel

    columns = model_config['cluster']['model']['add_columns']
    X = topicmodel.predict(X_trans)
    X_new = X.iloc[:, :topicmodel.model.num_topics - 1].values
    clustermodel.save_dir = os.path.join(SAVE_DIR, 'cluster')
    os.makedirs(clustermodel.save_dir)
    X_new = add_col_training_cat(new_data, columns, X_new, clustermodel, save=True, svd=False)
    logger.info("Mise en forme des données pour l'algo de clusterisation effectuée...")

    # Incorporation des nouvelles données au modèle de clustering
    logger.info('Mise à jour du modèle de clusterisation en cours...')
    clustermodel.update(X_new, delta=delta)
    logger.info('Mise à jour du modèle de clusterisation effectuée.')

    #####################
    # Sauvegarde des divers éléments
    #####################
    logger.info('sauvegarde des données...')
    for col in prepd_mrv_data.columns:
        if col not in X_trans.columns:
            X_trans[col] = ''

    mrv_file = model_config['data']['mrv']
    ext = mrv_file.split('.')[-1]
    new_mrv_file = mrv_file[:mrv_file.rfind(ext) - 1] + '_retrain.' + ext
    complete_mrv = pd.concat((mrv_data, new_data), ignore_index=True)
    complete_mrv.to_csv(new_mrv_file)
    model_config['data']['mrv'] = new_mrv_file
    logger.info(f'Données MRV aggregées ({len(complete_mrv)} lignes) sauvegardées dans {new_mrv_file}')

    prev_prep_data_file = os.path.join(model_config['data']['path'], model_config['data']['filename'])
    ext = prev_prep_data_file.split('.')[-1]
    new_prep_data_file = prev_prep_data_file[:prev_prep_data_file.rfind(ext)-1] + '_retrain.' + ext
    complete_preped_data = pd.concat((prepd_mrv_data, X_trans))
    complete_preped_data.to_pickle(new_prep_data_file)
    model_config['data']['filename'] = os.path.basename(new_prep_data_file)
    logger.info(f'Données MRV aggregées préparées ({len(complete_preped_data)} lignes) sauvegardées dans {new_prep_data_file}')

    model_config['config_name'] = retrain_name
    with open(os.path.join(SAVE_DIR, 'training_config.yaml'), 'w') as f:
        yaml.dump(model_config, f)

    logger.info('calcul des scores et visualisation du modèle de topic en cours...')
    new_topicmodel = train_topic.TopicModel(retrain_name, model_config['topic'], save_dir=SAVE_DIR)
    new_topicmodel.model = topicmodel.model
    new_topicmodel.corpus = topicmodel.corpus
    new_topicmodel.dictionary = topicmodel.dictionary
    new_topicmodel.build_viz()
    new_topicmodel.doc_topic_mat = topicmodel.doc_topic_mat
    new_topicmodel.global_save()
    new_topicmodel.logger.info(f'Coherence score : {new_topicmodel.get_coherence_score(save=True)}')

    logger.info('calcul des scores et visualisation du modèle de clustering en cours...')
    columns = model_config['cluster']['model']['add_columns']
    categorical = list(range(topicmodel.model.num_topics - 1,
                             topicmodel.model.num_topics - 1 + len(columns)))
    new_clustermodel = train_cluster.ClusterModelKProto(retrain_name, model_config['cluster'],
                                                        save_dir=SAVE_DIR,
                                                        categorical_columns_ind=categorical)
    new_clustermodel.model = clustermodel.model
    new_clustermodel.model.features = clustermodel.model.features
    X_training = new_clustermodel.model.features.iloc[:,:-1].values #tout sauf la colonne label
    new_clustermodel.compute_score(X_training, save=True)
    new_clustermodel.compute_evaluation_score(complete_mrv, save=True)
    new_clustermodel.predict_mrv(X_training, complete_mrv, save=True)
    new_clustermodel.save(X_training)
    logger.info(f'sauvegarde des modèle effectuée dans {SAVE_DIR}')
    


    