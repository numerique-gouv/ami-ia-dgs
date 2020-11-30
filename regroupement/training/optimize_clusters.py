"""
Auteur: 
    Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
Description:
    Fichier permettant d'optimiser le nombre de cluster pour un cluster modèle
    Pour lancer l'optimisation, il faut executer la commande : python3 optimize_clusters.py
"""

import warnings

warnings.filterwarnings('ignore')

import os, sys
import yaml
import pandas as pd
import numpy as np

from train_topic import TopicModel
from train_cluster import ClusterModel

import optuna
from optuna import Trial

import shutil

current_dir = os.path.dirname(__file__)
config_file = 'training_config.yaml'

if len(sys.argv) == 2:
    config_file = sys.argv[1]

if not os.path.isabs(config_file):
    config_file = os.path.abspath(os.path.join(current_dir, config_file))


with open(config_file, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

save_dir_topic = config['path_to_save'] #'/home/robin/Nextcloud/strar_clay/GitLab/Annexe/L3'
name = '26_10_2020'  
topic_dir = os.path.join(save_dir_topic,name)
topic_config_file = os.path.join(topic_dir, 'training_config_serveur.yaml')

 
with open(topic_config_file, 'r') as stream:
    config_topic = yaml.load(stream, Loader=yaml.FullLoader)

   
# chargement des données sources

data = pd.read_csv(config['data']['mrv'])

def objective(trial: Trial,data=data):
    """Fonction objectif pour l'optimiseur Optuna

    Args:
        trial (Trial): Un essai de paramètre

    Returns:
        macro_score*micro_score
    """
    try_name = str(trial.number)
    save_dir = os.path.join(config['path_to_save'],config['config_name'], try_name)

    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(os.path.join(current_dir, save_dir))

    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_file, save_dir)

    

    filename = os.path.join(config['data']['path'], config['data']['filename'])
    if not os.path.isabs(filename):
        filename = os.path.abspath(os.path.join(current_dir, filename))
    
    # CHargement du topic model
    with open(topic_config_file, 'r') as stream:
        config_topic = yaml.load(stream, Loader=yaml.FullLoader)
    
    # Loading
    topicmodel = TopicModel(name, config_topic['topic'], 
                                        save_dir=topic_dir)
    topicmodel.load(name)
    
   
    
    def add_col(df,col,X):
        """Permet d'ajouter des colonnes en plus de la représentation topic

        Args:
            df (pd.DataFrame): dataframe de la base de donnée MRveille
            col (list): liste des colonnes à ajouter
            X (array): représentation thèmatique

        Returns:
            X_new (array): représentation thèmatique complétée des colonnes présentes dans col
        """
        df_used = pd.DataFrame()
        for c in col : 
            df[c] = df[c].astype('category')
            df_used[c] = df[c].cat.codes
            X_new = np.concatenate((X,df_used.values),axis=1)
        return X_new

    columns =  config['cluster']['model']['add_columns']
    current_config = config['cluster']
    current_config['model']['kmeans']['n_cluster'] = trial.suggest_int("n_cluster", 1000, 5000)


    clustermodel = ClusterModel(try_name, current_config, save_dir=save_dir)
    clustermodel.topicmodel = topicmodel
    n = topicmodel.model.num_topics
    n_lignes = 10000
    X = topicmodel.doc_topic_mat.iloc[:n_lignes, :n-1].values
    data = data.iloc[:n_lignes,:]
    #X = add_col(data,columns,X)
    
    
    clustermodel.train(X)
    clustermodel.compute_score(X,save=True)
    macro_score,micro_score = clustermodel.compute_evaluation_score(data,save=True)
    
    clustermodel.save()
    
    clustermodel._logger.info(f'Micro score : {micro_score}')
    clustermodel._logger.info(f'Macro score : {macro_score}')

    score = micro_score * macro_score

    trial.report(score,step = 10)
    return score


# Optimisation
studyName = 'cluster_study'
maximum_time = 30 * 60 * 60  # second
number_of_random_points = 50


optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
study = optuna.create_study(study_name=studyName, direction="maximize")
study.optimize(objective, n_trials=number_of_random_points, timeout=maximum_time)  

# Sauvegarde du resultat
df = study.trials_dataframe()
df.to_json(os.path.join(config['path_to_save'], config['config_name'], config['config_name'] +studyName + '.json'))
print(study.best_trial)