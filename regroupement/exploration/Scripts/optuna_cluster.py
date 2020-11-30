



"""
Auteur: Quillivic Robin,  
"""

import sys
import yaml
import os
import json
import numpy as np
import pandas as pd

import optuna
from optuna import Trial

SAVE_PATH =  "/home/robin/Nextcloud/strar_clay/GitLab/Annexe/L3/"

with open('/home/robin/Nextcloud/strar_clay/GitLab/signalement-ia/regroupement/analyse/analyse_config.yaml', 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)
    
with open(os.path.join('/home/robin/Nextcloud/strar_clay/GitLab/signalement-ia/regroupement', 'config.yaml'), 'r') as stream:
    globale_config = yaml.load(stream, Loader=yaml.FullLoader)
    

path_mrv = globale_config['data']['mrv']['path']
data_mrv = pd.read_csv(path_mrv)


path_to_regroupement = '/home/robin/Nextcloud/strar_clay/GitLab/signalement-ia/regroupement' #os.path.dirname(os.path.dirname('.'))
sys.path.insert(1,os.path.join(path_to_regroupement, 'training/' ))

config_path = os.path.join(os.path.join(path_to_regroupement, 'training'),'training_config.yaml')


with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

import train_topic, train_cluster
from sklearn.cluster import FeatureAgglomeration, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

import joblib



########## Chargement des données et du Topic Model ####################

name = config_data['analyse']['name']
# Loading
topicmodel = train_topic.TopicModel(name, config['topic'], 
                                        save_dir=os.path.join(SAVE_PATH, name))
topicmodel.load(name)

mat = topicmodel.doc_topic_mat
n_topic = topicmodel.model.num_topics
n_ligne = None
X = mat.iloc[:n_ligne, :n_topic-1].values
X_trans = StandardScaler().fit_transform(X)

########## Construction du pipeline d'optimisation Optuna ####################


def objective(trial:Trial):
    """Fonction objectif pour l'optimiseur Optuna

    """
    
    # Liste des paramètres à optimiser
    param = {
        "eps" : trial.suggest_uniform("eps", 3, 50),
        "min_samples" :trial.suggest_uniform("min_samples", 5, 50),
        "n_jobs":-1
    }

   
    cluster = DBSCAN(**param).fit(X_trans)

    # Calcul des Score
    try :
        score = metrics.silhouette_score(X_trans, cluster.labels_, metric='euclidean')
    except : 
        score = 0
    print(score)
    return score

optuna.logging.set_verbosity = True
#Optimisation
studyName = '30_h_dbscan_study_effet'
maximum_time = 30*60*60#second
number_of_random_points = 100
study = optuna.create_study(study_name = studyName,  direction="maximize")
study.optimize(objective, n_trials=number_of_random_points, timeout=maximum_time)# On créer 1000 points

#Sauvegarde du resultat
df = study.trials_dataframe()
df.to_json(studyName+'.json')
print(study.best_trial)