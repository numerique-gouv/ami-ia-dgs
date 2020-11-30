"""
Auteur : Quillivic Robin, DataScientist chez Starclay, rquillivic@starclay.fr

Lance l'entrainement d'un set de configuration et créer les fichiers nécessaires pour le Livrable 3, construction du topic model et du cluster model
"""
import os
import sys
import yaml
import shutil
import logging

import pandas as pd
import numpy as np
import joblib

import train_cluster
import train_topic

from train_topic import TopicModel
from train_cluster import ClusterModel

with open(os.path.join(os.path.dirname(__file__), 'logging_config.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(log_config)


logger = logging.getLogger('entrainement')



if __name__ == "__main__":
    import shutil
    n_lignes = None #None
    logger.info('Chargement de la configuration des modèles...')

    current_dir = os.path.dirname(__file__)
    config_file = 'training_config.yaml'

    if len(sys.argv) == 2:
        config_file = sys.argv[1]

    if not os.path.isabs(config_file):
        config_file = os.path.abspath(os.path.join(current_dir, config_file))

    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    try_name = config['config_name']
    save_dir = os.path.join(config['path_to_save'], try_name)
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(os.path.join(current_dir, save_dir))

    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_file, save_dir)

    topic_model = TopicModel(try_name, config['topic'], save_dir=save_dir)

    filename = os.path.join(config['data']['path'], config['data']['filename'])
    if not os.path.isabs(filename):
        filename = os.path.abspath(os.path.join(current_dir, filename))
    
    try:
        df = pd.read_pickle(filename)
        if n_lignes is not None:
            df = df.iloc[:n_lignes, :]
        topic_model.logger.info('Chargement des données ! Ok !')
    except Exception as e:
        topic_model.logger.error(f'Error loading {filename}: {e}')
        raise ValueError(e)

    topic_model.build_dictionary(df)
    topic_model.build_corpus()
    topic_model.build_model()
    topic_model.build_viz()
    topic_model.build_doc_topic_mat()
    topic_model.logger.info(f'Coherence score : {topic_model.get_coherence_score(save=True)}')

    logger.info("Le topic modèle est entrainé, nous l'utilisons pour construire le modèle de clusterisation..")
    clustermodel = ClusterModel(try_name, config['cluster'], save_dir=save_dir)

    filename = os.path.join(config['data']['path'], config['data']['filename'])
    if not os.path.isabs(filename):
        filename = os.path.abspath(os.path.join(current_dir, filename))
    
    
    # Loading
    name = try_name
    topicmodel = topic_model
    clustermodel.topicmodel = topicmodel
    n = topicmodel.model.num_topics
    
    def add_col(df,col,X,save=False):
        """Permet d'ajouter des colonnes en plus de la représentation topic

        Args:
            df (pd.DataFrame): dataframe de la base de donnée MRveille
            col (list): liste des colonnes à ajouter
            X (array): représentation thèmatique

        Returns:
            X_new (array): représentation thèmatique complétée des colonnes présentes dans col
        """
        df_used = pd.DataFrame()
        from  sklearn.preprocessing import LabelEncoder
        for c in col : 
            le = LabelEncoder()
            df_used[c] = le.fit_transform(df[c].map(str).fillna(' ').values)
            if save :
                joblib.dump(le, os.path.join(clustermodel.save_dir,'le_'+str(c)+'.sav'))
            X_new = np.concatenate((X,df_used.values),axis=1)
        return X_new
    columns =  config['cluster']['model']['add_columns']
    data = pd.read_csv(config['data']['mrv'])
    
    X = topicmodel.doc_topic_mat.iloc[:n_lignes, :n-1].values
    X = add_col(data.iloc[:n_lignes,:],columns,X,save=True)


    clustermodel.train(X)
    clustermodel.compute_score(X,save=True)
    clustermodel.compute_evaluation_score(data.iloc[:n_lignes,:],save=True)
    clustermodel.predict_mrv(X, data.iloc[:n_lignes,:],save=True)
    
    clustermodel.save(X)

