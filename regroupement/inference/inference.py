"""
Auteur : Quillivic Robin; DatScientist chez starclay
Fichier permettant l'inference du numeros de cluster d'un nouveau document
"""

import numpy as np

import sys
import os

path_to_regroupement = os.path.dirname(os.path.dirname(__file__)) #os.path.dirname(os.path.dirname('.'))
sys.path.insert(1, os.path.join(path_to_regroupement, 'training'))
sys.path.insert(2,path_to_regroupement)

from data_preparation.prepare_data import *
import joblib


# loaded columns transformers
loaded_transformer = {}


def add_cols(df, cols, X, clustermodel):
    """
    Permet d'ajouter des colonnes en plus de la représentation topic;
    Charge les transformers si pas encore chargés

    Args:
        df (pd.DataFrame): dataframe de la base de donnée MRveille
        cols (list): liste des colonnes à ajouter
        X (array): représentation thèmatique
        clustermodel (ClusterModel): modèle de clustering utilisé

    Returns:
        X (array): X + colonnes ajoutées. représentation thèmatique complétée des colonnes présentes dans col
    """
    global loaded_transformer

    df_used = pd.DataFrame()
    for c in cols:
        try:
            if str(c) not in loaded_transformer:
                le = joblib.load(os.path.join(clustermodel.save_dir, 'le_' + str(c) + '.sav'))
                le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                loaded_transformer[str(c)] = le_dict
            le_dict = loaded_transformer[str(c)]
            df_used[c] = df[c].apply(lambda x: le_dict.get(str(x), -1))
        except Exception as e:
            raise RuntimeError(f'Error while adding col to data : {e}')
    X = np.concatenate((X, df_used.values), axis=1)
    return X


def inference_cluster_doc(model_config, clustermodel, topicmodel, data_doc, inference_results=None):
    """

    Renvoie les scores de probabilités d'appartenance aux clusters pour le ou les docs de data_doc

    Les prédictions des modèles de classifications peuvent être donnés via inference_results au format
        {key: [values]} avec key dans [DCO_ID, TEF_ID, TDY_ID, CDY_ID] et values dans les valeurs d'entrées connues
        (i.e. label de classe prédite)

    S'il n'y a qu'un doc dans data_doc, mais plusieurs valeurs dans les inférences, le doc est répliqué autant de fois
    que nécessaire. Cela permet de tenter de clusteriser avec plusieurs valeurs d'inférence différentes


    Args:
        model_config: configuration de training des modèles
        clustermodel (ClusterModel): modèle de clusterisation
        topicmodel (TopicModel): modèle de topic
        data_doc (pd.dataframe): le(s) document(s) au format pandas

        inference_results (dict, optional): {key: [values]} avec key dans [DCO_ID, TEF_ID, TDY_ID, CDY_ID] et values dans les valeurs d'entrées connues

    Returns:
        dataframe des scores par topics, dataframe des scores par cluster
    """
    # preparer la donnée, 
    X_trans = prepare_data(data_doc, save_dir=None, use_multiprocessing=False)
    # construire la distribution des thèmes
    X = topicmodel.predict(X_trans)
    # inférer les colonnes DCO, TYPO
    if inference_results is not None:
        if len(data_doc) == 1 and len(list(inference_results.values())[0]) != 0:
            nb_repeat = len(list(inference_results.values())[0])
            data_doc = data_doc.iloc[np.arange(len(data_doc)).repeat(nb_repeat)]
            X = X.iloc[np.arange(len(X)).repeat(nb_repeat)]
        # C'est ici que vont tes fonctions d'inférence
        if 'DCO_ID' in inference_results:
            data_doc['DCO_ID'] = inference_results['DCO_ID']
        if 'TEF_ID' in inference_results:
            data_doc['TEF_ID'] = inference_results['TEF_ID']
        if 'TDY_ID' in inference_results:
            data_doc['TDY_ID'] = inference_results['TDY_ID']
        if 'CDY_ID' in inference_results:
            data_doc['CDY_ID'] = inference_results['CDY_ID']
    
    # ajout des colonnes complémentaires
    cols = model_config['cluster']['model']['add_columns']
    cluster_input = X.iloc[:, :topicmodel.model.num_topics-1].values
    if len(cols):
        cluster_input = add_cols(data_doc, cols, cluster_input, clustermodel)

    # Inférer le cluster
    clusters_weights = clustermodel.soft_clustering_weights(cluster_input)
    return X.iloc[:, :topicmodel.model.num_topics-1], pd.DataFrame(clusters_weights, columns=[str(i) for i in range(len(clusters_weights[0]))])


if __name__ == "__main__" :

    import train_topic, train_cluster

    # Chargemennt des modèles du livrable 2
    """path_to_signalement_ia = os.path.dirname(path_to_regroupement)
    sys.path.insert(2,os.path.join(path_to_signalement_ia,'demonstrateur'))
    from main import load_model
    from lib_extraction import create_fus, from_xml_to_mrv_format, from_pdf_to_mrv_format, plumber_df

    with open(os.path.join(path_to_signalement_ia,'demonstrateur', 'config.yaml'), 'r') as stream:
        config_demonstrateur = yaml.load(stream, Loader=yaml.FullLoader)

    MODEL_PATH = config_demonstrateur['Model']['path']
    PREPRO_PATH = config_demonstrateur['Data']['prepro']['path']
    TEST_PATH = config_demonstrateur['Data']['demonstrateur']['path']"

    # load data for preprocessing
    with open(os.path.abspath(os.path.join(PREPRO_PATH,'Colonnes.json')), 'r') as file:
            Colonnes = json.load(file)
    with open(os.path.abspath(os.path.join(PREPRO_PATH,'mapping.json')), 'r') as file:
        mapping = json.load(file)"""

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as stream:
        globale_config = yaml.load(stream, Loader=yaml.FullLoader)
    path_mrv = globale_config['data']['mrv']['path']
    data = pd.read_csv(path_mrv)

    config_path = os.path.join(os.path.join(path_to_regroupement, 'training'), 'training_config.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    SAVE_PATH = config['path_to_save']

    name = '10_11_2020'
    model_config_path = os.path.join(SAVE_PATH, name, 'training_config.yaml')
    with open(model_config_path, 'r') as stream:
        model_config = yaml.load(stream, Loader=yaml.FullLoader)
    
    # Loading
    topicmodel = train_topic.TopicModel(name, model_config['topic'],save_dir=os.path.join(SAVE_PATH, name))
    topicmodel.load(name)

    clustermodel = train_cluster.ClusterModel(name, model_config['cluster'],save_dir=os.path.join(SAVE_PATH, name))
    clustermodel.topicmodel = topicmodel
    clustermodel.load(name)

    #exemple de document
    data_doc = data.iloc[:1]

    cluster = inference_cluster_doc(config, clustermodel, topicmodel, data_doc)
    
    print(cluster)
