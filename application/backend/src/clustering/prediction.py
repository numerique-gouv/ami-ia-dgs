"""
Module contenant la fonction de clusterisation de nouveaux documents

:author: cpoulet@starclay.fr
"""
import numpy as np
import pandas as pd

from regroupement.inference.inference import inference_cluster_doc
from clustering.clustering_models import ClusteringModels


def clusterize_doc(file_dataframe, file_results):
    """
    clusterise un document

    on crée 16 documents fictifs à partir des combinaisons des 2 meilleures prédictions de chaque modèle de classif,
    on clusterise ces 16 documents, puis on aggrège les résultats en pondérant avec le produit des scores de la combinaison
    de prédictions de chaque document

    :param file_dataframe: document en dataframe type MERVeil
    :param file_results: {key: prediction_df} pour key in [DCO, effet, dysfonctionnement, consequence]
    :return: df['class_name', 'proba'] pour les topics, pareil pour les clusters
    """
    def get_model_result(results, modelname):
        r_model = [v for v in results if v['model_name'] == modelname][0]
        return r_model['predictions']

    predictions = []
    predictions_priors = []
    for i in range(16):
        dco = get_model_result(file_results, 'DCO').iloc[i // 8]
        effet = get_model_result(file_results, 'effet').iloc[(i // 4) % 2]
        dysfonc = get_model_result(file_results, 'dysfonctionnement').iloc[(i // 2) % 2]
        conseq = get_model_result(file_results, 'consequence').iloc[i % 2]
        predictions.append([dco['class'], effet['class'], dysfonc['class'], conseq['class']])
        predictions_priors.append([dco['proba'], effet['proba'], dysfonc['proba'], conseq['proba']])

    inference_res_input = {}
    inference_res_input['DCO_ID'] = [v[0] for v in predictions]
    inference_res_input['TEF_ID'] = [v[1] for v in predictions]
    inference_res_input['TDY_ID'] = [v[2] for v in predictions]
    inference_res_input['CDY_ID'] = [v[3] for v in predictions]
    predictions_priors = np.array([np.prod(v) for v in predictions_priors])
    predictions_priors /= np.sum(predictions_priors)

    topics_df, weights_df = inference_cluster_doc(ClusteringModels().clustering_config,
                                                  ClusteringModels().clustermodel,
                                                  ClusteringModels().topicmodel,
                                                  file_dataframe,
                                                  inference_results=inference_res_input)

    weights_df *= predictions_priors.reshape(-1, 1)
    weights_df = weights_df.sum(axis=0)   # now a Series
    weights_df /= np.sum(weights_df)

    topics_output = pd.DataFrame(np.array([topics_df.columns.tolist(), topics_df.iloc[0].tolist()]).T,
                                 columns=['class_name', 'proba'])

    def translate_class_name(class_name):
        # mapping des topics d'affichage
        topic_ind = int(class_name[5:])
        translated_topic_ind = ClusteringModels().topics_visu_reverse_order[topic_ind+1]
        return f'Topic {translated_topic_ind}'

    topics_output['class_name'] = topics_output['class_name'].map(translate_class_name)
    topics_output["proba"] = pd.to_numeric(topics_output["proba"])
    topics_output = topics_output.sort_values('proba', ascending=False)

    cluster_output = pd.DataFrame(np.array([weights_df.axes[0].tolist(), weights_df.values.tolist()]).T,
                                  columns=['class_name', 'proba'])
    cluster_output["proba"] = pd.to_numeric(cluster_output["proba"])
    cluster_output = cluster_output.sort_values('proba', ascending=False)

    return topics_output, cluster_output
