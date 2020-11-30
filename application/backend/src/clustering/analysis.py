"""
Module contenant les fonctions permettant de répondre à l'API front autour de la visualisation des topics et clusters

:author: cpoulet@starclay.fr
"""
import os
import sys
import pandas as pd
import numpy as np
import math

from functools import lru_cache
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

from clustering.clustering_models import ClusteringModels

###################
# Doc related
###################


@lru_cache()
def get_doc_in_mrv(id_doc_or_name):
    """
    Renvoie l'ensemble des docs MRV correspondant au nom ou à l'id donné (il y a des doublons...)

    :param id_doc_or_name: id ou nom de document
    :return: df format MRVeil
    """
    MRV_DATA = ClusteringModels().MRV_DATA
    doc_id = docname_to_docid(id_doc_or_name)
    res = MRV_DATA[(MRV_DATA['NUMERO_DECLARATION'] == doc_id) | (MRV_DATA['DOC_NAME'] == doc_id)]
    if len(res):
        return res
    raise ValueError(f'{id_doc_or_name} not found')


@lru_cache()
def docid_to_docname(doc_id):
    """
    Pour faciliter la lecture front, on crée un nom sous la forme "ID - DCO"

    :param doc_id: str
    :return: str
    """
    mrv_doc = get_doc_in_mrv(doc_id)
    dco = mrv_doc['DCO'].tolist()[0]
    return f"{doc_id} - {dco}"


def docname_to_docid(docname_or_id):
    """
    renvois l'id correspondant au nom du doc

    :param docname: str
    :return: str
    """
    if '-' in docname_or_id:
        return docname_or_id.split('-')[0].strip()
    return docname_or_id

#
# @lru_cache()
# def get_doc_in_topicmodel(id_doc):
#     mat = ClusteringModels().topicmodel.doc_topic_mat
#     return mat[mat['NUMERO_DECLARATION'] == id_doc]


def get_all_docs_ids():
    """
    Renvoie tous les id de docs de MRVeil

    :return: [str]
    """
    return ClusteringModels().topicmodel.doc_topic_mat['NUMERO_DECLARATION'].values.tolist()


def get_nb_docs_total():
    """
    renvoie le nb de docs total dans MRVeil

    :return: int
    """
    return len(get_all_docs_ids())


def get_document_info(docid_or_docname):
    """
    Renvoie les information du document correspondant à l'id ou au nom

    :param docid_or_docname: str
    :return: {
            "Numero de déclaration": doc_id,
            "nom": doc_name,
            "DCO": dco,
            "Description incident": des,
            "Etat patient": etat }
    """
    try:
        mrv_doc = get_doc_in_mrv(docid_or_docname)
        doc_id = mrv_doc['NUMERO_DECLARATION'].tolist()[0]
        doc_name = mrv_doc['DOC_NAME'].tolist()[0]
        dco = mrv_doc['DCO'].tolist()[0]
        des = mrv_doc['DESCRIPTION_INCIDENT'].tolist()[0]
        etat = mrv_doc['ETAT_PATIENT'].tolist()[0]
        if isinstance(etat, float) and math.isnan(etat):
            etat = ""
        doc = {
            "Numero de déclaration": doc_id,
            "nom": doc_name,
            "DCO": dco,
            "Description incident": des,
            "Etat patient": etat }
        return doc
    except (ValueError, IndexError):
        raise ValueError(f'{docid_or_docname} is not a valid id')


def get_document_topics(id_doc_or_name, n_topic=5):
    """
    Renvoie la distribution de topics pour le doc correspondant à l'id ou au nom, triés décroissant

    :param id_doc_or_name: str
    :param n_topic: int. nb de résultats
    :return: [{
                'ind': topic_ind,
                'title': 'Topic: K',
                'score': float en %age,
                'words': [str]
            }]
    """
    try:
        mat = ClusteringModels().topicmodel.doc_topic_mat
        doc_id = docname_to_docid(id_doc_or_name)
        df_doc = mat[(mat['NUMERO_DECLARATION'] == doc_id) | (mat['DOC_NAME'] == doc_id)]
        id_doc = mat['NUMERO_DECLARATION'].tolist()[0]
        dist = df_doc.T.iloc[:ClusteringModels().nb_topics].rename(
            columns={df_doc.index[0]: id_doc}).sort_values(id_doc, ascending=False)
        if n_topic:
            dist = dist.iloc[:n_topic]
        topics = []
        for (topic_name, topic_score) in dist[id_doc].items():
            topic_ind = int(topic_name[5:])
            # on récupère l'indice d'affichage
            topic_display_ind = ClusteringModels().topics_visu_reverse_order[topic_ind+1]
            topics.append({
                'ind': topic_display_ind,
                'title': 'Topic: ' + str(topic_display_ind),
                'score': topic_score,
                'words': [t[0] for t in ClusteringModels().topicmodel.model.show_topic(topic_ind)]
            })
        return topics
    except IndexError:
        raise ValueError(f'{id_doc_or_name} is not a valid document id')


def get_document_clusterind(id_doc_or_name):
    """
    Renvoie l'indice du cluster auquel appartient le document

    :param id_doc_or_name: str
    :return: int
    """
    try:
        mat = ClusteringModels().topicmodel.doc_topic_mat
        doc_id = docname_to_docid(id_doc_or_name)
        df_doc = mat[(mat['NUMERO_DECLARATION'] == doc_id) | (mat['DOC_NAME'] == doc_id)]
        num_cluster = int(df_doc['cluster'].values[0])
        return num_cluster
    except IndexError:
        raise ValueError(f'{id_doc_or_name} is not a valid document id')


def get_document_complete(id_doc_or_name):
    """
    Renvoie l'ensemble des infos du document
    :param id_doc_or_name: str
    :return: {
            "Numero de déclaration": doc_id,
            "nom": doc_name,
            "DCO": dco,
            "Description incident": des,
            "Etat patient": etat,
            "topics": [{
                'ind': topic_ind,
                'title': 'Topic: K',
                'score': float en %age,
                'words': [str]
            }],
            "cluster": int
    """
    doc = get_document_info(id_doc_or_name)
    doc['cluster'] = get_document_clusterind(id_doc_or_name)
    doc['topics'] = get_document_topics(id_doc_or_name)
    return doc


###################
# Topicmodel related
###################


def get_topicmodel_pca():
    """
    Renvoie l'html de la visualisation de la pca du modèle de topics

    :return: str
    """
    return ClusteringModels().topics_pca


def get_topicmodel_nbtopics():
    """
    Renvoie le nb de topics

    :return: int
    """
    return len(ClusteringModels().topic_weights)


def get_topicmodel_coherence_score():
    """
    Renvoie le score de cohérence

    :return: float
    """
    return ClusteringModels().topicmodel.get_coherence_score()


def get_topicmodel_distance_mat():
    """
    Revoie la matrice de distance inter-topics

    :return: numpy array [nb_topics, nb_topics]
    """
    return ClusteringModels().topicmodel.mdiff


def get_topicmodel_complete():
    """
    Renvoie l'ensemble des infos du topicmodel

    :return: {
        'nb_topics': int,
        'coherence_score': float,
        'distances_matrix': numpy array [nb_topics, nb_topics],
        'pca': str
    }
    """
    return {
        'nb_topics': get_topicmodel_nbtopics(),
        'coherence_score': get_topicmodel_coherence_score(),
        'distances_matrix': get_topicmodel_distance_mat(),
        'pca': get_topicmodel_pca()
    }


###################
# Topic related
###################


def get_topic_weight(topic_ind):
    """
    Pour un topic donné, renvoie son poids dans le corpus

    :param topic_ind: int
    :return: float
    """
    try:
        return ClusteringModels().topic_weights[topic_ind]
    except IndexError:
        raise ValueError(f'{topic_ind} is not a valid topic id')


def get_topic_documents(topic_ind):
    """
    Renvoie la liste des documents ayant pour topic majoritaire topic_ind

    :param topic_ind: int
    :return: [noms de docs]
    """
    try:
        return ClusteringModels().topic_documents_groups[topic_ind]
    except IndexError:
        raise ValueError(f'{topic_ind} is not a valid topic id')


def get_topic_wordcloud(topic_ind, nb_words=30):
    """
    Construit le Nuage de mot du topic

    :param topic_ind: int
    :param nb_words: nb de mots à inclure
    :return: [(mot, fréquence)] par ordre décroissant
    """
    try:
        df = pd.DataFrame.from_dict(ClusteringModels().topicmodel.viz['tinfo'])
        values = df.groupby('Category').get_group(f'Topic{topic_ind}')
        # this is for lambda = 0
        # values['rel_Freq'] = values['Freq'] / values['Total']
        # values = values.sort_values(by='rel_Freq', ascending=False)[['Term', 'rel_Freq']].values.tolist()
        # this is for lambda = 1
        values = values.sort_values(by='Freq', ascending=False)[['Term', 'Freq']].values.tolist()
        if nb_words:
            values = values[:nb_words]
        return values
    except IndexError:
        raise ValueError(f'{topic_ind} is not a valid topic id')


def get_topic_topwords(topic_ind):
    """
    Renvoie les mots décrivant le mieux le topic selon le modèle de LDA

    :param topic_ind:
    :return: [str]
    """
    try:
        topic_model_ind = ClusteringModels().topics_visu_order[topic_ind]
        # on récupère l'indice d'affichage
        return [t[0] for t in ClusteringModels().topicmodel.model.show_topic(topic_model_ind-1)]

    except (KeyError, KeyError):
        raise ValueError(f"{topic_ind} is not a valid topic id")


def get_topic_complete(topic_ind):
    """
    Renvoie l'ensemble des infos d'un topic

    :param topic_ind: int
    :return: {
            'weight': float,
            'documents': [noms de docs],
            'wordcloud': [(mot, fréquence)] par ordre décroissant
        }
    """
    topic = {}
    topic['weight'] = get_topic_weight(topic_ind)
    topic['documents'] = get_topic_documents(topic_ind)
    topic['wordcloud'] = get_topic_wordcloud(topic_ind)
    return topic


###################
# Cluster model related
###################

def get_clustermodel_pca():
    """
    Renvoie la pca du modèle de clustering : dataframe cluster_ind,X,Y,W

    :return: dataframe cluster_ind,X,Y,W
    """
    return ClusteringModels().clustermodel_pca


def get_clustermodel_nbclusters():
    """
    Renvoie le nb de clusters

    :return: int
    """
    return len(ClusteringModels().clusters)


def get_clustermodel_scores():
    """
    Renvoie les différents scores du modèle au format json

    :return: {"silhouette_score": float, "calinski_score": float, "daves_score": float}
    """
    return ClusteringModels().clustermodel_scores


def get_clustermodel_distance_mat():
    """
    Revoie la matrice de distance inter-clusters

    :return: numpy array [nb_clusters, nb_clusters]
    """
    return ClusteringModels().clustermodel.dist


def get_clustermodel_complete():
    """
    Renvoie l'ensemble des infos du modèle de clustering

    :return: {
        'nb_clusters': int,
        'scores': {"silhouette_score": float, "calinski_score": float, "daves_score": float},
        'distances_matrix': numpy array [nb_clusters, nb_clusters]
        'pca': dataframe cluster_ind,X,Y,W
    }
    """
    return {
        'nb_clusters': get_clustermodel_nbclusters(),
        'scores': get_clustermodel_scores(),
        'distances_matrix': get_clustermodel_distance_mat(),
        'pca': get_clustermodel_pca()
    }


###################
# Cluster related
###################


@lru_cache()
def get_cluster_weight(cluster_ind):
    """Pour un cluster donné, renvoie son poids dans le corpus

    :param cluster_ind: Numéro de cluster
    :return: {
                'weight': %age,
                'nb_docs': int
            }
    """
    try:
        return ClusteringModels().clusters_weights[cluster_ind]
    except IndexError:
        raise ValueError(f'{cluster_ind} is not a valid cluster id')


@lru_cache()
def get_cluster_topics(cluster_ind, n_topics=None):
    """
    Renvoie la liste des topics les plus fréquents dans les docs du cluster, triés décroissant

    :param cluster_ind: int
    :param n_topics: int. Nb de meilleurs scores à garder
    :return: [{
                'ind': topic_ind,
                'title': 'Topic: K',
                'score': topic_weight,
                'words': [str]
            }]
    """
    try:
        cluster = ClusteringModels().clusters.get_group(cluster_ind)
        n = ClusteringModels().topicmodel.model.num_topics
        T = cluster.iloc[:, 0:n]
        T = T.apply(lambda x: np.where(x > 0.1), axis=1)
        topic = np.concatenate(T.values, axis=1)
        c = pd.DataFrame.from_dict(Counter(topic[0]), orient='index', columns=['count'])
        df_top_topic = c.sort_values(by='count', ascending=False)
        if n_topics:
            df_top_topic = df_top_topic.iloc[:n_topics]
        most_frequent_topic = df_top_topic.index
        weights = df_top_topic.values / df_top_topic.values.sum()
        weights = [elt[0] for elt in weights]
        topics = []
        for topic_ind, topic_weight in zip(most_frequent_topic, weights):
            # on récupère l'indice d'affichage
            topic_display_ind = ClusteringModels().topics_visu_reverse_order[topic_ind+1]
            topics.append({
                'ind': topic_display_ind,
                'title': 'Topic: ' + str(topic_display_ind),
                'score': topic_weight,
                'words': [t[0] for t in ClusteringModels().topicmodel.model.show_topic(topic_ind)]
            })
        return topics
    except IndexError:
        raise ValueError(f'{cluster_ind} is not a valid cluster id')


@lru_cache()
def get_cluster_dcos(cluster_ind, nb_dcos=None):
    """
    Renvoie les DCOs les plus présentes dans les documents du cluster, triés décroissant

    :param cluster_ind: int
    :param nb_dcos: nb de meilleurs résultats à garder
    :return: [{
                'ind': doc_ind,
                'title': dco,
                'score': dco_weight
            }]
    """
    try:
        cluster = ClusteringModels().clusters.get_group(cluster_ind)
        c = cluster.groupby('DCO_ID').count()['Topic0']
        df = c.sort_values(ascending=False)
        if nb_dcos:
            df = df.iloc[:nb_dcos]
        most_frequent_dco = df.index
        weights = df.values / df.values.sum()
        # w= [elt[0] for elt in weight]
        dcos = []
        for dco_ind, dco_weight in zip(most_frequent_dco, weights):
            dco = ClusteringModels().id_to_dco.get(int(dco_ind), 'NON_LISTE')
            dcos.append({
                'ind': dco_ind,
                'title': dco,
                'score': dco_weight
            })
        return dcos
    except IndexError:
        raise ValueError(f'{cluster_ind} is not a valid cluster id')


def get_cluster_wordcloud(cluster_ind, nb_words=25):
    """
    Construit le Nuage de mot du cluster

    :param topic_ind: int
    :param nb_words: nb de mots à inclure
    :return: [(mot, nb_occ)] par ordre décroissant
    """
    try:
        cluster = ClusteringModels().clusters.get_group(cluster_ind)
        text = np.sum(cluster['text_lem'].values)
        words = sorted(Counter(text).items(), key=lambda x: x[1], reverse=True)
        if nb_words:
            words = words[:nb_words]
        return words
    except IndexError:
        raise ValueError(f'{cluster_ind} is not a valid cluster id')


@lru_cache()
def get_cluster_documents(cluster_ind):
    """
    Renvoie la liste des documents du cluster

    :param cluster_ind: int
    :return: [(doc name, similarité)]
    """
    try:
        cluster = ClusteringModels().MRV_DATA.groupby('cluster').get_group(cluster_ind)
        sorted_docs = cluster.sort_values(by='cluster_weight', ascending=False)[['DOC_NAME', "cluster_weight"]].values.tolist()
        #
        # center = ClusteringModels().clustermodel.model.cluster_centers_[cluster_ind][:ClusteringModels().nb_topics]
        # if cluster_ind not in ClusteringModels().clusters.groups.keys():
        #     most_rpz_doc, less_rpz_doc = '', ''
        #     return most_rpz_doc, less_rpz_doc
        #
        # cluster = ClusteringModels().clusters.get_group(cluster_ind)
        # Y = cluster.iloc[:, :ClusteringModels().nb_topics].values
        # similarites = []
        # for elt in Y:
        #     similarites.append(float(cosine_similarity(center.reshape(1, -1), elt.reshape(1, -1))[0][0]))
        #
        # sorted_docs = sorted(zip(cluster['DOC_NAME'].values, similarites), key=lambda x: x[1], reverse=True)
        return sorted_docs
    except (KeyError, IndexError):
        if 0 <= cluster_ind < len(ClusteringModels().clusters):
            return []
        raise ValueError(f'{cluster_ind} is not a valid cluster id')


def get_cluster_complete(cluster_ind):
    """
    Renvoie l'ensemble des infos d'un cluster

    :param cluster_ind: int
    :return: {  'weight': get_cluster_weight(cluster_ind)
                'documents': get_cluster_documents(cluster_ind)
                'topics': get_cluster_topics(cluster_ind)
                'dcos': get_cluster_dcos(cluster_ind)
                'wordcloud': get_cluster_wordcloud(cluster_ind)
            }
    """
    cluster = {}
    cluster['weight'] = get_cluster_weight(cluster_ind)
    cluster['documents'] = get_cluster_documents(cluster_ind)
    cluster['topics'] = get_cluster_topics(cluster_ind)
    cluster['dcos'] = get_cluster_dcos(cluster_ind)
    cluster['wordcloud'] = get_cluster_wordcloud(cluster_ind)
    return cluster


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath('..'))

    a = get_document_info('R1700004')
    b = get_document_topics('R1700004')
    c = get_document_clusterind('R1700004')

    a = get_topicmodel_coherence_score()
    b = get_topicmodel_distance_mat()
    # TODO : problème d'indices : 0 fonctionne pas pour get_topic_wordcloud...
    # topic : 1 à N
    c = get_topic_weight(1)
    d = get_topic_documents(1)
    e = get_topic_wordcloud(1)

    f = get_clustermodel_scores()
    g = get_clustermodel_distance_mat()
    h = get_cluster_weight(0)
    i = get_cluster_topics(0)
    j = get_cluster_dcos(0)
    k = get_cluster_wordcloud(0)
    l = get_cluster_documents(0)
    print('hop')