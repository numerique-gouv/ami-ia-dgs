"""
Ce module sert à formatter les différents retours de l'API à destination du front

Sauf exception, on renvoie du json
"""


def format_prediction(model_name, dataframe, nb_results=10):
    """
    Formatting vers Table

    :param model_name: nom du modèle
    :param dataframe: résultat de la prédiction
    :param nb_results: taille de la liste (meilleurs scores)
    :return: {columns: ..., dataSource:...} pour table antpro
    """
    columns = [
        {
            'title': model_name,
            'dataIndex': 'cat',
            'key': 'cat',
        },
        {
            'title': 'Probabilité' if model_name != 'topics' else 'Répartition',
            'dataIndex': 'proba',
            'key': 'proba',
        }
    ]

    dataSource = []
    for i, (k, v) in enumerate(sorted(zip(dataframe['class_name'], dataframe['proba']),
                                      key=lambda x: x[1],
                                      reverse=True)):
        dataSource.append({
            'key': i,
            'cat': k,
            'proba': round(v * 100., 2)
        })
    if nb_results:
        dataSource = dataSource[:nb_results]

    return {'columns': columns, 'datasource': dataSource}


def format_model_performances(perf_dict):
    columns = [
        {
            'title': "Modèle",
            'dataIndex': 'model',
            'key': 'model',
        }
    ]
    for i in range(len(perf_dict['Unnamed: 0'])):
        columns.append({
            'title': perf_dict['Unnamed: 0'][i],
            'dataIndex': perf_dict['Unnamed: 0'][i].replace(' ', '_'),
            'key': perf_dict['Unnamed: 0'][i].replace(' ', '_')
        })

    dataSource = []
    for key, vals in perf_dict.items():
        if key == 'Unnamed: 0':
            continue
        model_line = {'key': key, 'model': key}
        for i in range(len(vals)):
            model_line[perf_dict['Unnamed: 0'][i].replace(' ', '_')] = vals[i]
        dataSource.append(model_line)

    return {'columns': columns, 'datasource': dataSource}



def format_doc_ids(doc_ids):
    return {'doc_ids': doc_ids, 'nb_docs': len(doc_ids)}


def format_topics_and_dcos(topics_list):
    return [{'topic': d['title'],
                        'value': d['score'],
                        'label': d['score'],
                        'tooltip': ', '.join(d['words'][:10]) if 'words' in d else d['score']}
                       for d in topics_list]


def format_wordcloud(word_list, max_words=100):
    if max_words is None:
        max_words = len(word_list)
    return {'wordcloud': [{'word': k[0], 'id': i, 'weight': k[1]}
                           for i, k in enumerate(word_list[:max_words])]}


def format_doc_content(doc_id, doc_content):
    doc_content['doc_id'] = doc_id
    return doc_content


def format_document_cluster(doc_id, document_cluster):
    return {'doc_id': doc_id, 'cluster': document_cluster}


def format_document_topics(doc_id, document_topics):
    return {'doc_id': doc_id, 'topics': format_topics_and_dcos(document_topics)}


def format_document_complete(doc_id, document):
    document = format_doc_content(doc_id, document)
    document['topics'] = format_document_topics(doc_id, document['topics'])['topics']
    document['cluster'] = format_document_cluster(doc_id, document['cluster'])['cluster']
    return document


def format_topicmodel_nbtopics(nb_topics):
    return {'nb_topics': nb_topics}


def format_topicmodel_score(topicmodel_score):
    return {'coherence_score': float(topicmodel_score)}


def format_distances_mat(dist_mat):
    return {'distances_matrix': dist_mat.tolist()}


def format_topicmodel_pca(pca):
    """ Ici on renvoie une str qui sera interprétée comme une réponse html (pour embedding dans une iframe) """
    return pca


def format_topic_topwords(topic_ind, topwords):
    return {'topic_id': topic_ind, 'topwords': ", ".join(topwords)}


def format_topicmodel_complete(topic_model):
    topic_model['nb_topics'] = format_topicmodel_nbtopics(topic_model['nb_topics'])['nb_topics']
    topic_model['coherence_score'] = format_topicmodel_score(topic_model['coherence_score'])['coherence_score']
    topic_model['distances_matrix'] = format_distances_mat(topic_model['distances_matrix'])['distances_matrix']
    # topic_model['pca'] = format_topicmodel_pca(topic_model['pca'])['pca']
    return topic_model


def format_topic_weight(topic_ind, topic_weight):
    return {'topic_id': int(topic_ind), 'weight': float(topic_weight)}


def format_topic_wordcloud(topic_ind, wordcloud, max_words=100):
    return {'topic_id': int(topic_ind), 'wordcloud': format_wordcloud(wordcloud, max_words)['wordcloud']}


def format_topic_documents(topic_ind, topic_docs, max_docs=100):
    if max_docs is None:
        max_docs = len(topic_docs)
    return {'topic_id': int(topic_ind), 'documents': [{'doc_name': t[0], 'topic_score': t[1]} for t in topic_docs[:max_docs]]}


def format_topic_complete(topic_ind, topic):
    topic['topic_id'] = int(topic_ind)
    topic['weight'] = format_topic_weight(topic_ind, topic['weight'])['weight']
    topic['documents'] = format_topic_documents(topic_ind, topic['documents'])['documents']
    topic['wordcloud'] = format_topic_wordcloud(topic_ind, topic['wordcloud'])['wordcloud']
    return topic


def format_clustermodel_nbclusters(nb_clusters):
    return {'nb_clusters': nb_clusters}


def format_clustermodel_scores(scores):
    return {'scores': scores}


def format_clustermodel_pca(pca):
    return {'pca': {
        'clusters_ind': pca['cluster_ind'].values.tolist(),
        'X': pca['X'].values.tolist(),
        'Y': pca['Y'].values.tolist(),
        'weights': pca['W'].values.tolist(),
    }}


def format_clustermodel_complete(cluster_model):
    cluster_model['nb_clusters'] = format_clustermodel_nbclusters(cluster_model['nb_clusters'])['nb_clusters']
    cluster_model['scores'] = format_clustermodel_scores(cluster_model['scores'])['scores']
    cluster_model['distances_matrix'] = format_distances_mat(cluster_model['distances_matrix'])['distances_matrix']
    cluster_model['pca'] = format_clustermodel_pca(cluster_model['pca'])['pca']
    return cluster_model


def format_cluster_weight(cluster_id, weight):
    return {'cluster_id': int(cluster_id), 'weight': weight}


def format_cluster_documents(cluster_id, cluster_docs, max_docs=100):
    if max_docs is None:
        max_docs = len(cluster_docs)
    return {'cluster_id': int(cluster_id),
            'documents': [{'doc_name': t[0], 'document_similarity': t[1]} for t in cluster_docs[:max_docs]]}


def format_cluster_topics(cluster_id, topics):
    return {'cluster_id': int(cluster_id),
            'topics': format_topics_and_dcos(topics)}


def format_cluster_dcos(cluster_id, dcos):
    return {'cluster_id': int(cluster_id),
            'dcos': format_topics_and_dcos(dcos)}


def format_cluster_wordcloud(cluster_id, wordcloud, max_words=100):
    return {'cluster_id': int(cluster_id), 'wordcloud': format_wordcloud(wordcloud, max_words)['wordcloud']}


def format_cluster_complete(cluster_id, cluster):
    cluster['cluster_id'] = int(cluster_id)
    cluster['weight'] = format_cluster_weight(cluster_id, cluster['weight'])['weight']
    cluster['documents'] = format_cluster_documents(cluster_id, cluster['documents'])['documents']
    cluster['topics'] = format_cluster_topics(cluster_id, cluster['topics'])['topics']
    cluster['dcos'] = format_cluster_dcos(cluster_id, cluster['dcos'])['dcos']
    cluster['wordcloud'] = format_cluster_wordcloud(cluster_id, cluster['wordcloud'])['wordcloud']
    return cluster


def format_dco_clusters(dco_name, clusters):
    return {'dco_name': dco_name,
            'clusters': [{'topic': str(d['ind']),
                          'value': d['score'],
                          'label': d['score'],
                          'tooltip': d['score']}
                            for d in clusters]}