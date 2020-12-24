"""
Classe singleton permettant de charger les modèles de topic modelling et de clusterisation

:author: cpoulet@starclay.fr
"""

import os
import pandas as pd
import yaml
import json
import numpy as np
import logging
from sklearn.decomposition import PCA

from regroupement.training import train_topic, train_cluster
from backend_utils.config_parser import get_local_file, parse_full_config


class ClusteringModels:
    class __ClusteringModels:

        def __init__(self, with_preprocess=False):
            """
            Chargement des modèles

            :param with_preprocess: si True, on charge (ou précalcule) tout ce qui est nécessaire au front
            """
            self.logger = logging.getLogger('clustering_models')
            self.config = parse_full_config(get_local_file('../config.yaml'), get_local_file('../config_env.yaml'))

            self.DATA_PATH = self.config['data']['mrv']['path']
            self.MODEL_PATH = self.config['models']['path']
            CLUSTER_PATH = os.path.abspath(self.config['clusters']['path'])

            # self.MRV_DATA = pd.read_csv(os.path.join(self.DATA_PATH, 'declaration_mrv_complet.csv'))
            with open(os.path.join(CLUSTER_PATH, "cluster", 'mrv_with_clustering.json')) as f:
                self.MRV_DATA = pd.DataFrame.from_dict(json.load(f))

            with open(os.path.join(CLUSTER_PATH, 'training_config.yaml')) as f:
                self.clustering_config = yaml.load(f)

            # dictionaire des id pour le DCO
            self.id_to_dco = pd.read_csv(os.path.abspath(os.path.join(self.DATA_PATH, "referentiel_dispositif.csv")),
                                         delimiter=';', encoding='ISO-8859-1').to_dict()
            self.id_to_dco = {self.id_to_dco['DCO_ID'][i]: self.id_to_dco['LIBELLE'][i] for i in range(len(self.id_to_dco['DCO_ID']))}

            def add_docname(mat):
                mat['DCO'] = mat['DCO_ID'].apply(lambda x: self.id_to_dco.get(x, 'NON_LISTE')).fillna('INCONNU')
                mat['DOC_NAME'] = mat['NUMERO_DECLARATION'] + ' - ' + mat['DCO'].replace('/', '-')
                return mat
            self.MRV_DATA = add_docname(self.MRV_DATA)

            ###################
            # load models
            ###################
            self.logger.info('Loading Topic model')
            self.topicmodel = train_topic.TopicModel(self.clustering_config['config_name'],
                                                     self.clustering_config['topic'],
                                                     save_dir=CLUSTER_PATH)
            self.topicmodel.load(self.clustering_config['config_name'])
            self.topics_visu_order = {i+1: t for (i, t) in enumerate(self.topicmodel.viz['topic.order'])}
            self.topics_visu_reverse_order = {t: i+1 for (i, t) in enumerate(self.topicmodel.viz['topic.order'])}

            self.logger.info('Loading Cluster model')
            self.clustermodel = train_cluster.ClusterModel(self.clustering_config['config_name'],
                                                           self.clustering_config['cluster'],
                                                           save_dir=CLUSTER_PATH)
            self.clustermodel.topicmodel = self.topicmodel
            self.clustermodel.load(self.clustering_config['config_name'])

            if with_preprocess:
                #################
                # Apply or load generic preprocess
                #################

                self.logger.info('Preprocessing models')
                self.nb_topics = self.topicmodel.model.num_topics
                self.topic_weights = self.get_topic_weights()

                if os.path.exists(os.path.join(CLUSTER_PATH, 'topics_mat.npy')):
                    self.logger.info('...loading topics distance mat')
                    self.topicmodel.mdiff = np.load(os.path.join(CLUSTER_PATH, 'topics_mat.npy'))
                else:
                    self.logger.info('...Calculating topics distance mat')
                    self.build_topic_distances()
                    np.save(os.path.join(CLUSTER_PATH, 'topics_mat.npy'), self.topicmodel.mdiff)
                self.topicmodel.doc_topic_mat['cluster'] = self.clustermodel.model.labels_
                self.topicmodel.doc_topic_mat['nb_mot'] = self.topicmodel.doc_topic_mat['text_lem'].map(len)
                
                self.topicmodel.doc_topic_mat = add_docname(self.topicmodel.doc_topic_mat)
                if os.path.exists(os.path.join(CLUSTER_PATH, 'topics_documents.json')):
                    self.logger.info('...loading topics doc lists')
                    with open(os.path.join(CLUSTER_PATH, 'topics_documents.json')) as f:
                        json_data = json.load(f)
                        self.topic_documents_groups = [json_data[str(i)] for i in range(len(json_data))]
                else:
                    self.logger.info('...Calculating topics doc lists')
                    self.topic_documents_groups = self.build_topic_documents_groups()
                    with open(os.path.join(CLUSTER_PATH, 'topics_documents.json'), 'w') as f:
                        json.dump({i: l for i, l in enumerate(self.topic_documents_groups)}, f)
                topics_pca_path = os.path.join(CLUSTER_PATH, 'LDA', self.clustering_config['config_name'] + '.html')
                if os.path.exists(topics_pca_path):
                    self.logger.info('...loading topics pca')
                    with open(topics_pca_path) as f:
                        lines = []
                        for l in f.readlines():
                            l = l.strip('\n')
                            if '//' in l and l.strip().index('//') == 0:
                                continue
                            if not l or l == '\n':
                                continue
                            lines.append(l)
                        self.topics_pca = '\n'.join(lines)
                else:
                    self.logger.info('... Topics PCA not available')
                    self.topics_pca = None

                if os.path.exists(os.path.join(CLUSTER_PATH, 'clustermodel_scores.json')):
                    self.logger.info('...Loading Cluster model scores')
                    with open(os.path.join(CLUSTER_PATH, 'clustermodel_scores.json')) as f:
                        self.clustermodel_scores = json.load(f)
                else:
                    self.logger.info('...Calculating Cluster model scores')
                    self.clustermodel_scores = self.calculate_clustermodel_scores()
                    with open(os.path.join(CLUSTER_PATH, 'clustermodel_scores.json'), 'w') as f:
                        json.dump(self.clustermodel_scores, f)

                self.clusters = self.topicmodel.doc_topic_mat.groupby('cluster')
                # self.clustermodel.build_cluster_centers(self.topicmodel)
                self.clustermodel.build_dist_mat()
                self.clusters_weights = self.get_clusters_weights()
                if os.path.exists(os.path.join(CLUSTER_PATH, 'cluster_model_pca.csv')):
                    self.logger.info('...Loading Cluster model pca coordinates')
                    self.clustermodel_pca = pd.read_csv(os.path.join(CLUSTER_PATH, 'cluster_model_pca.csv'))
                else:
                    self.logger.info('...Calculating Cluster model pca coordinates')
                    coordinates = PCA(n_components=2).fit_transform(self.clustermodel.model.cluster_centers_)
                    df = pd.DataFrame()
                    df['cluster_ind'] = range(len(self.clusters))
                    df['X'] = coordinates[:, 0] / max([abs(v) for v in coordinates[:, 0]])
                    df['Y'] = coordinates[:, 1] / max([abs(v) for v in coordinates[:, 1]])
                    df['W'] = [w['weight'] for w in self.clusters_weights]
                    df.to_csv(os.path.join(CLUSTER_PATH, 'cluster_model_pca.csv'))
                    self.clustermodel_pca = df

            self.logger.info('...loading done')

        def get_display_topic_name(self, topic_ind):
            return f"Topic {self.topics_visu_reverse_order[topic_ind]}"

        def get_topic_weights(self):
            """
            Calcul du poids des topics

            :return: liste de poids, avec 0 en 1er (les topics commencent à 1)
            """
            df = pd.DataFrame.from_dict(self.topicmodel.viz['mdsDat'])
            return [0.0] + df['Freq'].values.tolist()    # on décale car les topics commencent à 1

        def build_topic_documents_groups(self):
            """
            calcul de la répartition des documents par topics, ordonné selon l'ordre de la visualisation gensim

            :return: [[liste de noms de docs, nom de topic]] avec [] en 0 car les topics commencent à 1
            """
            mat = self.topicmodel.doc_topic_mat
            mat = mat[mat['nb_mot'] > mat['nb_mot'].describe()['25%']]
            groups = []
            for k in range(self.nb_topics):
                # on récupère l'indice d'affichage
                visu_ind = self.topics_visu_order[k+1]
                seuil = mat['Topic' + str(visu_ind-1)].describe()['75%']
                top_doc = mat[mat['Topic' + str(visu_ind-1)] >= seuil].sort_values(by='Topic' + str(visu_ind-1), ascending=False) \
                    .drop_duplicates(subset=['NUMERO_DECLARATION'])
                groups.append(top_doc[['DOC_NAME', 'Topic' + str(visu_ind-1)]].values.tolist())
            return [[]] + groups    # on décale car les topics commencent à 1

        def build_topic_distances(self):
            """
            calcul de la matrice des distances inter-topic, ordonné selon l'ordre de la visualisation gensim
            :return:
            """
            self.topicmodel.build_topic_mat()
            dist_mat = self.topicmodel.mdiff
            order = [self.topics_visu_order[i+1] - 1 for i in range(self.nb_topics)]
            dist_mat2 = dist_mat[order, :][:, order]
            self.topicmodel.mdiff = dist_mat2

        def calculate_clustermodel_scores(self):
            """
            Calcul des scores du modèle

            :return: json
            """
            X = self.topicmodel.doc_topic_mat.iloc[:, :self.nb_topics].values
            self.clustermodel.compute_score(X)
            return {
                'Silhouette': float(self.clustermodel.silhouette_score),
                'Calinski Harabasz': float(self.clustermodel.calinski_harabasz_score),
                'Davies Bouldin': float(self.clustermodel.davies_bouldin_score),
            }

        def get_clusters_weights(self):
            """
            Calcul des poids des clusters

            :return: list de {weight: float, nb_docs: int}
            """
            clusters = self.MRV_DATA.groupby('cluster')
            clusters_nb_docs = []
            for i in range(len(clusters)):
                try:
                    clusters_nb_docs += [len(clusters.get_group(i))]
                except KeyError:
                    clusters_nb_docs += [0]
            nb_docs_total = sum(clusters_nb_docs)
            clusters_weights = [np.round(clen / float(nb_docs_total) * 100, 2) for clen in clusters_nb_docs]
            return [{
                'weight': w,
                'nb_docs': n
            } for w, n in zip(clusters_weights, clusters_nb_docs)]

    # instance chargée
    _instance = None

    def __init__(self, with_preprocess=False):
        """
        Instancation. Chargement des modèles au 1er appel.

        :param with_preprocess: si True, on charge tout ce qui est nécessaire pour le front
        """
        if ClusteringModels._instance is None:
            ClusteringModels._instance = ClusteringModels.__ClusteringModels(with_preprocess)

    def __getattr__(self, name):
        return getattr(self._instance, name)