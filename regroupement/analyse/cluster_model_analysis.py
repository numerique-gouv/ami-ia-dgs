""" 
Auteur :  Quillivic Robin, Data Scientist chez Starclay rquillivic@starclay.fr

Permet d'analyse un modèle de clusterisation de manière globale : 
- Affichage en 2 dimensions
- Calcul des score de performances :silhouette, calinski et bouldin
- Matrice de distance inter-cluster

"""

import pandas as pd
import gensim
import numpy as np
import os
import sys


import matplotlib.pyplot as plt
import plotly.offline as py

import yaml
from spacy.lang.fr.stop_words import STOP_WORDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotly import graph_objs as pgo


path_to_regroupement = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(path_to_regroupement))

import training.train_cluster
import training.train_topic


class clusterModelAnalysis:
    def __init__(self,clustermodel):
        self.cluster_model = clustermodel

    def print_score(self, X):
        """Affiche les scores du modèle de clustering

        Args:
            X (np.array): Données d'entraiement du ClusterModel
        """
        self.cluster_model.compute_score(X)

        # Compris entre -1 et 1 avec 1 comme meilleure valeure
        print('Le score silhouette est de: ',
              self.cluster_model.silhouette_score)
        # Entre 0 et +inf avec 0 en pire valeure
        print('Le score calinski harabasz est de:',
              self.cluster_model.calinski_harabasz_score)
        # Entre 0 et + inf avec meilleure valeure
        print('Le score de davies bouldin  est de',
              self.cluster_model. davies_bouldin_score)

    def plot_dist_mat(self, topicmodel,title="Distance cosinus inter cluster"):
        """
        Affichage de la matrice des distances inter cluster

        Args:
            title (str, optional): titre de l'image. Defaults to "Distance cosinus inter cluster".
        """
        self.cluster_model.build_cluster_centers(topicmodel)
        self.cluster_model.build_dist_mat()
        fig, ax = plt.subplots(figsize=(18, 14))
        data = ax.imshow(self.cluster_model.dist,
                         cmap='RdBu_r', origin='lower')
        plt.title(title)
        plt.colorbar(data)

    def prepare_cluster_plot(self, topicmodel):
        """Prépare l'affichage en 2D du modèle des Clusters

        Args:
            topicmodel (TopicModel): Modèle des thèmes

        Returns:
            df_cluster (pd.DataFrame): Coordonnées, poids de chaque cluster pour l'affichage
        """
        df_doc_topic = topicmodel.doc_topic_mat
        df_doc_topic['cluster'] = self.cluster_model.model.labels_
        group = df_doc_topic.groupby('cluster')
        self.cluster_model.build_cluster_centers(topicmodel)
        X = self.cluster_model.model.cluster_centers_
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X_scaled)

        df_cluster = pd.DataFrame()
        df_cluster['cluster'] = group['text'].count().index
        df_cluster['weight'] = group['text'].count().tolist()
        df_cluster['X'] = X_reduced[:, 0]
        df_cluster['Y'] = X_reduced[:, 1]

        return df_cluster

    def plot_cluster(self, topicmodel):
        """Affiche un cluster

        Args:
            topicmodel (TopicModel): topic modèle associé
        """
        df_cluster = self.prepare_cluster_plot(topicmodel)
        trace = pgo.Scatter(x = df_cluster['X'],
                            y = df_cluster['Y'],
                            text = df_cluster.index,
                            mode = 'markers',
                            #
                            marker = pgo.scatter.Marker(size=df_cluster['weight'],
                                                      sizemode='diameter',
                                                      sizeref=df_cluster['weight'].max(
                            )/50,
            opacity=0.5)
        )

        layout5 = pgo.Layout(title='Distribution des Clusters (PCA n=2)',
                             xaxis=pgo.layout.XAxis(showgrid=True,
                                                    zeroline=True,
                                                    showticklabels=True),
                             yaxis=pgo.layout.YAxis(showgrid=True,
                                                    zeroline=True,
                                                    showticklabels=True),
                             hovermode='closest'
                             )

        fig5 = pgo.Figure(layout=layout5)
        fig5.add_trace(trace)
        fig5.layout.update(height=800, width=800)
        fig5.show()
