"""
Auteur : Quillivic Robin, Data Scientist, rquillivic@starclay.fr 

Permet d'analyse de manière globale un topic modèle : 
- visualisation interactive en 2D
- matrice des distances inter-topic
- calcul du score de cohérence

"""

import pandas as pd
import os
import sys
import pyLDAvis
import pyLDAvis.gensim

import matplotlib.pyplot as plt
from pyLDAvis import PreparedData


path_to_regroupement = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(path_to_regroupement))


class topicModelAnalysis():
    def __init__(self,topicmodel):
        self.topic_model = topicmodel

    def prepared_data_from_dict(self):
        """Permet d'afficher une visualisation interactive de LDA à partir d'un fichier json et de la librairie pyLDAvis

        Returns:
            PreparedData: données préparées
        """
        vis_data = self.topic_model.viz
        topic_coordinates = pd.DataFrame.from_dict(vis_data['mdsDat'])
        topic_info = pd.DataFrame.from_dict(vis_data['tinfo'])
        token_table = pd.DataFrame.from_dict(vis_data['token.table'])
        R = vis_data['R']
        lambda_step = vis_data['lambda.step']
        plot_opts = vis_data['plot.opts']
        client_topic_order = vis_data['topic.order']

        return PreparedData(topic_coordinates, topic_info,
                            token_table, R, lambda_step, plot_opts, client_topic_order)

    def plot_lda_viz(self):
        """
        Affiche la visualisation préparer dans prepared_data_from_dict

        Returns:
            visualisation  pyLDAvis.display(viz_data): 
        """
        viz_data = self.prepared_data_from_dict()
        pyLDAvis.enable_notebook()
        return pyLDAvis.display(viz_data)

    def plot_topic_mat(self, title="Matrice des distances (Jaccard) inter topics"):
        """
        Fonction pour afficher la matrice des distances inter_topic.
        Se base sur la bibliothèque matplotlib.

        Args:
            title (str, optional): Titre de l'image Defaults to "Matrice des distances (Jaccard) inter topics".
        """

        self.topic_model.build_topic_mat()
        fig, ax = plt.subplots(figsize=(18, 14))
        data = ax.imshow(self.topic_model.mdiff, cmap='RdBu_r', origin='lower')
        plt.title(title)
        plt.colorbar(data)
        
    def from_json_to_html(self,path):
        """Créer à partir d'un fichier json un fichier html de la visualisation

        Args:
            path ([type]): [description]
        """
        data = self.prepared_data_from_dict()
        pyLDAvis.save_html(data, path)