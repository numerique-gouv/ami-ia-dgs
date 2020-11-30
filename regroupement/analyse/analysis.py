"""
Auteur :  Quillivic Robin, rquillivic@straclay.fr
Description : 
Contient la classe et les fonctions essentielles pour analyser les modèles de regroupement (topic model + cluster model)
"""

import yaml
import sys


import pandas as pd
import gensim
import numpy as np
import os
import pyLDAvis
import pyLDAvis.gensim


from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import joblib
from sklearn import metrics


from sklearn.cluster import FeatureAgglomeration, DBSCAN, OPTICS, KMeans
from sklearn.metrics import pairwise_distances

import yaml
from spacy.lang.fr.stop_words import STOP_WORDS
from scipy import spatial

path_to_regroupement = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(path_to_regroupement))

import training.train_topic
import training.train_cluster

path_to_conf_file = os.path.join(os.path.dirname(__file__))
with open(os.path.join(path_to_conf_file, 'analyse_config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as stream:
    globale_config = yaml.load(stream, Loader=yaml.FullLoader)

try:
    path_mrv = globale_config['data']['mrv']['path']
    data_mrv = pd.read_csv(path_mrv)

    prepared_mrv_path = config_data['data']['prepared_mrv']['path']
    prepared_mrv_data = pd.read_pickle(prepared_mrv_path)
except :
    data_mrv = None


def display_doc(id_doc, mrv):
    """Affiche un signalement avec un format lisible

    Args:
        id_doc (int): id du document
        mrv (pd.DataFrame): base de donnée des signalements Mrveille

    Returns:
        Doc (str): signalement au format str
    """
    dco = mrv[mrv['NUMERO_DECLARATION'] == id_doc]['DCO'].tolist()[0]
    des = mrv[mrv['NUMERO_DECLARATION'] == id_doc]['DESCRIPTION_INCIDENT'].tolist()[
        0]
    etat = mrv[mrv['NUMERO_DECLARATION'] == id_doc]['ETAT_PATIENT'].tolist()[0]
    doc = "Numero de déclaration : "+str(id_doc)+"\n DCO :  "+str(
        dco)+"\n Description incident : \n"+str(des)+" \n  Etat patient : \n"+str(etat)

    return doc


class Analysis:
    def __init__(self,topicmodel,clustermodel):
        self.cluster = pd.DataFrame
        self.topicmodel = topicmodel
        self.clustermodel = clustermodel
        self.seuil = 0.1
        self.top_topics = None
        self.top_dcos = None
        self.top_topics_docs = None
        self.top_cluster_docs = None

        self.wc = None
        self.id_to_dco = None

    def load(self, path_to_conf_file):
        """Charge les fichiers nécessaires l'analyse

        Args:
            path_to_conf_file (str): chemin vers le fichier de configration
        """

        self.id_to_dco = pd.read_csv(os.path.abspath(os.path.join(globale_config['data']['id_to_dco']['path'])
                                                     ), delimiter=';', encoding='ISO-8859-1')

    def get_topic_weight(self, x):
        """Pour un topic donné, renvoie son poids dans le corpus

        Args:
            x (int): Numéro de topic

        Returns:
            weight (weight): poids du topic x
        """
        df = pd.DataFrame.from_dict(self.topicmodel.viz['mdsDat'])
        total = df['Freq'].sum()
        perc = df[df['topics'] == int(x)]['Freq']/total * 100

        return np.round(perc.values[0], 2)

    def get_significant_topic(self, x):
        """Selectionne les topics dans un documents supérieur au seuil x

        Args:
            x (array): Distribution de topic sur un document

        Returns:
            array: numero des topics significatifs
        """
        return np.where(x > self.seuil)

    def get_top_topics(self, k=5):
        """Renvoie les 5 topics les plus présents dans le cluster : self.cluster

        Args:
            k (int, optional):Nombre de topics à sélectionner. Defaults to 5.

        Returns:
            df (pd.DataFrame ): Dataframe contenant les 5 topics majoritaires et leur % respectifs
        """
        n = self.topicmodel.model.num_topics
        T = self.cluster.iloc[:, 0:n -
                              1].apply(lambda x: self.get_significant_topic(x), axis=1)
        topic = np.concatenate(T.values, axis=1)
        c = pd.DataFrame.from_dict(
            Counter(topic[0]), orient='index', columns=['count'])
        df_top_topic = c.sort_values(by='count', ascending=False).iloc[:k]
        most_frequent_topic = df_top_topic.index
        weight = df_top_topic.values/df_top_topic.values.sum()
        w = [elt[0] for elt in weight]
        topic = []
        for elt in most_frequent_topic:
            word = [t[0] for t in self.topicmodel.model.show_topic(elt)[:2]]
            word_str = ' '.join(word)
            title = 'Topic: '+str(elt) + '(' + word_str + '...)'
            topic.append(title)
        df = pd.DataFrame()
        df['top_topics'] = topic
        df['%'] = [100*x for x in w]
        self.top_topics = df
        return df

    def get_top_dcos(self, k=5):
        """Renvoie les 5 DCOs les plus présents dans le cluster : self.cluster
        Args:
            k (int, optional):Nombre de DCOs à sélectionner. Defaults to 5.

        Returns:
            df_top_dcos (pd.DataFrame): Dataframe contenant les 5 DCOs majoritaires et leur % respectifs
        """
        c = self.cluster.groupby('DCO_ID').count()['Topic0']
        df = c.sort_values(ascending=False).iloc[:k]
        most_frequent_dco = df.index
        weight = df.values/df.values.sum()
        #w= [elt[0] for elt in weight]
        dcos = []
        for elt in most_frequent_dco:
            dco = self.id_to_dco[self.id_to_dco['DCO_ID']
                                 == int(elt)]['LIBELLE']
            if len(dco)>0:                
                dcos.append(dco.iloc[0])
            else :
                dcos.append('NON_LISTE')
                
        df_top_dcos = pd.DataFrame()
        df_top_dcos['top_dcos'] = dcos
        df_top_dcos['%'] = [100*x for x in weight]
        self.top_dcos = df_top_dcos
        return df_top_dcos

    def build_wc(self):
        """
        Construit le Nuage de mot du cluster
        """
        text = np.sum(self.cluster['text'].values)
        wc = WordCloud(background_color="white", stopwords=STOP_WORDS,
                       width=1000, height=500, max_words=30).generate(text)
        self.wc = wc

    def plot_wc(self):
        """
        Affiche le Nuage de mot du cluster self.cluster
        """
        # lower max_font_size
        plt.figure(figsize=(10, 20))
        plt.imshow(self.wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def get_top_docs_topics(self, x: int, mrv=data_mrv, n_doc=1):
        """
        Renvoie les documents les plus représentatifs pour un topic donné
        Args:
            x (int): numero du topic
            mrv (pd.DataFrame): base de données mrv
            n_doc (int, optional): Nombre de documents représentatifs à renvoyer .Default to 1

        Returns: 
            doc : Document(s) représentatifs du topics
        """
        mat = self.topicmodel.doc_topic_mat
        n_topics = self.topicmodel.model.num_topics
        mat['nb_mot'] = mat['text_lem'].map(len)
        mat = mat[mat['nb_mot'] > mat['nb_mot'].describe()['25%']]
        Group = []
        for k in range(n_topics):
            seuil = mat['Topic'+str(k)].describe()['75%']
            top_doc = mat[mat['Topic'+str(k)] >= seuil].sort_values(by='Topic'+str(
                k), ascending=False).drop_duplicates(subset=['NUMERO_DECLARATION'])
            Group.append(top_doc)
        if n_doc == 1:
            id_doc = Group[x-2]['NUMERO_DECLARATION'].values[0]
            return(display_doc(id_doc, mrv))

        elif n_doc == 2:
            id_doc1 = Group[x-2]['NUMERO_DECLARATION'].values[0]
            id_doc2 = Group[x-2]['NUMERO_DECLARATION'].values[1]
            return '\n-----------\n Document 1 : \n ' + display_doc(id_doc1, mrv) + '\n-----------\n Document 2 : \n '+display_doc(id_doc2, mrv)

        elif n_doc == 3:
            id_doc1 = Group[x-2]['NUMERO_DECLARATION'].values[0]
            id_doc2 = Group[x-2]['NUMERO_DECLARATION'].values[1]
            id_doc3 = Group[x-2]['NUMERO_DECLARATION'].values[2]
            return '\n-----------\n Document 1 : \n ' + display_doc(id_doc1, mrv) + '\n-----------\n Document 2 : \n '+display_doc(id_doc2, mrv) + '\n-----------\n Document 3 : \n '+display_doc(id_doc3, mrv)

        else:
            raise Exception

    def get_top_docs_cluster(self, x: int, mrv=data_mrv):
        """
         Renvoie les documents le plus représentatif et le moins représentatif pour un cluster donné

        Args:
            x (int): numero du cluster
            mrv ([type], optional): Données MRV pour obtenir l'id du document. Defaults to data_mrv.

        Returns: 
            most_rpz_doc, less_rpz_doc (str,str): Document le plus et document le moins représentatifs du cluster x
        """
        mat = self.topicmodel.doc_topic_mat
        n_topics = self.topicmodel.model.num_topics
        mat['cluster'] = self.clustermodel.model.labels_
        mat['nb_mot'] = mat['text_lem'].map(len)
        mat = mat[mat['nb_mot'] > mat['nb_mot'].describe()['25%']]
         
        Clusters = mat.groupby('cluster')
        
        center = self.clustermodel.model.cluster_centers_[x]
        if x not in Clusters.groups.keys(): 
            most_rpz_doc, less_rpz_doc = '', ''
            return most_rpz_doc, less_rpz_doc
        cluster = Clusters.get_group(x)
        Y = cluster.iloc[:, :n_topics - 1].values
        similarite = []
        for elt in Y:
            similarite.append(1 - spatial.distance.cosine(center, elt))

        ix_doc = np.argmax(similarite)
        id_doc = cluster['NUMERO_DECLARATION'].iloc[ix_doc]

        most_rpz_doc = display_doc(id_doc, mrv)

        ix_doc = np.argmin(similarite)
        id_doc = cluster['NUMERO_DECLARATION'].iloc[ix_doc]

        less_rpz_doc = display_doc(id_doc, mrv)
        return(most_rpz_doc, less_rpz_doc)


def TopicAnalysis(x, topicmodel,clustermodel, n_term=25, n_doc=1):
    """Construit les élements nécessaires à l'analyse d'un topic

    Args:
        x (int): Numérosdu topic
        topicmodel (gensim.model): topicmodel
        n_term (int, optional): Nombre de terme à afficher pour caratériser le topic. Defaults to 20.
        n_doc (int, optional): Nombre de document représentatifs à afficher Defaults to 1. Max 3

    Returns:
        doc (str): document associé au topic
        term (str): termes associés au topic
        weight (int): poids du topic dans le corpus
    """
    analyse = Analysis(topicmodel,clustermodel)
    doc = analyse.get_top_docs_topics(x, mrv=data_mrv, n_doc=n_doc)

    df = pd.DataFrame.from_dict(topicmodel.viz['token.table'])
    term = '\n '.join(df.groupby('Topic').get_group(x).sort_values(
        by='Freq', ascending=False)['Term'].iloc[:n_term].tolist())
    weight = analyse.get_topic_weight(x)
    return (doc, term, weight)


def plotTopicAnalysis(x, topicmodel,clustermodel, n_doc=1):
    """Analyse du topic numéro x

    rgs:
        x (int): Numéros du topic
        topicmodel (gensim.model): topicmodel
        n_doc (int, optional): Nombre de document représentatifs à afficher Defaults to 1. Max 3
    """
    doc, term, weight = TopicAnalysis(x, topicmodel,clustermodel, n_doc=n_doc)
    wc = WordCloud(background_color="white", stopwords=STOP_WORDS,
                   width=1000, height=500, max_words=30).generate(term)
    print("Analyse du Topic "+str(x)+" représentant "+str(weight) + "% du corpus")
    print("------------------------")
    print("Documents les plus représentatifs: ")
    print("------------------------")
    print(doc)
    print("------------------------")
    print("Nuage de mot des termes associés au topic:")

    fig = plt.figure(figsize=(10, 15))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Termes associés au thème sélectionné")

    plt.show()


def ClusterAnalysis(x: int, topicmodel, clustermodel):
    """Permet de réaliser l'analyse d'un cluster

    Args:
        x (int): Numéros du topic
        topicmodel (TopicModel): Topic model
        clustermodel (ClusterModel): Modèle de cluster

    Returns: 
        wc (WordCloud): Nuage de mot du cluster
        df_dcos (pd.DataFRame): dcos associés au cluster
        df_topics (pd.DataFRame): topics associés au cluster
        most_rpz_doc, less_rpz_doc (str,str): Document le plus et le moins représentatifs du cluster 
        weight (float) : Poids du cluster dans le corpus
    """
    analyse = Analysis(topicmodel,clustermodel)
    clustermodel.build_cluster_centers(topicmodel)

    analyse.load(path_to_conf_file)

    mat = topicmodel.doc_topic_mat
    mat['cluster'] = clustermodel.model.labels_
    Clusters = mat.groupby('cluster')

    cluster = Clusters.get_group(x)
    analyse.cluster = cluster

    df_topics = analyse.get_top_topics()
    df_dcos = analyse.get_top_dcos()

    analyse.build_wc()
    wc = analyse.wc

    most_rpz_doc, less_rpz_doc = analyse.get_top_docs_cluster(x)
    weight = np.round(len(cluster)/len(mat)*100, 2)

    return(wc, df_dcos, df_topics, most_rpz_doc, less_rpz_doc, weight)


def plotClusterAnalysis(x, topicmodel, clustermodel):
    """
    Affiche l'analyse d'un cluster 

    Args:
        x (int): Numéros du topic
        topicmodel (TopicModel): modèle de topic
        clustermodel (ClusterModel): modèle de cluster
    """
    wc, df_dcos, df_topics, most_rpz_doc, less_rpz_doc, weight = ClusterAnalysis(
        x, topicmodel, clustermodel)

    fig = plt.figure(figsize=(7, 15))

    ax = fig.add_subplot(311)
    df_topics.sort_values('%').plot(x='top_topics', y='%',
                                    kind='barh', ax=ax, title='Distributuion des Topics')

    ax2 = fig.add_subplot(312)
    df_dcos.sort_values('%').plot(x='top_dcos', y='%', kind='barh',
                                  ax=ax2, title='Distribution des DCO', color='orange')

    plt.subplot(313)
    plt.imshow(wc, interpolation="bilinear")
    plt.title('Nuage des mots les plus présents dans ce cluster')
    plt.axis("off")
    
    n_docs = int(weight*len(topicmodel.corpus)/100)
    plt.suptitle(f"Analyse du cluster numéro " + str(x) +
                 " représentant "+str(weight) + "% du corpus ("+ str(n_docs) +" documents)")

    plt.show()
    print("------------------------")
    print('Document le plus représentatif du cluster sélectionné')
    print("------------------------")
    print(most_rpz_doc)
    print("------------------------")
    print("------------------------")
    print('Document le moins représentatif du cluster sélectionné')
    print("------------------------")
    print(less_rpz_doc)
    print("------------------------")


def plotClusterAnalysis2(x, topicmodel, clustermodel):
    """
    Affiche l'analyse d'un cluster version matplolib pour l'affichage des textes

    Args:
        x (int): Numéros du topic
        topicmodel (TopicModel): modèle de topic
        clustermodel (ClusterModel): modèle de cluster
    """
    wc, df_dcos, df_topics, most_rpz_doc, less_rpz_doc = ClusterAnalysis(
        x, topicmodel, clustermodel)

    fig = plt.figure(figsize=(7, 25))

    ax = fig.add_subplot(511)
    df_topics.sort_values('%').plot(x='top_topics', y='%',
                                    kind='barh', ax=ax, title='Distributuion des Topics')

    ax2 = fig.add_subplot(512)
    df_dcos.sort_values('%').plot(x='top_dcos', y='%', kind='barh',
                                  ax=ax2, title='Distribution des DCO', color='orange')

    plt.subplot(513)
    plt.imshow(wc, interpolation="bilinear")
    plt.title('Nuage des mots les plus présents dans ce cluster')
    plt.axis("off")

    plt.subplot(514)
    plt.text(0, 0.3, str(most_rpz_doc), fontsize=10, wrap=True)
    plt.title('Document le plus représentatif du cluster sélectionné')
    plt.axis('off')

    plt.subplot(515)
    plt.text(0, 0.3, str(less_rpz_doc), fontsize=10, wrap=True)
    plt.title('Document le moins représentatif du cluster sélectionné')
    plt.axis('off')

    plt.suptitle("Analyse du cluster numéro " + str(x))

    plt.show()


def plotCompareClusterAnalysis(x1, x2, topicmodel, clustermodel):
    """
    Affiche l'analyse de deux clusters afin de les comparer. Les élements présentés sont les mếmes que ceux qui sont dans ClusterAnalysis

    Args:
        x1 (int): Numéro du topic 1
        x2 (int): Numéro du topic 2
        topicmodel (TopicModel): modèle de topic
        clustermodel (ClusterModel): modèle de cluster
    """
    wc1, df_dcos1, df_topics1, most_rpz_doc1, less_rpz_doc1, weight1 = ClusterAnalysis(
        x1, topicmodel, clustermodel)
    wc2, df_dcos2, df_topics2, most_rpz_doc2, less_rpz_doc2, weight2 = ClusterAnalysis(
        x2, topicmodel, clustermodel)

    fig = plt.figure(figsize=(20, 25))

    ax = fig.add_subplot(5, 2, 1)
    df_topics1.sort_values('%').plot(
        x='top_topics', y='%', kind='barh', ax=ax, title='Distribution des Topics')

    ax21 = fig.add_subplot(5, 2, 2)
    df_topics2.sort_values('%').plot(
        x='top_topics', y='%', kind='barh', ax=ax21, title='Distribution des Topics')

    ax2 = fig.add_subplot(5, 2, 3)
    df_dcos1.sort_values('%').plot(x='top_dcos', y='%', kind='barh',
                                   ax=ax2, title='Distribution des DCOs', color='orange')

    ax22 = fig.add_subplot(5, 2, 4)
    df_dcos2.sort_values('%').plot(x='top_dcos', y='%', kind='barh',
                                   ax=ax22, title='Distribution des DCOs', color='orange')

    plt.subplot(5, 2, 5)
    plt.imshow(wc1, interpolation="bilinear")
    plt.title('Nuage des mots les plus présents dans ce cluster')
    plt.axis("off")

    plt.subplot(5, 2, 6)
    plt.imshow(wc2, interpolation="bilinear")
    plt.title('Nuage des mots les plus présents dans ce cluster')
    plt.axis("off")

    plt.subplot(5, 2, 7)
    plt.text(0, 0.3, str(most_rpz_doc1[:250]), fontsize=10, wrap=True)
    plt.title('Document le plus représentatif du cluster '+str(x1))
    plt.axis('off')

    plt.subplot(5, 2, 8)
    plt.text(0, 0.3, str(most_rpz_doc2[:250]), fontsize=10, wrap=True)
    plt.title('Document le plus représentatif du cluster '+str(x2))
    plt.axis('off')

    plt.subplot(5, 2, 9)
    plt.text(0, 0.3, str(less_rpz_doc1[:250]), fontsize=10, wrap=True)
    plt.title('Document le moins représentatif du cluster '+str(x1))
    plt.axis('off')

    plt.subplot(5, 2, 10)
    plt.text(0, 0.3, str(less_rpz_doc2[:250]), fontsize=10, wrap=True)
    plt.title('Document le moins représentatif du cluster '+str(x2))
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f"Comparaison de deux clusters: {x1} ({weight1}%)  et {x2} ({weight2}%) ")


def DocumentAnalysis(id_doc, topicmodel, clustermodel, mrv):
    """Permet de réaliser l'analyse d'un document

    Args:
        id_doc (str): Numéro de la déclaration
        topicmodel (TopicModel): topicmodel
        clustermodel (ClusterModel): modèle de cluster
        mrv(pd.DataFrame) : base de donnée MRveille

    Returns: 
        dist (pd.DataFrame): topics associés au document
        wc (WordCloud): Nuage de mot du cluster
        num_cluster (int) : Numero de clsuter associé
        doc (str): Document associé au numeros id_doc

    """
    mat = topicmodel.doc_topic_mat
    n_topics = topicmodel.model.num_topics
    mat['cluster'] = clustermodel.model.labels_
    df_doc = mat[mat['NUMERO_DECLARATION'] == id_doc]
    num_cluster = df_doc['cluster'].values[0]
    dist = df_doc.T.iloc[:n_topics-1].rename(
        columns={df_doc.index[0]: id_doc}).sort_values(id_doc, ascending=True).iloc[-5:]
    topic = []
    for elt in dist.index.map(lambda x: int(x[5:])):
        word = [t[0] for t in topicmodel.model.show_topic(elt)[:4]]
        word_str = ' '.join(word)
        title = 'Topic: '+str(elt) + ' (' + word_str + '...)'
        topic.append(title)
    dist.index = topic
    doc = display_doc(id_doc, mrv)

    analyse = Analysis(topicmodel,clustermodel)
    

    Clusters = mat.groupby('cluster')

    cluster = Clusters.get_group(num_cluster)
    analyse.cluster = cluster
    analyse.build_wc()
    wc = analyse.wc

    return (dist, num_cluster, doc, wc)


def plotDocumentAnalysis(id_doc, topicmodel, clustermodel, mrv):
    """Permet d'afficher l'analyse d'un document. C'est a dire d'afficher les élements renvoyés par DocumentAnalysis

    Args:
        id_doc (str): Numéro de la déclaration
        topicmodel (TopicModel): topicmodel
        clustermodel (ClusterModel): modèle de cluster
        mrv(pd.DataFrame) : base de donnée MRveille
    """
    dist, num_cluster, doc, wc = DocumentAnalysis(
        id_doc, topicmodel, clustermodel, mrv)
    print(doc)
    print("Appartient au cluster numero : "+str(num_cluster))
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.title('Nuage des mots les plus présents dans ce cluster')
    plt.axis("off")
    plt.title("Nuage de mot du cluster numéros " + str(num_cluster))
    plt.show()

    plt.figure(figsize=(10, 10))
    dist[id_doc].plot(kind='barh')
    plt.title("Distribution des topics sur le document sélectionné")
    plt.show()
