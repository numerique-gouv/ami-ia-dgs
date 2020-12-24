"""
Auteur : Quillivic Robin, Data scientist chez StarClay, rquillivic@starclay.fr

Description: 
    Entrainement des modèles de clusterisation 

"""
import os
import sys
import yaml
import logging
import json

import numpy as np
import numpy.matlib as matlib
import pandas as pd
import matplotlib.pyplot as plt

import joblib
from sklearn import metrics


from sklearn.cluster import FeatureAgglomeration, DBSCAN, OPTICS, KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import mixture
from sklearn.mixture import BayesianGaussianMixture


path_to_regroupement = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(1, os.path.join(path_to_regroupement))
from utils import loading_function

from train_topic import TopicModel
import joblib

with open(os.path.join(os.path.dirname(__file__), 'training_config.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

#model_name = config['cluster']['model']['name']
#n_cluster = config['cluster']['model']['n_cluster']

#conf_path = config['path_to_save']
#try_name = config['config_name']
#SAVE_PATH = os.path.join(conf_path, try_name)


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

class ClusterModel:

    def __init__(self,try_name,config_dict,save_dir):
        self.n_cluster = None
        self.model = None
        self.dist = None
        self._logger = logging.getLogger(self.__class__.__name__)
        
        self.try_name = try_name
        self.config_dict = config_dict
        self.save_dir = os.path.join(save_dir, 'cluster')
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, X):
        """
        Entrainement du modèle

        Args:
            X (np.array): Données d'entrainement des cluster
        """
        
        
        model_name = self.config_dict['model']['name']
               
        if model_name == 'kmeans':
            self.n_cluster = self.config_dict['model']['kmeans']['n_cluster']
            agglo = KMeans(self.n_cluster)
            agglo.fit(X)
            self.model = agglo
            
        elif model_name == 'DBSCAN': 
            eps =  self.config_dict['model']['DBSCAN']['eps']
            min_samples =  self.config_dict['model']['DBSCAN']['min_samples']
            X_trans = StandardScaler().fit_transform(X)
            agglo = DBSCAN(eps=eps,min_samples=min_samples)
            agglo.fit(X_trans)
            self.model = agglo
            
        elif model_name == "HDBSCAN":
            import hdbscan
            min_cluster_size = self.config_dict['model']['HDBSCAN']['min_cluster_size']
            min_samples= self.config_dict['model']['HDBSCAN']['min_samples']
            cluster_selection_epsilon = self.config_dict['model']['HDBSCAN']['cluster_selection_epsilon']
            cluster_selection_method = self.config_dict['model']['HDBSCAN']['cluster_selection_method']
            X_trans = StandardScaler().fit_transform(X)
            agglo = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True, min_samples=min_samples, cluster_selection_epsilon = cluster_selection_epsilon, cluster_selection_method=cluster_selection_method)
            agglo.fit(X_trans)
            self.model = agglo 
            
        elif model_name == 'gaussianMixture' : 
            n_components = self.config_dict['model']['gaussianMixture']['n_components']
            agglo = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
            agglo.fit(X)
            self.model = agglo
            self.model.labels_ = agglo.predict(X)
            
        elif model_name == 'bayesianGaussianMixture' : 
            n_components = self.config_dict['model']['bayesianGaussianMixture']['n_components']
            agglo = BayesianGaussianMixture(n_components=n_components, covariance_type='full')
            agglo.fit(X)
            self.model = agglo
            self.model.labels_ = agglo.predict(X)
            
        elif model_name =='agglomerative': 
            n_clusters = self.config_dict['model']['agglomerative']['n_clusters']
            linkage = self.config_dict['model']['agglomerative']['linkage']
            metric = self.config_dict['model']['agglomerative']['metric']
            agglo = AgglomerativeClustering(n_clusters=n_clusters,
                                    linkage=linkage, affinity=metric)
            agglo.fit(X)
            self.model = agglo

        else:
            print('model not supported')

    def save(self,X):
        """Sauvegarde du modèle et des données

        Raises:
            ValueError: le chemin spécifié n'existe pas
        """
        path = self.save_dir
        if not os.path.exists(path):
            self._logger.error(f'path {path} does not exists')
            raise ValueError('file does not exist')
        
        file_path = os.path.join(self.save_dir, self.try_name+'.sav')
        if self.model is not None:
            joblib.dump(self.model, file_path)
            self._logger.info(f'Modèle de cluster enregistré dans {file_path}')
        result = pd.DataFrame(data=X)
        result['label'] = self.model.labels_
        self.model.features = result
        features_path = os.path.join(self.save_dir, self.try_name+'_features.json')
        result.to_json(features_path)
        self._logger.info(f"Features d'entrainement enregistrées dans {features_path}")

    def load(self, filename):
        """charge un modèle de clustering

        Args:
            filename (str): nom du modèle à charger

        Raises:
            ValueError: le fichier n'existe pas
            ValueError: le chargement du fichier à échouer
        """
        self._logger.info(f'Loading from {os.path.join(self.save_dir, filename)}')
        # Dict
        path = os.path.join(self.save_dir, filename+'.sav')
        if type(path) == str:
            if not os.path.exists(path):
                self._logger.error(f'path {path} does not exists')
                raise ValueError('file does not exist')
        try:
            self.model = joblib.load(path)
            if os.path.exists(os.path.join(self.save_dir, self.try_name+'_features.json')):
                # used for retrain
                self.model.features = pd.read_json(os.path.join(self.save_dir, self.try_name+'_features.json'))
            self._logger.info(f'loaded object from {path}')
        except Exception as e:
            self._logger.error(f'Error loading {path}: {e}')
            raise ValueError(f'{path} : {e}')

    def predict(self, X):
        return self.model.predict(X)

    def predict_with_score(self, X):
        return self.model.predict(X), self.model.score(X)

    def soft_clustering_weights(self, X, fuzziness_param=2):
        """
        from https://towardsdatascience.com/confidence-in-k-means-d7d3a13ca856

        Function to calculate the weights from soft k-means

        :param X: Array of data. shape = N x F, for N data points and F Features
        :param fuzziness_param: fuzziness of the clustering. Default 2
        """

        Nclusters = self.model.cluster_centers_.shape[0]
        Ndp = X.shape[0]
        Nfeatures = X.shape[1]

        # Get distances from the cluster centres for each data point and each cluster
        EuclidDist = np.zeros((Ndp, Nclusters))
        for i in range(Nclusters):
            EuclidDist[:, i] = np.sum((X - np.matlib.repmat(self.model.cluster_centers_[i], Ndp, 1)) ** 2, axis=1)

        # Denominator of the weight from wikipedia:
        invWeight = EuclidDist ** (2 / (fuzziness_param - 1)) * matlib.repmat(
            np.sum((1. / EuclidDist) ** (2 / (fuzziness_param - 1)), axis=1).reshape(-1, 1), 1, Nclusters)
        Weight = 1. / invWeight

        return Weight

    def compute_score(self, X,save=False):
        """
        Calcul des scores d'évaluation du modèles : 
        - silhouette_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
        - calinski_harabasz_: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
        - davies_bouldin_score : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html

        Args:
            X (np.array): Données d'entraînements
        """
        self.silhouette_score = metrics.silhouette_score(
            X, self.model.labels_, metric='euclidean')
        self.calinski_harabasz_score = metrics.calinski_harabasz_score(
            X, self.model.labels_)
        self.davies_bouldin_score = metrics.davies_bouldin_score(
            X, self.model.labels_)
        
        if save:
            res = {'config': str(self.config_dict),
                   'silhouette_score': str(self.silhouette_score),
                    'calinski_score': str(self.calinski_harabasz_score),
                    'daves_score': str(self.davies_bouldin_score)}
            with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
                json.dump(res, f)

    def build_cluster_features(self,topicmodel,data,save=False):
        n = topicmodel.model.num_topics
        cols = self.config_dict['model']['add_columns']
        #vecteur du topic model
        X = topicmodel.doc_topic_mat.iloc[:, :n-1].values
        #Ajout des collones
        if len(cols):
            X_new = add_col(data, cols, X,save=False)
        
        result = pd.DataFrame(data=X)
        result['label'] = self.model.labels_
        self.model.features = result
        if save :
            features_path = os.path.join(self.save_dir, self.try_name+'_features.json')
            result.to_json(features_path)
            self._logger.info(f"Features d'entrainement enregistrées dans {features_path}")

    def build_cluster_centers(self):
        """calcul les centres des clusters par une moyennes des points dans le cluster

        Args:
            topicmodel (gensim.model): topic model associé au modèle de clustering
        """
        mat = self.model.features
        Clusters = mat.groupby('label')
        cluster_center = []
        for elt in Clusters.groups.keys():
            df = Clusters.get_group(elt).iloc[:,:-1]#on ne prend pas la colonne label
            cluster_center.append(df.mean(axis=0).values)
            
        self.model.cluster_centers_ = np.asarray(cluster_center)
        
    def build_dist_mat(self):
        """
        Construction de la matrice des distances cosinus inter-cluster

        Returns:
            dist (np.array): matrice des distance inter cluster
        """
        
        self.dist = pairwise_distances(
            self.model.cluster_centers_, metric='cosine')
        return self.dist

    def plot_dist_mat(self, title="Distance cosinus inter cluster"):
        """
        Affichage de la matrice des distances inter cluster

        Args:
            title (str, optional): titre de la. Defaults to "Distance cosinus inter cluster".
        """
        fig, ax = plt.subplots(figsize=(18, 14))
        data = ax.imshow(self.dist, cmap='RdBu_r', origin='lower')
        plt.title(title)
        plt.colorbar(data)

    def get_furthest(self):
        """
        Retourne les deux clusters les plus éloignés l'un de l'autre. 

        Returns:
            c (np.array): numéro des deux cluster les plus loin
        """
        maxi = self.dist.max()
        c = np.where(self.dist == maxi)
        return c

    def get_closest(self):
        """
        Retourne les deux clusters les plus proches. 

        Returns:
            c (np.array): numéro des deux cluster les plus proches
        """
        mini = np.min(self.dist[np.nonzero(self.dist)])
        c = np.where(self.dist == mini)
        return c

    def get_evaluation_tables(self, data):
        """Permet de construire les micro et macro score pour un modèle de clustering en se basant sur les données:


        Args:
            data (pd.DataFrame): base de données MRveille

        Returns:
            df_table_micro (pd.DataFrame): table contenant les scores des micro clusters, c'est à dire leur taux de cohérence avec les macro cluster
            df_table_macro (pd.DataFrame): table contenant les scores des macro clusters, c'est à dire leur taux de cohérences sur les variables: DCO, fabricant et Typologie
        """
        # calcul des label des micro_clusters (environ 30 000)
        data = data.fillna(' ')
        data['label'] = data['DCO']+' '+data['TEF_ID'].map(str)+' '+data['CDY_ID'].map(str)+' '+data['TDY_ID'].map(str)+' '+data['FABRICANT']
        le = preprocessing.LabelEncoder()
        data['micro_label'] = le.fit_transform(data['label'])
        # ajout des cluster du modèles à évaluer
        data['cluster']= self.model.labels_
        
        # calcul du score de cohérence des mirco labels
        micro_group = data.groupby(by='micro_label')
        micro_scores, weights, macro_clusters = [], [], []
        DCOs,tefs,tdys,cdys,fabs = [], [], [], [], []

        for g in micro_group.groups.keys():
            df_micro = micro_group.get_group(g)
            macro_cluster = df_micro.groupby('cluster').count()['DCO'].sort_values(ascending=False).index[0]
            weight = len(df_micro)
            micro_score = len(df_micro[df_micro['cluster']==macro_cluster])/weight # Nombre d'élements égales au cluster majoritaire dans le micro_cluster
            micro_scores.append(micro_score)
            weights.append(weight)
            macro_clusters.append(macro_cluster)
            DCOs.append(df_micro['DCO'].iloc[0])
            tefs.append(df_micro['TEF_ID'].iloc[0])
            tdys.append(df_micro['TDY_ID'].iloc[0])
            cdys.append(df_micro['CDY_ID'].iloc[0])
            fabs.append(df_micro['FABRICANT'].iloc[0])

        # Construction de la table avec ppour chaque micro cluster, son label, son macron label, son score de cohérnce, son poid et ses informations 
        df_table_micro = pd.DataFrame()
        df_table_micro['micro_label'] =list(micro_group.groups.keys())
        df_table_micro['macro_label'] = macro_clusters
        df_table_micro['weight'] = weights
        df_table_micro['score'] = micro_scores
        df_table_micro['DCO'] = DCOs
        df_table_micro['TEF_ID'] = tefs
        df_table_micro['TDY_ID'] = tdys
        df_table_micro['CDY_ID'] = cdys
        df_table_micro['FABRICANT']=fabs

        # calcul des scores de cohérence des macros_label
        macro_group = data.groupby(by='cluster')
        macro_dco_scores, macro_typo_scores, macro_fabricant_scores, weights = [],[],[],[]

        for g in macro_group.groups.keys():
            df_macro = macro_group.get_group(g)
            weight = len(df_macro)
            weights.append(weight)
            micro_label_list = df_table_micro[df_table_micro['macro_label']==g]['micro_label'].tolist() #liste des micro cluster dans le clusters
            if len(micro_label_list)==0 :
                macro_dco_score,macro_typo_score,macro_fabricant_score = 0,0,0 # Si le cluster n'est jamais présent, c'est un cluster poubelle..
            else :
                df_table_micro_label = df_table_micro[df_table_micro['micro_label'].isin(micro_label_list)]
                macro_dco_score = 1/len(set(df_table_micro_label['DCO'].tolist()))
                macro_typo_score = 1/3*(1/len(set(df_table_micro_label['TEF_ID'].tolist()))+1/len(set(df_table_micro_label['TDY_ID'].tolist()))+1/len(set(df_table_micro_label['CDY_ID'].tolist())))
                macro_fabricant_score = 1/len(set(df_table_micro_label['FABRICANT'].tolist()))
            #Ajout des valeures à la listes
            macro_dco_scores.append(macro_dco_score)
            macro_typo_scores.append(macro_typo_score)
            macro_fabricant_scores.append(macro_fabricant_score)

        
        df_table_macro = pd.DataFrame()  
        df_table_macro['macro_label']= macro_group.groups.keys()
        df_table_macro['weight'] = weights
        df_table_macro['dco_score']= macro_dco_scores
        df_table_macro['typo_score']= macro_typo_scores
        df_table_macro['fabricant_score'] = macro_fabricant_scores

        self.df_table_micro = df_table_micro
        self.df_table_macro = df_table_macro

        return df_table_micro,df_table_macro

    def compute_evaluation_score(self,data,save=False):
        """Calcul les score de cohérence micro et macro
        Dans le cas macro, on associe un poids de 0.5 pour le DCO, 0.3 pour la TYPO et 0.2 pour le fabricant
        Dans le cas micro, il s'agit d'une moyenne pondérée des scores de cohérence des micro clusters

        Args:
            data (pd.DataFrame): base de donnée MRveille
            save (bool, optional): Faut-il sauver les scores. Defaults to False.

        Returns:
            micro_score (float): moyenne pondérée des scores de cohérences des micro clusters
            macro_score (float): moyenne pondérée des scores de cohérences des macro clusters
        """
        df_table_micro,df_table_macro = self.get_evaluation_tables(data)

        micro = df_table_micro['weight'] * df_table_micro['score']
        micro_score = micro.sum()/df_table_micro['weight'].sum()

        macro = df_table_macro['weight']*(0.5*df_table_macro['dco_score'] + 0.3*df_table_macro['typo_score'] + 0.2*df_table_macro['fabricant_score'])
        macro_score = macro.sum()/df_table_macro['weight'].sum()

        self.micro_score = micro_score
        self.macro_score = macro_score

        if save:
            res = {'config': str(self.config_dict),
                   'macro_score': str(self.macro_score),
                    'micro_score': str(self.micro_score)
                    }
            with open(os.path.join(self.save_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(res, f)
        
        return micro_score, macro_score


    def predict_mrv(self,X, data,save=False):
        clusters_weights = self.soft_clustering_weights(X)
        cluster_result = pd.DataFrame(clusters_weights, columns=[str(i) for i in range(len(clusters_weights[0]))])
        #calcul du cluster associé à chaque document
        L = cluster_result.iloc[:,:-1].apply(np.argmax,axis=1)
        # calcul de son poids
        weight = cluster_result.iloc[:,:-1].apply(np.max,axis=1)
        resultat = data
        # On ajoute les colonnes label, cluster et cluster weight à la base de donnée et on le sauvegarde
        resultat['label'] = self.model.labels_
        resultat['cluster'] = L
        resultat['cluster_weight'] = weight
        if save :
            resultat.to_csv(os.path.join(self.save_dir, 'mrveille_with_cluster.csv'))

    def update(self, new_data,delta=0.001):
        i=0
        for featureset in new_data:
            #Calcul des distances au centroids
            distances = [np.linalg.norm(featureset - self.model.cluster_centers_ [centroid]) for centroid in range(len(self.model.cluster_centers_))]
            weights = self.soft_clustering_weights(new_data)
            # Si le point est suffisament proche d'un centroid existant
            # Si la proba la plus grande est superieur au seuil alors un cluster existant correspond
            #print(max(weights[i]))
            if max(weights[i]) > delta:
                # selection du cluster associé ie le plus proche
                #label = distances.index(min(distances))
                label = np.argmax(weights[i])
                # mise à jour de la liste des labels
                np.append(self.model.labels_,label)
                # mise à jour de la liste des features
                new_feature = np.append(featureset,label)
                self.model.features.append(dict(zip(new_feature, self.model.features.columns)),ignore_index=True)
                # mise à jour du centre du cluster concerné
                self.model.cluster_centers_[label] = self.model.features.groupby('label').get_group(label).iloc[:,:-1].mean(axis=0)
                self._logger.info(f'Un nouveau document a été ajouté au cluster {label}')
            else:
                # on ajoute un cluster
                self.model.n_clusters = self.model.n_clusters + 1
                label = self.model.n_clusters
                np.append(self.model.labels_,label)
                # mise à jour des features
                new_feature = np.concatenate(featureset,label)
                self.model.features.append(dict(zip(new_feature, self.model.features.columns)),ignore_index=True)
                np.append(self.model.cluster_centers_,featureset)
                self._logger.info(f'Un nouveau cluster a été crée, le modèle compte désormais {self.model.n_clusters}')
            i+=1    






if __name__ == "__main__":
    import shutil

    current_dir = os.path.dirname(__file__)
    config_file = 'training_config.yaml'
    
    save_dir_topic = config['path_to_save'] #'/home/robin/Nextcloud/strar_clay/GitLab/Annexe/L3'
    name = '26_10_2020_text_lem'
    
    topic_dir = os.path.join(save_dir_topic,name)
    topic_config_file = os.path.join(topic_dir, 'training_config_serveur.yaml')

    if len(sys.argv) == 2:
        config_file = sys.argv[1]

    if not os.path.isabs(config_file):
        config_file = os.path.abspath(os.path.join(current_dir, config_file))

    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        
    with open(topic_config_file, 'r') as stream:
        config_topic = yaml.load(stream, Loader=yaml.FullLoader)

    try_name = config['config_name']
    save_dir = os.path.join(config['path_to_save'], try_name)
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(os.path.join(current_dir, save_dir))

    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(config_file, save_dir)

    clustermodel = ClusterModel(try_name, config['cluster'], save_dir=save_dir)

    filename = os.path.join(config['data']['path'], config['data']['filename'])
    if not os.path.isabs(filename):
        filename = os.path.abspath(os.path.join(current_dir, filename))
    
    # CHargement du topic model
    with open(topic_config_file, 'r') as stream:
        config_topic = yaml.load(stream, Loader=yaml.FullLoader)
    
    # Loading
    topicmodel = TopicModel(name, config_topic['topic'], 
                                        save_dir=topic_dir)
    topicmodel.load(name)
    
    clustermodel.topicmodel = topicmodel
    n = topicmodel.model.num_topics
    n_lignes = None
    
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
    X = add_col(data,columns,X,save=True)
    #
    #X = X[:,-5:]
    
    clustermodel.train(X)
    clustermodel.compute_score(X,save=True)
    clustermodel.compute_evaluation_score(data,save=True)
    
    clustermodel.save()
    
