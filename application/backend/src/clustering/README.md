Package CLUSTERING
==================

Package contenant les fonctionnalités associées au topic modelling et au clustering.
Les modèles sont chargés une fois dans un singleton, puis utilisé pour l'analyse ou la prédiction.

IMPORTANT:

- les clusters sont numérotés de 0 à X, de la même manière, dans toutes les sources de données
- les topics, en revanche...
    - sont numérotés de 0 à X par le topicmodel, sans ordre particulier
    - de 1 à X+1 PAR POIDS DECROISSANT dans la visualisation gensim
    
N'ayant pas la main sur la visualisation gensim, l'ensemble de l'affichage est aligné sur sa numérotation.
Il y a donc une certaine gymnastique dans le code pour permettre le mapping correct des données.

1. Contenu
----------

- **clustering_models.py** : Classe singleton pour charger les modèles
- **prediction.py** : fonction de clusterisation des fichiers inconnus
- **analysis.py** : fonctions de visualisation des topics et clusters


2. data nécessaire
------------------

dans *src/data/clusters* :

- **cluster** : dossier contenant le modèle de clusterisation
- **LDA** : dossier contenant le modèle de topic modelling
- **training_config.yaml** : fichier de configuration utilisé à l'entrainement des modèles
- **topics_mat.npy** : distances inter-topic (généré par le backend, ordonné décroissant)
- **topics_documents.json** : appartenance des docs aux topics (généré par le backend, ordonné décroissant)
- **clustermodel_scores.json** : scores du modèle de clustering (généré par le backend)
- **cluster_model_pca.csv** : pca 2d du modèle de clustering (généré par le backend)


3. Update des modèles
---------------------

- vider le dossier *src/data/clusters*
- copier les nouveaux modèles dedans (dossiers *cluster* et *LDA* + *training_config.yaml*)
- redémarrer le service

