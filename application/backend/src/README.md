API TRAITEMENT DES SIGNALEMENTS DGS
=====================================

Cette api propose 2 types de fonctions:

- des fonctions de prédiction pour des nouveaux fichiers
- des fonctions de visualisation associées au topic modelling et au clustering

Elle peut fonctionner sous 2 modes:

- **avec front**: toutes les routes sont activées
- **sans front**: seules les routes de prédiction sont activées

Elle est protégé par une authentification login/mdp à fournir en header de toutes les requètes.


1. Contenu
----------

- **main_server.py**: script principal de l'API
- **config.yaml**: fichier de configuration
- **config_env.yaml**: surcharge de la config par des variables d'env
- **logging.ini**: config du logging
- **password.json**: dictionnaire de logons/mdp hashés
- **backend_utils**: fichiers de fonctions annexes (voir backend_utils/README.md)
- **clustering**: fichiers de fonctions de clustering (voir clustering/README.md)
- **prediction**: fichiers de fonctions de prédiction (voir prediction/README.md)
- **data**: dossier data (voir plus bas)
- **users**: code associé à l'authentification

Dossiers copiés au build du docker:

- **regroupement/training** : classes de training des modèles de topic modelling et de clustering
- **regroupement/inference** : classes d'inférence des modèles de topic modelling et de clustering
- **regroupement/data_preparation** : classes de data_prep des modèles de topic modelling et de clustering
- **regroupement/utils.py** : foctions annexes des modèles de topic modelling et de clustering
- **prediction_models/inference**: classes d'inférence des modèles de classif
- **prediction_models/config.yaml**: config des modèles de classif


2. Configuration
----------------

    app:
      log_level: DEBUG                          # log level
      activate_front: True                      # choix du mode
      clean_results_in_min: 10                  # période de nettoyage des données de prédiction
      password_file: password.json              # fichier contenant les login/mdp hashés
      
    models:
      path: "data/models"                       # chemin vers les modèles de classif
    clusters:
      path: "data/clusters"                     # chemin vers les modèles de topic et clusters
    data:
      mrv:
        path: "data/mrv"                        # chemin vers les données mrv


3. Data
-------

- dans *src/data/models* : ensemble des fichiers sauvegardés
- dans *src/data/mrv*:
    - **déclaration_mrv_complet.csv** : export MRVeil complet
    - **colonnes.json** : fichier utilisé pour extraire les données des documents inputs
    - **mapping.json** : fichier utilisé pour extraire les données des documents inputs
    - **referentiel_consequence.csv** : fichier de réferentiel
    - **referentiel_dispositif.csv** : fichier de réferentiel
    - **referentiel_consequence_dysfonctionnement.csv** : fichier de réferentiel
    - **referentiel_consequence_effet_connus.csv** : fichier de réferentiel
- dans *src/data/clusters* :
    - **cluster** : dossier contenant le modèle de clusterisation
    - **LDA** : dossier contenant le modèle de topic modelling
    - **training_config.yaml** : fichier de configuration utilisé à l'entrainement des modèles
    - **topics_mat.npy** : distances inter-topic (généré par le backend)
    - **topics_documents.json** : appartenance des docs aux topics (généré par le backend)
    - **clustermodel_scores.json** : scores du modèle de clustering (généré par le backend)
    - **cluster_model_pca.csv** : pca 2d du modèle de clustering (généré par le backend)


4. endpoints 
------------

### Authentification

Par défaut, un seul utilisateur est connu : 'dgs_admin', dont le mdp hashé est stocké dans password.json.

Pour créer des utilisateurs, le endpoint suivant est disponible. Seul dgs_admin peut ajouter de nouveaux utilisateurs.
Ceux-ci sont ajoutés à password.json avec le nouveau mot de passe hashé (Les mots de passe ne sont jamais stockés en clair).

- **/dgs-api/users - POST** : création d'un nouvel utilisateur

        request.json : {'username': ..., 'password': ...}
        -> 201, {"username": ...}

Pour l'ensemble des requètes, username/password doit être envoyé dans le header HTTP_AUTH


### Prédiction

Ces endpoints sont actifs tout le temps.

Si config['app']['activate_front'] est à True, les fonctions sont formattées en sortie par backend_utils/antpro_formatting.py

#### Prédiction

Les requètes d'entrées doivent contenir des fichiers dans request.files, qui est un dictionnaire nom_du_fichier: descripteur

La sortie contient une clé permettant de récupérer **une seule fois** les résultats de la requète, au format que l'on veut (json, csv, excel)

Formatting de sortie antpro:

    {
        file_1: [
            { 
                model_name: nom_du_modele,
                prédictions: {
                    columns = [
                        {
                            'title': model_name,
                            'dataIndex': 'cat',
                            'key': 'cat',
                        },
                        {
                            'title': 'Probabilité',
                            'dataIndex': 'proba',
                            'key': 'proba',
                        }
                    ],
                    datasource: [
                        {
                            'key': i,
                            'cat': k,
                            'proba': round(v * 100., 2)
                        },...
                    ]
                }
            }, ...]
         ...
         last_results_key: cle_str
     }
              

- **/dgs-api/predict/dco - POST** : prédiction de la DCO

        -> modele_name: DCO
        
- **/dgs-api/predict/dysfonctionnement - POST** : prédiction du dysfonctionnement

        -> modele_name: dysfonctionnement
        
- **/dgs-api/predict/consequence - POST** : prédiction de la conséquence

        -> modele_name: consequence
        
- **/dgs-api/predict/effet - POST** : prédiction des effets

        -> modele_name: effet
        
- **/dgs-api/predict/gravite - POST** : prédiction de la gravité

        -> modele_name: gravité_ordinale et gravité_binaire
        
- **/dgs-api/predict/clustering - POST** : même que all_models car le clustering nécessite toutes les autres prédictions

        -> modele_name: topics et cluster
        
- **/dgs-api/predict/all_models - POST**: prédictions de tous, clustering inclus

#### Récupération des prédictions

- **/dgs-api/predict/last_results/<last_results_key> - GET** : récupération des résultats en json
- **/dgs-api/predict/last_results/<last_results_key>/<output_format> - GET**: récupération des résultats en csv ou excel

        -> output_format: csv | excel
        -> le contenu du fichier est dans answer.content


### Visualisation

Ces endpoints ne sont actifs que si config['app']['activate_front'] est à True.

Toutes les fonctions de visualisation sont formattées en sortie par backend_utils/antpro_formatting.py

#### Endpoints documents

- **/dgs-api/documents/all_ids - GET**

    -> {'doc_ids': doc_ids, 'nb_docs': len(doc_ids)}
    
- **/dgs-api/documents/<docid_or_docname>/content - GET**

    ->  {
            "doc_id": doc_id,
            "Numero de déclaration": doc_id,
            "nom": doc_name,
            "DCO": dco,
            "Description incident": des,
            "Etat patient": etat 
         }
         
- **/dgs-api/documents/<docid_or_docname>/topics - GET**

    -> {'doc_id': doc_id, 'topics': [{'topic': titre de topic,
                                      'value': score,
                                      'label': score,
                                      'tooltip': 'mot1, mot2, ... mot10')}]}
    
- **/dgs-api/documents/<docid_or_docname>/cluster - GET**

    -> {'doc_id': doc_id, 'cluster': int}
    
- **/dgs-api/documents/<docid_or_docname> - GET**

    ->  {
            "doc_id": doc_id,
            "Numero de déclaration": doc_id,
            "nom": doc_name,
            "DCO": dco,
            "Description incident": des,
            "Etat patient": etat,
            'topics': [{'topic': titre de topic,
                        'value': score,
                        'label': score,
                        'tooltip': 'mot1, mot2, ... mot10')}],
            'cluster': int
         }
         
#### Endpoints Topic model

- **/dgs-api/topics/model/nb_topics - GET**

    -> {'nb_topics': nb_topics}
    
- **/dgs-api/topics/model/coherence - GET**

    -> {'coherence_score': float}
    
- **/dgs-api/topics/model/distances - GET**

    -> {'distances_matrix': list(list de float)) [nb_topics x nb_topics]}
    
- **/dgs-api/topics/model/pca - GET**

    -> page html
    
- **/dgs-api/topics/model - GET**

    -> {'nb_topics': nb_topics,
        'coherence_score': float,
        'distances_matrix': list(list de float)) [nb_topics x nb_topics)}
    
#### Endpoints Topics

- **/dgs-api/topics/<topic_ind>/weight - GET**

    -> {'topic_id': int(topic_ind), 'weight': float}
    
- **/dgs-api/topics/<topic_ind>/documents - GET**

    -> {'topic_id': int(topic_ind), 'documents': [{'doc_name': str, 'topic_score': float}]}

- **/dgs-api/topics/<topic_ind>/wordcloud - GET**

    -> {'topic_id': int(topic_ind), 'wordcloud': [{'word': str, 'id': int, 'weight': float}]}
    
- **/dgs-api/topics/<topic_ind> - GET**

    -> {'topic_id': int(topic_ind), 
        'weight': float,
        'documents': [{'doc_name': str, 'topic_score': float}],
        'wordcloud': [{'word': str, 'id': int, 'weight': float}]
        }

#### Endpoints Cluster model

- **/dgs-api/clusters/model/nb_clusters - GET**

    -> {'nb_clusters': nb_clusters}
    
- **/dgs-api/clusters/model/scores - GET**

    -> {'scores': {metric_name: float}}
    
- **/dgs-api/clusters/model/distances - GET**

    -> {'distances_matrix': list(list de float)) [nb_topics x nb_topics]}
    
- **/dgs-api/clusters/model/pca - GET**

    -> {'pca': {
            'clusters_ind':[int],
            'X': [float],
            'Y': [float],
            'weights': [float],
        }}
    
- **/dgs-api/clusters/model - GET**

    -> {'nb_clusters': nb_clusters,
        'scores': {metric_name: float},
        'distances_matrix': list(list de float)) [nb_topics x nb_topics],
        'pca': {
                'clusters_ind':[int],
                'X': [float],
                'Y': [float],
                'weights': [float],
            }
        }
    

#### Endpoints clusters

- **/dgs-api/clusters/<cluster_ind>/weight - GET**

    -> {'cluster_id': int, 'weight': {
                                        'weight': %age,
                                        'nb_docs': int
                                    }}
                                    
- **/dgs-api/clusters/<cluster_ind>/documents - GET**

    -> {'cluster_id': int,
        'documents': [{'doc_name': str, 'document_similarity': float}]}

- **/dgs-api/clusters/<cluster_ind>/topics - GET**

    -> {'cluster_id': int,
        'topics': [{'topic': titre de topic,
                    'value': score,
                    'label': score,
                    'tooltip': 'mot1, mot2, ... mot10')}]}

- **/dgs-api/clusters/<cluster_ind>/dcos - GET**

    -> {'cluster_id': int,
        'dcos': [{'topic': titre de topic,
                  'value': score,
                  'label': score,
                  'tooltip': 'mot1, mot2, ... mot10')}]}

- **/dgs-api/clusters/<cluster_ind>/wordcloud - GET**

    -> {'cluster_id': int,
         'wordcloud': [{'word': str, 'id': int, 'weight': float}]}

- **/dgs-api/clusters/<cluster_ind> - GET**

    -> {'cluster_id': int,
        'documents': [{'doc_name': str, 'document_similarity': float}],
        'weight': {
                    'weight': %age,
                    'nb_docs': int
                },
         'topics': [{'topic': titre de topic,
                    'value': score,
                    'label': score,
                    'tooltip': 'mot1, mot2, ... mot10')}],
         'dcos': [{'topic': titre de topic,
                  'value': score,
                  'label': score,
                  'tooltip': 'mot1, mot2, ... mot10')}],
         'wordcloud': [{'word': str, 'id': int, 'weight': float}]}

