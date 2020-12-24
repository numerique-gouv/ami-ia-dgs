Livrable 3 : ALGORITHME DE REGROUPEMENT DES SIGNALEMENTS
========================================================

1. Déscription
-----------
Permet d'entraîner, de visualiser les modèles de regrouement des signalements d'incident et d'inférer de nouveaux documents.

Le modèle est constitué de deux étapes :

- Construction de thèmes sur le corpus avec un modèle de type  topic modelingLDA
- Utilisation de la distribution de ces thèmes pour regrouper les documents avec un modèle de type Kmeans

2. Utilisation:
------------

1 - Installer les bibliothèques nécessaires : pip install -r requirements.txt
2 - Modifier le fichier config.yaml pour qu'il corresponde à votre configuration
3 - Pour chacune des taches du procesus, le pré-traitment des données, l'entrainement des modèles, la visualisation et l'inférence de nouveau document, il y a un fichier config associé dans le dossier spécifique. Il faut alors le configurer en fonction de votre configuration et de vos choix algorithmiques.

### 2.1 Pre-traitement des données


Les pré-traitement des données permet d'extraire des caractéritiques textuelles pertinentes.
- Si vouis choisissez de construire l'extraction des termes méciaux, alors assurez vous d'avoir le fichier umls.csv
- Lancer python3 data_preparation/prepare_data.py

Le résultat est un fichier cleaned_data.pkl dans le dossier "path_to_save" de la config, contenant une dataframe pandas avec les champs suivants:

- **'NUMERO_DECLARATION', 'DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'FABRICANT', 'DCO_ID'** -> champs MRVeil
- **'text'** : concatenation de 'DESCRIPTION_INCIDENT' + '. ' + 'ETAT_PATIENT'
- **'text_lem'**: 'text' lemmatisé via spacy, désaccentué et en lettres minuscules --> liste de mots
- **'rake_kw'**: keywords détectés par multi_rake.Rake dans 'text', lemmatisés via spacy, désaccentué et en lettres minuscules --> liste de sous-phrases
- **'bigram'**: bigrammes détectés par gensim.models.Phrases dans 'text_lem'
- **'trigram'**: bigrammes détectés par gensim.models.Phrases dans 'bigram'
- **'med_term'**: termes détectés dans umls.csv, lemmatisés via spacy, désaccentué et en lettres minuscules --> liste de sous-phrases
- **'med_term_uniq'**: set de valeurs uniques par doc à partir de 'med_term'

/!\ Le préprocessing complet est très long (8/10h), car la lemmatisation de certains champs nécessite de retrier les mots clés 
(Rake donne les mots clés par score, et non par ordre d'apparition dans la phrase).

2.2 Entrainement
~~~~~~~~~~~~~~~~
**Objectifs:**

Les modèles de regroupements des signalements d'incidents répondent à deux objectifs :
    1 - La possibilité de réaliser une cartographie visuelle des signalements présents dans la base de donnée. C'est à dire être capable de décrire rapidement le contenu de la base de donnée MRveille, d'identifier l'importance de certains groupes de signalements et de réaliser des études semaine par semaine sur les nouveaux signalements. Ces études semaine par semaine, peuvent répondre à différent cas d'usage :
        - communication en interne
        - ordonancement intelligent 
    2 - L'identification des signalements n'appartennant pas à des clusters déjà existants afin de signaler un nouveau type de signalement.
 
 **Description**

 Afin de répondre aux objectifs présentés ci-dessus, nous avons choisi une solution qui se compose de deux briques principales :

 - Un topic modèle qui permet de capturer les informations présentes dans les champs de données textuelles : **Description Incident et Etat Patient** à travers la construction de thèmes. En effet, cette aproche probabiliste basée sur les distribution de Dirichlet permet d'associer un thème à chaque mot du corpus et ainsi de décrire un document par une distribution de thèmes. Dans le code proposé, vous avez le choix de différents paramètres :
    - le nombres de thèmes
    - le modèle utilisé : LDA, LDA-multi et HDP
    - le nombres de passes, c'est à dire le nombre de mise à jour des probabilités à postériori du modèles sur les données d'entrainement. En augmentant le nombre de passes on rend le modèle plus adapté aux données MRveille mais il perd alors en géralité.

 - La deuxième brique de notre solution est un modèle de clusterisation, qui se base alors sur la représentation thématique et prend en compte un certain nombres de variables catégorielles présentes dans les documents : le fabricant, la réference commerciale, la classification de l'incident. Il est également possible d'ajouter le résultat de l'inférence des premiers modèles : DCO_ID, TEF_ID, CDY_ID, TDY_ID et la GRAVITÉ. En tant qu'utisateur, vous pouvez choisir différent paramètres dans le fichier training/training_config.yaml: 
    - le modèle de clusterisation: Kmeans, dbscan, mélange de gaussienne
    - le nombre de cluster si le modèle le permet
    - des paramètres spécifiques à chaque modèle (cf training_config.yaml)

Notre architecture est une approche dites non supervisées qui est difficile à évaluer car les métriques mathematiques disponibles pour évaluer la qualité du regroupement d'incidents ne correspondent pas au besoin métier. En collaboration avec le metier, nous avons ainsi développé une métrique adaptée à notre problème. Cette métrique se base sur un clustering idéal qui correspond au la segmentation de la base de donnée MRveille sur les variables : DCO et TYPOLOGIE (effet, dysfonctionnement et conséquence).

Cette segmentation idéale, permet de construire environ 33 000  micro-clusters, nous allons alors comparer nos modèles qui forment des macro-clusters à cette segmentation à l'aide :
- d'un micro-score qui évalue à quel point un micro-cluster est regroupé dans un seul macro-cluster​
- d'un macro-score qui évalue à quel point un macro-cluster regroupe des micro-clusters semblables​

Une analyse complète de ces métriques est présentée dans le power-point suivant : https://starclay-my.sharepoint.com/:p:/g/personal/rquillivic_starclay_fr/EV2iMA2D9TxHrrouA5wRQp0Bb1PjJByLRqr7ApCBF99T5g?e=OwA4px

Actuelement, les modèles livrés sont : 
- topic modèle : multi-LDA, 153 topics
- cluster modèle : kmeans, 1500 clusters, avec les colonnes fabriant, DCO, TYPO et CLASSIFICATION concaténées à la représentation thématique

Avec les scores suivants :
- micro score : 0.24
- macro score : 0.82
- macro x micro : 0.2

Utilisation
~~~~~~~~~~~

    - Modifier le fichier training_config.yaml
    - lancer le script  python3 training/train.py
    - un dossier du nom précisé dans la config est alors créer et il contient l'ensemble des fichiers nécessaire pour sont insertion dans le pipeline d'industrialisation. 

Visualisation
~~~~~~~~~~~~~
    - Modifier le fichier analyse_config.yaml
    - Ouvrir le jupyter Notebook et executer l'ensemble des cellules pour analyser le fichier

Inference
~~~~~~~~~

Le fichier **inference/inference.py** contient une fonction permettant d'inférer les topics et les clusters de documents
au format MRV. Si le modèle de clustering nécessite des valeurs liées aux modèles de classification, ceux-ci doivent être inféré en amont.

Exploration 
~~~~~~~~~~~

Le dossier exploration regroupe des scripts qui ne sont pas mis en production mais qui retrace l'évolution du projet et présente les tentatives algorithmiques qui ont été ménées.


Auteurs
-------

* **Robin Quillivic**  
* **Cyril Poulet** 


