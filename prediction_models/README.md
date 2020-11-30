Livrable 2 : *ALGORITHME D'INFERENCE DE LA DCO, TYPOLOGIE et GRAVITE*
=====================================================================

1.  Description
---------------

1.1  Aperçu
~~~~~~~~~~~

Le dossier **prediction_models** contient le livrable 2, c'est à dire la brique d'intelligence artificielle qui s'intégrera coté ANSM pour prédire la DCO, la Typologie et la Gravité des signalements qui arrivent dans le flux des signalements au format XML.

Il permettra également d'entraîner les modèles pour des mises à jours régulières mensuelles ou bi-mensuelles.

Enfin, il contient l'historique du code et donc des approches testées.

1.2 Details
~~~~~~~~~~~

Une fois le script lancé, les modèles sont stockés dans le dossier *models_date-du-jour* et sont entraînés sur la totalité de la base de donnée MRveille.

A partir d'un fichier csv contenant les données de la base MRveille, il est possible de reproduire les résultats annoncés et sauvegardés dans le fichier performances.csv.

Afin de construire ce livrable, nous avons choisi de construire les modèles de manière indépendante pour chaque problème de classification. Ainsi, l'ensemble des opérations de nettoyage, de traitement de données, d'entraînement et d'évaluation sont construites et différentes pour chaque problème. Le code se trouve dans chacun des fichier suivant:
- /training
  - **model_dco.py**
  - **model_typologie.py**
  - **model_gravite.py**
  - **model_gravite_binaire.py**

Le fichier **global_training.py** permet selon les arguments choisis de réaliser l'entraînement et le stockage des modèles

Le fichier **OrdinalClassifier.py** est utilisé dans la gravité à 5 classes pour prendre en compte  l'ordre des classes dans la classification.

Enfin, le dossier **inference** permet d'uniformiser les modèles suivant leur origines, il créer les classes de chaque modèle avec une méthode *load*, une méthode *predict*. Il est utilisé dans l'application/backend.

2. Utilisation
--------------

2.1 Le fichier config.yaml
~~~~~~~~~~~~~~~~~~~~~~~~~~

Afin de réaliser l'entraînement des modèles, il y a deux champs à modifier dans le fichier config:

- training/savepath: Le lieu de sauvegarde des modèles
- training/data/mrv/filepath : l'adresse vers le lien du fichier de la base MRveille.

Les autres champs sont à configurer pour les autres fonctionnalités.

2.2 Reproduire les résultats et obtenir les modèles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1 - Se placer dans l'environnement virtuel dédié

2 - Installer les dépendances : pip install -r requirements.txt

    2.1 S'assurer avec un pip freeze que les bonnes versions ou versions supérieures sont installées.

    2.2 Si ce n'est pas le cas, tapez pip install --upgrade pip et relancer le pip install -r requiremements.txt

3 - Modifier le fichier config.yaml comme expliqué dans le 2.1. (Variable DATA_PATH dans le fichier de config.yaml)

4 - Lancer le script : python3 main.py (ou python main.py)

5 - Les modèles sont stockés au format  .sav et .h5 dans le dossier models et les performances dans le fichier performances.csv

Pour suivre l'avancé du du script, il suffit de suivre les informations écrites dans *inference.log*.

2.2 Configuration technique
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le temps de calcul peut être assez long (>1h), nous conseillons d'avoir au minimum :
    - 8 Gb de mémoire RAM pour l'entraînement
    - 4 coeurs CPU (i5)

3. Lecture des performances
---------------------------

3.1 Choix des metriques
~~~~~~~~~~~~~~~~~~~~~~~

Nous avons choisi différents indicateur de performances pour les différentes variables en fonction des besoins et de la structure des données :

- La **DCO** et la **GRAVITE 5 classes** sont évalués selon la balanced_accuracy (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html). Cette mesure étant définie comme la moyenne des recalls (cf presentation), elle permet de tenir compte du déséquilibre des classes et de nous pénaliser autant sur les classes peu fréquentes que sur les classes fréquentes.

- La **TYPOLOGIE** est évaluée selon le f1-samples (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html). C'est la mesure classique des problèmes en multi-label.

- La **GRAVITÉ binaire** est évalué selon le f1-binary sur la classe critique (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.htm). Cette mesure nous permet de ce concentrer sur la détection de la classe critique.

Pour comprendre plus en détails les métriques utilisées nous avons préparé un support de présentation disponible en suivant ce lien:
 https://starclay-my.sharepoint.com/:p:/g/personal/rquillivic_starclay_fr/EfyfEgs2qkRHuUkM5K3u_UIBP0bSmPtbm1f00aESCxAkgg?e=4jvmf1

3.2 Commentaire sur les performances (au 28/08/2020)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **DCO**, bacc = 0.54, les classes rares sont mal détectés et pénalise ainsi la moyenne des recall.
- **GRAVITE** :
  - 5 classes, b_acc = 0.53, même explication que pour le DCO
  - Binaire, f1_binary = 0.48, une classe critique sur 2 n'est pas détecté, ce score est améliorable
- **Typologie**:
  - Type de dysfonctionnement, f1_sample = 0.47, le grand nombre de classe disponible nous pénalise (1500) ce qui explique ce résultat un peu moins bon pour le dysfonctionnement
  - Conséquence de dysfonctionnement, f1_sample =0.81, généralement bien détecté
  - Type d'effet, f1_sample = 0.67, généralement bien détecté
