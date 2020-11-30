Module d'entraînement des modèles de classification
===================================================

Description
-----------
Ce module nous permets d'entrainer les modèles de classification pour les variables :
    1 - DCO
    2 - Type de dysfonctionnements
    3 - Type d'effets
    4 - Type de consequences
    5 - Gravité

Pour chacune de ces variables, il y a un script associé qui permet de réaliser:
- la préparation des données
- l'entrainement du modèle sur l'ensemble du jeux de données
- la séparation en jeu de train et de test pour évaluer les performances sur les données citoyennes professionnelles

Préparation des données
~~~~~~~~~~~~~~~~~~~~~~~
La préparation des données est équivalente pour les 5 problèmes de classification:
1 - Nettoyage des données textuelles pour chaque champs:
    - DCO, ['DESCRIPTION_INCIDENT', 'FABRICANT','REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']
    - TYPO,  ["DESCRIPTION_INCIDENT", "ETAT_PATIENT","FABRICANT", 'ACTION_PATIENT', 'LIBELLE_COMMERCIAL']
    - GRAVITE,  ['DESCRIPTION_INCIDENT','ETAT_PATIENT', 'FABRICANT', 'CLASSIFICATION']
2 - Vectorisation via un pipeline de tf-idf pour chaque colonne. Les hypermaramètres du tf-idf ont été oiptimisés via optuna, voici un jeu de paramètres d'exemples :
    - analyzer='word',
    - min_df=5,
    - ngram_range=(1, 1),
    - stop_words=STOP_WORDS,
    - max_features=2977,
    - norm='l2'
3 - Encodage via un encoder des variables catégorielles qui est ensuite sauvegardé pour gérer l'inférence de nouveaux documents
4 - Pour les modèles de type deep-learning, nous rajoutons une opération de réduction de dimension (SVD,1000 composantes)

Séparation  en jeu de donnée d'entrainement et de test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
La création de ces jeux de données est une étape indispensable pour évaluer la qualité des modèle. Toutefois, elle s'est avérée particulièrement diffcile dans notre cas pour plusieurs raison:
    1 -  Le numéro de déclaration n'est pas un identifiant unique, donc il ne peut pas servire de séparation sinon le jeu de train et de test serait trop proche l'un de l'autre et les performances serait faussées
    2 - Le problème de classification de la typologie s'est avérée être un problème multilabel, il a donc été nécessaire de construire un jeu de donnée multilabel associé à chaque sous classe de la typologie pour pouvoir évaluer les modèles.
    3 - De manière générale, les classes étaient très désequilibrées et nosu devons donc adapté nos métriques et otre séparation train/test a cette configuration.

Pour dépasser ces difficultés avons utilisé les lignes de codes suivantes :
- train_index, test_index = next(GroupShuffleSplit(random_state=1029).split(df_n, groups=df_n['DESCRIPTION_INCIDENT'])) , la descriptionde l'incident a servi de cléf de séparation pour les DCO
- MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1029), pour gérer  création d'un train/test set multilabel.


Modèle
~~~~~~
3 types de modèles ont été mis en place:
    1 -  Support Vector Machines (SVM) : Cet algorithme repose sur deux idées clés qui permettent de traiter des problèmes de discrimination non linéaire, et de reformuler le problème de classement comme un problème d'optimisation quadratique. Cette approche est très adapté a la classification de textes car l’implémentation faite par LIBLINEAR qui utilise un kernel linéaire est très performante sur des vecteurs de grande dimensions (comme ceux produit par les modèles vectoriels Tf-Idf).
    2 - bi-LSTM : réseaux de neurones récurrents du type LSTM bidirectionnels qui tiennent compte de l’aspect séquentiel des mots dans une phrase (et des paragraphes dans un document) et qui sont donc particulièrement adaptés aux données textuelles.
    3 - Ordinal SVM: une petite astuce qui permet de tenir compte de l'ordre des classes et ainsi gérer le problème de la gravité. http://medicalresearch.inescporto.pt/breastresearch/data/_uploaded/publications/2005JaimeNN.pdf


Utilisation
-----------

Il faut lancer le script global_training.py et récupérer le dossier sauvegarder pour le placer dans l'architecture du L4. Ce dossier contient les modèles au format .sav et les encodeurs ainsi qu'un fichier json sur les résulats.


Limites et remarques
--------------------

Il est important de noter que les modèles proposés sont des modèles pouvant être améliorés et il sera intéressant d'y repasser un peu de temps pour améliorer les premier résultats.