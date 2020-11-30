Exploration
===========

Ce dossier constitue la mémoire du projet, il retrace l'ensemble des essaies réalisés au cours du projet. Il a deux objectifs:

- Rendre compte du travail effectué
- Permettre aux nouvelles exploration d'être fructueuse en évitant de refaire des tests déjà réalisé.

Il est constitué de deux sous dossier:

- Livrable 2: Scripts et Notebooks concernant les algorithmes de classification du Livrables 2
- Livrable 3: Scripts et Notebooks concernant les algorithmes de regroupement et de détection d'anomalie du Livrables 3

1. Livrable 2
-------------

Ce dossier constitue la mémoire du livrable 2, les pistes qui ont fonctionné et celles qui n'ont pas aboutie
Il est composé d'un dossier de Notebook et d'un dossier de Scripts

Nota :  Les pistes qui ont ponctionnées, sont codés de manière plus documenté et structurée dans le dossier **inférence** qui correspond au Livrable 2.

1.1 Notebook
~~~~~~~~~~~~

Les notebooks dans ce dossier retracent les recherches menées pour mettre au point la solution de classification de texte dans le projet avec la DGS sur les trois cycles suivantes :
- Le DCO
- La Typologie
- La gravité

**Exploration:**

- _Exploration_pdf_psig_ :  
  - Comment extraire et structurer l'information depuis les pdf ?
  - Quelle est la qualité des données dans cette base de données ?
- _Exploration_declaration_mrv_:
  - Compréhesion de la problématique
  - Analyse quantitative des données
  - Analyse qualitative des données

**DCO:**

- _DCO_MRV_Approche_naive_ :
  - TFIDF + SVM
  - SVD ? Random forest ? NaiveBayes ?
- _DCO_MRV_ Approche_1_1_ : 
  - Comment améliorer l'approche naive ?
  - Grid search sur SVM
  - Comment probabiliser le SVM
  - Comment améliorer le tfidf ?
  - Les embeddings sont-ils une solitions ?
- _DCO_MRV_Approche_2_1_ : 
  - Construction de différents tfidf et étude de leurs impacts sur les performances du SVM
  - Meilleurs résultats obtenus avec 4 tfidf sur 4 varaiables en entrée
- _DCO_MRV_Approche_2_2_ :
  - Finetuning de XG_boost avec optuna
  - Résultats peu encourageants
- _DCO_MRV_Approche_3 : 
  - Optimisation des paramètres du tfidf avec optuna
- Obtention du meilleur score balanced accuracy = 0.72

**Gravité:**

- _GRAVITE_APPROCHE_ML_1_ :
  - Application du SVM gagnant (pipeline DCO) au problème de la gravité (binaire et multi)
  - Obtention de résultats proches des Transformers proposés par Boris
- _TYPO_GRAVITE_APPROCHE_ML_1_2 :
  - Optimisation des paramètres du tfidf et obtention des performances suivantes:
  - Justesse : 0.7354525503757399 Justesse pondéré:  0.7567361506788588  f1_weighted :  0.7461994757033263

**Typologie:**

- _TYPO_MRV_APPROCHE_ML_1:_

  - Analyse des variables effets et dysfonctionnements
  - Application du SVM gagnant (pipeline DCO) au problème de la Typologie
  - Obtention de résultats très décevants --> le problème de la typologie est plus difficile que prévu.
- _TYPO_MRV_APPROCHE_ML_2_:_
  - Utilisation d'un bagging
  - visualisation en matrice de confusion
  - Résultats décevant, calcul de la précision à k
  - Approche multilabel envisagé
- _TYPO_MRV_APPROCHE_cosine:_
  - Utilisation des vecteurs de BERT Finetunés pour comparer les vecteurs des intutulés et des documents.
  - Résultat peu probant car il y a des intitulés trop proches les uns des autres.
- _TYPO_MRV_APPROCHE_deep_:_
  - exploration de la librairie ktrain
  - utilisation de differents embbeding
  - test de l'ensemble des modèles disponibles dans ktrain:NBSVM DistilBert etc.

- _TYPO_EFFT_APPROCHE_multilabel_ :
  - Traitement du problème des effets en le considérant comme multilabel
  - Test de différentes approches OneVSRest, OnevsOne, ClassifierChain
  - Test de différents modèles : XgBoost, LSTM, SVM, NaiveBAyes, librairie ktrain, BERT Transformer
  - Test de différentes représentations :TFIDF, EMBEDING, CountVectorizer
  - Résultats des approches dans le tableau suivant : https://starclay-my.sharepoint.com/:x:/g/personal/rquillivic_starclay_fr/EZPS3DrBBQ9MrZskrcwKVAEBGsLY61W089kd8RFvIEirjg?e=GL9jYV
- _TYPO_EFFT_LSTM_ :
  - Test de différentes architectures mêlant réseaux récurrents et self attention :
        - Quel est l'impact du drop out ?
        - Rajouter des couches augmentent-ils les performaces ?
        - L'utilisation de réseaux bidirectionnel est-elle pertinente ?
        - Une couche d'attention est-elle utile ?
        - Attention is all we need, really ?
        - Utilisation des embeddings 
        - Concaténation des modèles sur différentes entrées ?
- _TYPO_EFFET_LSTM_suite:_
  - Essaie des modèles de type RCNN
  - Pas d'amélioration des résulats vis à vis du bi-LSTM

Les notebooks sur les conséquences et les Type de dysfonctionnement présente les mêmes approches mais avec des données différentes.

1.2 Scripts:
~~~~~~~~~~~

- Classifier_Chain:
  - Permet d'entrainer un ClassifierChain combiné avec un XGboost sur les Classes : TYPE_EFFET

- finetunexgb:
  - Utilisé pour Finetuner un classifier XGboost avec la bibliothèque Optuna sur la variable TYPE_EFFET

- make_data_split_again:
  - Permet de séparer les données de manière stratifiée dans le cas multilabel
  - Permet de créer un dataset train et test au format .pkl

- Matrice_confusion:
  - Affiche une matrice de confusion pour un modèle de svm sur les DCO

- Normalized_text_DCO:
  - Lemmatise les données textuelles de la base MRVeil et les sauvegarde dans une colonne de la DataFrame

- Sentence_Embeding:
  - Utilisation de la bibliothèque sentence transformer pour encodé les déclarations
  - Sauvegarde des vecteurs train et test au format npy pour la variable TEF_ID 

- svm_emb:
  - Construction d'un modèle SVM à partir des sentence Embeding construit avec Camembert

- svm_model:
  - Construction d'un modèle SVM + TF-IDF pour la variable TEF_ID
  - print du f1 sample

- Xgboost_tunning:
  - Construction d'un modèle XGBoost + TF-IDF pour la variable DCO
  - Optimisation avec Optuna
  - Sauvegarde de l'étude au format csv

1.3 Neural_NLP Classifier
~~~~~~~~~~~~~~~~~~~~~~~~~

 Dossier contenant les fichier de configuration pour les différents test réalisés avec la bibliothèque: https://github.com/Tencent/NeuralNLP-NeuralClassifier

1.4 Camembert Vanilla
~~~~~~~~~~~~~~~~~~~~~

cf ReadMe Associé

1.5 Xtrem Classification et Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cf ReadMe Associé

1.6 Zero-shot learning
~~~~~~~~~~~~~~~~~~~~~~

cf ReadMe Associé

1.7 Architecture exotique de Bert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cf ReadMe Associé