Exploration Livrable 3
======================

Ce dossier constitue la mémoire du livrable 3, les pistes qui ont fonctionné et celles qui n'ont pas abouties. Il est composé d'un dossier de Notebook et d'un dossier de Scripts

Nota :  Les pistes qui ont fonctionnées, sont codés de manière plus documentés et structurées dans le dossier **regroupement** qui correspond au Livrable 3.

1. Notebooks:
-------------

- _Exploration_Livrable_3:_
  - Approche Naive pour la clusteurisation :Kmeans, LDA
  - Visualisation de ces deux approches

- _Autoencodeur_V1:_
  - Ce Notebook à pour but de présenter la construction d'un autoencodeur permettant de détecter les signalements d'un nouveau type. Pour ce faire nous utilisons l'exemple des prothèses PIP comme jeu de données de validation. La méthode est la suivante.
    - Nous excluons les declarations concernants les PIP et créant ainsi deux dataset : PIP et NO_PIP
    - Avec le dataset PIP nous créons un jeu de test (0.2) et de train qui nous permet d'entrainer notre autoencodeur
    - Nous construisons et entrainons un autoencodeur avec la Loss MSE (Mean Squared Error) et le TFIDF du texte en entrée
    - Nous définition un seuil d'anomalie en calculant la MSE sur le jeu de test PIP
    - Nous calculons le mse pour les données des prothèse PIP et assignons la valeure 1 à celle qui sont au dessus du seuil
    - Nous calculons le taux de détetion d'anomalie : Nombre d'anomalie détecté par l'auto-encoder/nombre de signalements PIP

- _LDA_V1:_ 
Développement du pipeline pour LDA
  - Suppression des stop word
  - Concaténation des Colonnes Description Incident et Etat Patient
  - Calcul des 3-gramms, + extraction du fabricant + Lemmatisation + extraction du vocabulaire medicale
  - Application d'un count vectorizer
  - application d'un LDA
  - Algorithme sur la distance de basée sur les thèmes

2. Scripts
----------

Une présentation de chaque script est disponible au début de chacun d'un.

