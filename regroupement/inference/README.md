Module d'inférence pour le clustering
=====================================

Description
----------

Ce module est un module très simple qui repose sur un seul fichier inference.py qui permet d'inférer un nouveau document qui est au format MRveille.
Il permet de réaliser les étapes suivantes:
- chargement du fichier
- chargement des modèles de thèmes et de clusters (préciser dans le script)
- préparation du fichier et des données textuelles (prepare_data.py)
- construction de la représentation thématique du document
- ajout des colonnes contenant les données catégorielles du document
- inférence du cluster d'appartenance

Ce module est réutilisé dans le démonstrateur (L4).
