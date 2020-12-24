Package PREDICTION
==================

Package contenant les fonctionnalités associées à la classification.

1. Contenu
----------

- **models.py** : chargement et prédiction
- **prediction_context.py** : contextualisation des prédictions (récupération des labels associés)


2. data nécessaire
------------------

- dans *src/data/models* : ensemble des fichiers sauvegardés
- dans *src/data/mrv*:
    - **déclaration_mrv_complet.csv** : export MRVeil complet
    - **colonnes.json** : fichier utilisé pour extraire les données des documents inputs
    - **mapping.json** : fichier utilisé pour extraire les données des documents inputs
    - **referentiel_consequence.csv** : fichier de réferentiel
    - **referentiel_dispositif.csv** : fichier de réferentiel
    - **referentiel_consequence_dysfonctionnement.csv** : fichier de réferentiel
    - **referentiel_consequence_effet_connus.csv** : fichier de réferentiel


3. update des modèles
---------------------

- vider le dossier *src/data/models*
- copier les nouveaux modèles dedans  (modèles généré via le dossier prediction_models, voir prediction_models/README.md)
- redémarrer le service

