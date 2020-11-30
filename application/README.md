Livrable 4 : INTERFACE DE TRAITEMENT DES SIGNALEMENTS
=====================================================

1. Description
--------------

Backend du démonstrateur du livrable 4. Pour le front-end, voir projet dédié.

Le backend est déployé dans du docker. Voir backend/README.md pour savoir comment compiler l'image et la lancer, et comment updater les données.

Voir backend/src/README.md pour des explications plus précise sur le code, son fonctionnement et les données nécessaires.


Pour créer des utilisateurs, utiliser le script create_backend_users.py.

IMPORTANT : Si le password admin est perdu, il faut recréer son compte.
Pour cela:

- editer backend/src/main_server pour enlever l'auth sur la fonction new_user (commenter le décorateur @auth.login_required l 379)
- lancer le serveur. Si en docker, monter le fichier backend/src/password.json
- faire une requète de création d'utilisateur avec comme nouvel utilisateur 'dgs_admin': "new_mdp"
- une fois validé, vérifier que password.json a bien enregistré le nouveau hash
- commiter password.json sur le dépôt git


2. Authors
----------

* **Robin Quillivic**  
* **Cyril Poulet**  

3. License
----------

Ce projet est privé.
This project is completely private.

