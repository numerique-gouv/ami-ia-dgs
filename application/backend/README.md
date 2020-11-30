DOCKER API TRAITEMENT DES SIGNALEMENTS DGS
============================================


1. Build
--------

Utiliser *./build_docker_image.sh* génère l'image **starclay/dgs_backend:latest**

Pour cela, on copie temporairement des dossier d'autres parties du repo pour les intégrer au contexte docker.


2. Run
------

Le fichier *run_backend.sh* donne un exemple de lancement du docker.

Point importants:

- la data nécessaire n'est pas incluse dans l'image, il faut donc la monter via --volume sur */dgs_backend/data*
- seul l'utilisateur 'dgs_admin' est par défaut dans src/password.json. Il est conseillé de monter src/paasword.json dans le docker
pour assurer une persistence des utilisateurs créés de redémarrage en redémarrage du conteneur.
- par défaut, l'image se lance en mode "front". Pour la lancer en mode "prod", il faut surcharger la variable d'env *APP_ACTIVATE_FRONT*
- le port interne est le 5000
- les données sont stockées pendant un temps configurable en attendant un export csv ou excel. Le temps est configurable via *APP_CLEAN_RESULTS_IN_MIN* (tous les résultats plus vieux que ce délai sont supprimés).


3. Update des données
---------------------

voir la description dans src/README.md, et les instructions dans src/clustering/README.md et src/prediction/README.md