Module d'entraînement pour le topic modelling et le clustering
==============================================================
 Comme expliqué dans le readme général, nous avons construit une solution pour regrouper les signalements avec deux briques :
 - topic modèle
 - cluster modèle

 Nous détaillons ici comment effectuer l'entrainement de ces modèles.

Description
-----------
Ce dossier se compose des fichiers suivant qui dépendent tous du fichier de configuration training_config.yaml :
- train : permet de réaliser un entrainement complet et de générer les fichiers utiles pour la visualisation et le L4
- train_topic.py: ilpermet d'entrainer seulement un topic model 
- optimize_topic.py : permet d'optimiser le topic modèle sur le nombre de thèmes et sur le choix des colonnes (basé sur optuna)
- train_cluster: entraine un modèle de clustering à partir d'une représentation thématique existante
- optimize_cluster: permet d'optimiser le nombre de clusters d'un modèle de clustering (basé sur optuna)

Lors d'un entrainement, le dosser indiquer dans le training_config['config_name'] est crée et se structure comme suit:
- config.yaml, la configuration associé à cet entrainement
- cluster:
    - name.sav: le modèle de clustering
    - result.json: les scores silhouette, bouldin_davies et calinski du modèle
    - evaluation_result.json: le micro_score et le macro score du modèle
    - name_data.json : prediction du cluseting sur la base de donnée MRveille
- LDA :
    - name.dict: le dictionaire de terme du modèle
    - name.mm: le corpus du modèle
    - name.model.id2word: le lien entre les mots et leur identifiant
    - name.mm.index: l'index du corpus
    - name.model : le modèle de thème
    - name.expElogbeta.npy: les poids associé au modèle de thème
    - name.model.state, name.model.state.sstats.npy: fichier associé au modèle LDA
    - name_result.json : le score  de cohérence du modèle
    - name_training.log: le fichier de log associé à l'entrainement
    - name.json: donnée lié à la visualisation en 2D des thèmes
    - name.html: fichier html de la visualisation
    - name.pkl: matrice thème document
    - name_evaluation.json: balanced accuracy d'un svm entrainé sur les donnée thématique pour classifier les DCO

Utilisation
-----------
Afin de réentrainer les modèles, la procédure à suivre est :
 - s'assurer d'avoir les bonnes biblithèques (sinon pip install -r requierements)
 - configuger le fichier training_config.yaml
 - lancer le script associé à la tache associé (python3 train.py par exemple)


Limites
------
Les scripts présents possèdent quelques limites :
- les modèle de thèmes supporté sont seulement ceux dans gensim
- si vous choisissez un modèle de thème hirarchique alors, la visualisation 2D ne sera pas disponible
- les modèles de cluster disponibles sont seulement ceux présents dans la librairie scikit-learn


