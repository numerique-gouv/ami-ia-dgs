Interface graphique de démonstration pour le projet MS10 - DGS/StarClay (version 1.0)
=====================================================================================

1. Présentation
---------------

Cette Interface graphique légère a pour but de présenter les modèles de classification de texte pour inférer les variables:

- DCO_ID
- Typologie
- Gravité

Des données de test sont disponibles sur demande sur demande pour 3 formats : xml, pdf et csv

Cette interface est un démonstrateur, elle n'a pas pour but de rentrer en production mais de simplement communiquer sur les algorithmes et leurs potentielles applications.

2. Installation
---------------

2.1 Configuration requise
~~~~~~~~~~~~~~~~~~~~~~~~~

- python 3.6 ou supérieur
- pip à jour (pip install --upgrade pip)
- Avoir le package ptyhon prediction_models dans le même répertoire.

Pour le bon fonctionnement de l'application, il est recommandé d'avoir au moins 8Go de RAM et un CPU avec 2.5Ghz.

2.2 instruction d'instalation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Télécharger le fichier et Le décompresser à l'endroit de votre choix (C:\votre\chemin\vers\le\dossier\)

2) Ouvrir Anaconda Prompt et taper : cd C:\votre\chemin\vers\le\dossier\

3) Ouvrir le fichier config.yaml et modifier les champs pour qu'il corresponde à votre architecture

4) Une fois  dans le dossier demonstrateur

- Lancer la commande: conda activate starclay_plumber (l'environnement dédié)
- Lancer la commande: pip install -r requirements.txt
- Lancer la commande: streamlit run main.py

Une fenêtre de votre navigateur se lance et l'application s'affiche.


3. Utilisation
--------------

Dans un navigateur, à l'adresse suivante, http://localhost:8501/ l'interface graphique de l'application permet de charger un pdf ou un fichier csv et d'observer les prédictions des modèles pour la DCO, la TYPOLOGIE et la GRAVITÉ.


4. Utilisation via le terminal
------------------------------

Afin de réaliser des batchs de tests sur plusieurs fichiers pdf ou XML, le script **result_excel.py** permet d'inférer plusieurs documents en une seul fois et de stocker les résultats au format excel. Il faut indiquer le chemin vers le dossier contenant les dossiers xml ou pdf dans la variable: /Data/démonstrateur.

4.1 instruction
~~~~~~~~~~~~~~~

Après avoir installer les requierements.txt, vous pouvez indiquer dans le fichier **result_excel.py** le chemin d'accès d'un dossier contenant des fichiers PDF et/ou XML encodé ou non en base64 :

- Dans le fichier de configuration modifier la ligne 9 demonstrateur/file_path en indiquant le chemin du dossier concerné.
- lancer le Script : python result_excel.py
- Les résultats sont stockés dans le fichier résultat.xlsx
