Module de préparation des données 
=================================

Description
-----------
Lors de la gestion de données textuelles il est essentielle de les traiter correctement pour obetenir de bon résultats. Ce pé-traitement passe les étapes:
    1 - Gestion des lignes vides et des caratères spéciaux
    2 - Concaténation des colonnes interessantes (Description incident et Etat patient)
    3 - tokenisation et lemmatisation
    4 - post-processing des lemmes :
        - mise en minuscule
        - suppression des stop_words
        - supression des nombres, des dates
        - supression des mot de moins de deux caratères
        - supression des accents

    5 - Extraction des mots clefs
    6 - Construction des bigrames et des trigrames

Le résultat est un fichier cleaned_data.pkl dans le dossier "path_to_save" de la config, contenant une dataframe pandas avec les champs suivants:

- **'NUMERO_DECLARATION', 'DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'FABRICANT', 'DCO_ID'** -> champs MRVeil
- **'text'** : concatenation de 'DESCRIPTION_INCIDENT' + '. ' + 'ETAT_PATIENT'
- **'text_lem'**: 'text' lemmatisé via spacy, désaccentué et en lettres minuscules --> liste de mots
- **'rake_kw'**: keywords détectés par multi_rake.Rake dans 'text', lemmatisés via spacy, désaccentué et en lettres minuscules --> liste de sous-phrases
- **'bigram'**: bigrammes détectés par gensim.models.Phrases dans 'text_lem'
- **'trigram'**: bigrammes détectés par gensim.models.Phrases dans 'bigram'
- **'med_term'**: termes détectés dans umls.csv, lemmatisés via spacy, désaccentué et en lettres minuscules --> liste de sous-phrases
- **'med_term_uniq'**: set de valeurs uniques par doc à partir de 'med_term'

L'ensemble de ces colonnes peuvent être combinées pour entrainer le modèle de thèmes

Fonctionnement 
--------------

La préparation des données est basé sur des librairie open source que nous détaillons ici : 
- NLTK  https://www.nltk.org/ : bibliothèque historique du NLP, permet un prototyage rapide mais ses modèles sont moins adaptés au français
- spacy https://spacy.io/ : lemmatisation/tokenisation/NER basé sur des modèles de type bi-LSTM entrainé sur différents corpus français
- gensim https://radimrehurek.com/gensim/ : bibliothèque permettant de réaliser des transformations/vectoisation des donénes textuelles mais également de les néttoyer, par exemple de désacentuer les textes.


Pour executer le prétraitement, lancez la commande python3 data_preparation/prepare_data.py
