""" 
Auteur: 
    - Quillivic Robin, Data SCientist chez StarClay, rquilivic@starclay.fr
Description: 
    Fichier permettant de démarrer le serveur streamlit et d'afficher le démonstrateur dans le navigateur.
    Il repose sur l'ensemble des librairies présententes dans demonstrateur, son rôle et de synthètiser et de présenter
    les résultats de l'inférence pour un fichier pdf/csv chargé par l'utilisateur.

"""
# Importation des bibliothèques à utiliser
import warnings
warnings.filterwarnings('ignore')


import streamlit as st
import joblib
import pandas as pd
import json
import numpy as np
import tensorflow as tf


import os
import magic
import copy
import yaml


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demonstrateur import lib_extraction
from demonstrateur import lib_plot

from prediction_models.inference.inference_dco import DcoModel
from prediction_models.inference.inference_typologie import TypologieModel
from prediction_models.inference.inference_gravite import GraviteModel
from prediction_models.inference.encodeur import EncodeurModel


import logging.config
with open(os.path.join(os.path.dirname(__file__), 'log_conf.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)

logger = logging.getLogger('demonstrateur')

with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)

MODEL_PATH = os.path.abspath(config_data['Model']['path'])
PREPRO_PATH = os.path.abspath(config_data['Data']['prepro']['path'])


# load data for preprocessing
with open(os.path.abspath(os.path.join(PREPRO_PATH,'Colonnes.json')), 'r') as file:
        Colonnes = json.load(file)
with open(os.path.abspath(os.path.join(PREPRO_PATH,'mapping.json')), 'r') as file:
    mapping = json.load(file)

m = magic.Magic()


# load model
@st.cache(allow_output_mutation=True)
def load_model():
    """Fonction permettant de charger les modèles
    """
    # Chargement du modèle des DCO
    DCO_model = DcoModel()
    DCO_model.load(MODEL_PATH)

    # Chargement des modèles de typologie
    
    Effet_model = TypologieModel()
    Effet_model.typo = 'tef'
    Effet_model.load(MODEL_PATH)
    
    Dysfonctionnement_model = TypologieModel()
    Dysfonctionnement_model.typo = 'tdy'
    Dysfonctionnement_model.load(MODEL_PATH)
    
    Consequence_model = TypologieModel()
    Consequence_model.typo = 'cdy'
    Consequence_model.load(MODEL_PATH)
    

    # Modèle de Gravité
    Gravite_1234_model = GraviteModel()
    Gravite_1234_model.load(MODEL_PATH)
    
    Gravite_01_model = GraviteModel()
    Gravite_01_model.binaire = True
    Gravite_01_model.load(MODEL_PATH)

    # Encodeur
    classification_encoder = EncodeurModel()
    classification_encoder.load(MODEL_PATH)
    
    return(DCO_model, Dysfonctionnement_model, Consequence_model, Effet_model, Gravite_1234_model, Gravite_01_model, classification_encoder)


@st.cache
def load_data(file):
    """Permet de charger en cache les données au format csv avec Pandas.

    Args:
        file ([type]): chemin indiquant ou se trouve les données vsc

    Returns:
        data (pd.DataFrame): Dataframe contenant les données
    """
    data = pd.read_csv(file, index_col=0, encoding='utf8')
    return data


# Application
def main():
    """
    Lance le serveur streamlit et structure, réalise l'inference du fichier chargé et présente les résultats de manière structurée. 
    """
    st.title("DIRECTION GENERALE DE LA SANTE (DGS)")
    st.header("Chargement des modèles")
    DCO_model, Dysfonctionnement_model, Consequence_model, Effet_model, Gravite_1234_model, Gravite_01_model, classification_encoder = load_model()
    st.text('Done ! ')
    st.header("Chargement du fichier PDF")
    uploaded_file = st.file_uploader(
        "chargement d'un fichier", type=["csv", "pdf"])
    if uploaded_file is not None:
        file = copy.copy(uploaded_file)
        if 'PDF' in m.from_buffer(uploaded_file.read(1024)):
            df_data = lib_extraction.from_pdf_to_mrv_format(
                uploaded_file, Colonnes, mapping)
        else:
            df_data = load_data(file)
        if st.checkbox('Afficher les données'):
            st.text("Les données entrées sont les suivantes : ")
            try:
                st.table(df_data[['DM', 'DESCRIPTION_INCIDENT', 'FABRICANT',
                                  'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']])

            except:
                st.table(df_data[['DCO', 'DESCRIPTION_INCIDENT', 'FABRICANT',
                                  'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']])
                X = df_data[['DESCRIPTION_INCIDENT', 'FABRICANT',
                             'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']].fillna('')

        # df_data.to_csv('./data_test/test.csv')
        st.header("Résultat de l'inférence")
        st.subheader("Inférence du DCO")
        st.text(
            'Le modèle chargé possède une justesse pondérée (balanced accuracy) de 54%')
        try:
            st.text('La bonne réponse est : '+str(df_data['DCO'].values))
        except:
            st.text('La bonne réponse est : '+str(df_data['DM'].values))

        try:
            X = df_data[['DESCRIPTION_INCIDENT', 'FABRICANT',
                         'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']].fillna('')
            X['LIBELLE_COMMERCIAL'] = X['LIBELLE_COMMERCIAL'] + '. '+df_data['DM']
        except:
            X = df_data[['DESCRIPTION_INCIDENT', 'FABRICANT',
                         'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']].fillna('')
        X = X.applymap(str)  # Pour éviter les erreurs de type
        # faire la prédiction
        Proba = DCO_model.predict(X)
        char_data = lib_plot.create_chart_data(Proba)
        # afficher le resultat
        st.table(char_data[['class_name', 'proba']][:5])  # tableau

        X = df_data[['FABRICANT', 'CLASSIFICATION', 'DESCRIPTION_INCIDENT',
                     'ETAT_PATIENT', 'ACTION_PATIENT']].fillna('NON RENSEIGNE')
        X = X.applymap(str)  # Pour éviter les erreurs de type
        X['CLASSIFICATION'] = classification_encoder.transform(
            X['CLASSIFICATION'])
        X_dys = Dysfonctionnement_model.transform(X)
        X_dys = np.reshape(X_dys, (X_dys.shape[0], 1, X_dys.shape[1]))
        st.subheader("Inférence de la TYPOLOGIE")
        st.subheader('DYSFONCTIONNEMENT')
        st.text('Le modèle chargé possède un f1-sample  de 47%')
        st.text('La bonne réponse est :' +
                str(df_data['TYPE_DYSFONCTIONNEMENT'].values))
        Proba = Dysfonctionnement_model.predict(X_dys)
        char_data = lib_plot.create_chart_data_typo(
            Proba, lib_plot.get_name_dysfonctionnement, le=lib_plot.Config().le_dys)
        # afficher le resultat
        st.table(char_data[['class_name', 'proba']][:5])

        st.subheader("CONSEQUENCE DYSFONCTIONNEMEMT")
        st.text('Le modèle chargé possède un f1-sample  de 81%')
        st.text('La bonne réponse est : ' +
                str(df_data['CONSEQUENCE_DYSFONCTIONNEMENT'].values))
        X_cons = Consequence_model.transform(X)
        X_cons = np.reshape(X_cons, (X_cons.shape[0], 1, X_cons.shape[1]))
        Proba = Consequence_model.predict(X_cons)
        char_data = lib_plot.create_chart_data_typo(
            Proba, lib_plot.get_name_consequence, le=lib_plot.Config().le_consequence)
        # afficher le resultat
        st.table(char_data[['class_name', 'proba']][:5])

        st.subheader("EFFET ")
        st.text('Le modèle chargé possède un f1-sample  de 67%')
        st.text('La bonne réponse est :' + str(df_data['TYPE_EFFET'].values))
        X_effet = Effet_model.transform(X)
        X_effet = np.reshape(X_effet, (X_effet.shape[0], 1, X_effet.shape[1]))
        Proba = Effet_model.predict(X_effet)
        char_data = lib_plot.create_chart_data_typo(
            Proba, lib_plot.get_name_effet, le=lib_plot.Config().le_effet)
        # afficher le resultat
        st.table(char_data[['class_name', 'proba']][:5])

        st.subheader("Inférence de la GRAVITÉ 5 catégories")
        st.text('La bonne réponse est :'+str(df_data['GRAVITE'].values))
        st.text(
            'Le modèle chargé possède une justesse pondérée (balanced accuracy) de 48%')
        X = df_data[['CLASSIFICATION', 'DESCRIPTION_INCIDENT', 'ETAT_PATIENT',
                     'ACTION_PATIENT', 'FABRICANT']].fillna('')
        X = X.applymap(str)

        Proba = Gravite_1234_model.predict(X)
        char_data = lib_plot.create_chart_data_Gravite(
            Proba, lib_plot.dec_di_multi)
        st.table(char_data[['class_name', 'proba']][:5])

        st.subheader("Inférence de la GRAVITÉ Binaire")
        st.text(
            'Le modèle chargé possède une justesse pondérée (balanced accuracy) de 90%')
        st.text('et un f1 score binaire pour la classe critique de 54% ')

        Proba = Gravite_01_model.predict(X)
        char_data = lib_plot.create_chart_data_Gravite(
            Proba, lib_plot.dec_di_bin)
        st.table(char_data[['class_name', 'proba']][:5])


if __name__ == '__main__':
    main()
