# -*- coding: utf-8 -*-

""" 
Auteur: 
    - Quillivic Robin, Data SCientist chez StarClay, rquilivic@starclay.fr
Description: 
    Fichier regroupant l'ensemble des fonctions utiles pour l'inférence d'un grand nombre de pdf 
    et la structuration du résultat au format .xlsx.

"""

import warnings
warnings.filterwarnings('ignore')

import re
import pandas as pd
import json
import numpy as np
import os

from timeit import default_timer as timer
import magic
import yaml
import logging
import logging.config

from .main import load_model
from . import lib_plot
from .lib_extraction import create_fus, from_xml_to_mrv_format, from_pdf_to_mrv_format, plumber_df


with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)


with open(os.path.join(os.path.dirname(__file__), 'log_conf.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)

logger = logging.getLogger('demonstrateur')


MODEL_PATH = config_data['Model']['path']
PREPRO_PATH = config_data['Data']['prepro']['path']
TEST_PATH = config_data['Data']['demonstrateur']['path']

# load data for preprocessing
with open(os.path.abspath(os.path.join(PREPRO_PATH,'Colonnes.json')), 'r') as file:
        Colonnes = json.load(file)
with open(os.path.abspath(os.path.join(PREPRO_PATH,'mapping.json')), 'r') as file:
    mapping = json.load(file)
dys_ref = pd.read_csv(os.path.join(PREPRO_PATH,'referentiel_dispositif_dysfonctionnement.csv'),';')

m = magic.Magic()


def main(PATH, save=True):
    """
    Script runnable pour l'inference des champs manquant dans les déclaration à partir d"un dossier contenant des fichiers pdf et/ou XML.

    Args:
        PATH (str): chemin vers le dossier contenant les données pdf/XML
        save (bool, optional): Enregistre ou non le fichier Excel. Defaults to True.
    """

    # 1
    start = timer()
    logger.info('Chargement des Données et Conversion au format MRVeille')
    PDF_FILES = [file for file in os.listdir(PATH) if file.endswith(".pdf")]
    XML_FILES = [file for file in os.listdir(PATH) if file.endswith(".xml")]

    df_data = pd.DataFrame(index=np.arange(0, len(PDF_FILES)+len(XML_FILES)))

    L = [from_pdf_to_mrv_format(PATH+'/'+PDF_FILES[i], Colonnes, mapping, from_path=True)
         for i in range(len(PDF_FILES))]
    M = [from_xml_to_mrv_format(PATH+'/'+XML_FILES[i], Colonnes, mapping)
         for i in range(len(XML_FILES))]

    df_data = pd.concat(L+M)
    File_names = PDF_FILES+XML_FILES
    logger.info(str(len(df_data)) + ' fichiers chargés ! ')

    # 2
    logger.info('Chargement des modèles!')
    DCO_model, Dysfonctionnement_model, Consequence_model, Effet_model, Gravite_1234_model, Gravite_01_model, classification_encoder = load_model()
    logger.info('Done  !')
    
    df_result = pd.DataFrame()

    #df_result['NUMERO_DECLARATION'] = df_data['NUMERO_DECLARATION'].tolist()
    df_result['fichier'] = File_names
    N = []
    for text in File_names:
        try:
            N.append(re.findall(r'\d+', text)[-1])
        except:
            N.append('INCONNU')
    # logger.info(N)
    df_result['NUMERO_DECLARATION'] = N
    df_result['NUMERO_DECLARATION'] = df_data['NUMERO_DECLARATION'].tolist()
    df_result['TYPE_VIGILANCE'] = df_data['TYPE_VIGILANCE'].tolist()
    df_result['DM']  = df_data['DM'].tolist()

    # 3
    logger.info('Inférence DC0 ! ')
    DC0 = []
    # On tient compte du DMsi il est renseigné
    try:
        X = df_data[['DESCRIPTION_INCIDENT', 'FABRICANT',
                         'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']].fillna('')
        X['LIBELLE_COMMERCIAL'] = X['LIBELLE_COMMERCIAL'] + '. '+df_data['DM']
    except:
            X = df_data[['DESCRIPTION_INCIDENT', 'FABRICANT',
                         'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']].fillna('')
  
    X = X.applymap(str)
    Proba = DCO_model.predict(X)  # Liste des proba pour les n pdf
    DCO = []
    DCO_ID = []
    for prob in Proba:
        # On sélectionne seulement la réponse la plus problable
        char_data = lib_plot.create_chart_data([prob]).iloc[0]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + \
            '  ('+str(round(char_data['proba'], 3))+')'
        DCO.append(text)
        DCO_ID.append(char_data["class"])

    df_result['DCO'] = DCO
    df_result['DCO_ID'] = DCO_ID

    logger.info('Done !')

    logger.info('TYPOLOGIE ! ')
    DYSFONCTIONNEMENT, CONSEQUENCES, EFFETS,POST_DYSFONCTIONNEMENT = [], [], [],[]
    TDY_ID, CDY_ID, TEF_ID, post_TDY_ID = [], [], [],[]
    X = df_data[['FABRICANT', 'CLASSIFICATION', 'DESCRIPTION_INCIDENT',
                 'ETAT_PATIENT', 'ACTION_PATIENT']].fillna('NON RENSEIGNE')
    X = X.applymap(str)  # Pour éviter les erreurs de type
    try :
        X['CLASSIFICATION'] = classification_encoder.transform(X['CLASSIFICATION'])
    except Exception as e:
        logger.error(e)
        X['CLASSIFICATION'] = [0*i for i in range(len(X))]                                                    

    X_dys = Dysfonctionnement_model.transform(X)
    X_dys = np.reshape(X_dys, (X_dys.shape[0], 1, X_dys.shape[1]))
    Proba = Dysfonctionnement_model.predict(X_dys)
    for i in range(len(Proba)):
        char_data = lib_plot.create_chart_data_typo(
            [Proba[i]], lib_plot.get_name_dysfonctionnement, le=lib_plot.le_dys).iloc[0]
        # Post-processing, on évite la classe D0
        if char_data["class"] == 'D0':
            char_data = lib_plot.create_chart_data_typo(
                [Proba[i]], lib_plot.get_name_dysfonctionnement, le=lib_plot.le_dys).iloc[1]
            
        # Post processing concernant l'utilisation du référentiel

        try :
            df_data['DM'] = df_data['DM'].fillna(' ')
            if (len(df_data['DM'].iloc[i])>1) and (df_data['TYPE_DECLARANT'].iloc[i]=="Etablissements et professionnels de santé"):
                dco_id = df_result['DCO_ID'].iloc[i]
                dys_list = dys_ref[dys_ref['DCO_ID']==dco_id]['TDY_ID'].tolist()
                #print(dys_list)
                df_prob = lib_plot.create_chart_data_typo([Proba[i]], lib_plot.get_name_dysfonctionnement, le=lib_plot.le_dys)
                n = len(df_prob)
                for p in range(n): 
                    if (int(df_prob['class'].iloc[p][1:]) in dys_list) and (int(df_prob['class'].iloc[p][1:]) not in [2338,1859]):
                        char_data_post = df_prob.iloc[p]
                        break
                    elif p==n-1 : 
                        logger.info(f"Le post processing n'a pas eu d'effet sur le {i}ème document car l'intersection entre la prediction et la liste des dysfonctionnement est vide")
                        char_data_post = char_data
            else :
                logger.info(f"Le post processing n'a pas eu  d'effet sur le {i}ème document d'effet car la condition d'application sur le DM et le type de déclarant n'est pas remplie DM  = {df_data['DM'].iloc[i]} et TYPE_DECLARANT = {df_data['TYPE_DECLARANT'].iloc[i]}")
                char_data_post = char_data
            text = char_data_post['class_name'] + \
            '  ('+str(round(char_data_post['proba'], 3))+')'
            POST_DYSFONCTIONNEMENT.append(text)
            classe = char_data_post["class"]
            post_TDY_ID.append(classe)
        except:            
            logger.warning("Il y a eu un problème lors de la lecture du champs DM, ainsi le post processing n'est pas effectif pour le "+str(i)+"ème document")
            text = ''
            classe = 0
            post_TDY_ID.append(classe)
            POST_DYSFONCTIONNEMENT.append(text)
        
        # On concatène le texte et la proba
        text = char_data['class_name'] + \
            '  ('+str(round(char_data['proba'], 3))+')'
        DYSFONCTIONNEMENT.append(text)
        TDY_ID.append(char_data["class"])

    X_cons = Consequence_model.transform(X)
    X_cons = np.reshape(X_cons, (X_cons.shape[0], 1, X_cons.shape[1]))
    Proba = Consequence_model.predict(X_cons)
    for i in range(len(Proba)):
        char_data = lib_plot.create_chart_data_typo(
            [Proba[i]], lib_plot.get_name_consequence, le=lib_plot.le_consequence).iloc[0]
        if (char_data["class"] == 'C0') & (df_data['TYPE_VIGILANCE'].iloc[i].lower() in ['réactovigilance', 'reactovigilance']):
            char_data = lib_plot.create_chart_data_typo(
                [Proba[i]], lib_plot.get_name_consequence, le=lib_plot.le_consequence).iloc[1]
        # On concatène le texte et la proba
        text = char_data['class_name'] + \
                '  ('+str(round(char_data['proba'], 3))+')'
        CONSEQUENCES.append(text)
        CDY_ID.append(char_data["class"])

    X_effet = Effet_model.transform(X)
    X_effet = np.reshape(X_effet, (X_effet.shape[0], 1, X_effet.shape[1]))
    Proba = Effet_model.predict(X_effet)
    for i in range(len(Proba)):
        char_data = lib_plot.create_chart_data_typo(
            [Proba[i]], lib_plot.get_name_effet, le=lib_plot.le_effet).iloc[0]
        if (char_data["class"] == 'E1213') & (df_data['TYPE_VIGILANCE'].iloc[i].lower() in ['réactovigilance', 'reactovigilance']):
            char_data = lib_plot.create_chart_data_typo(
                [Proba[i]], lib_plot.get_name_effet, le=lib_plot.le_effet).iloc[1]
        # On concatène le texte et la proba
        text = char_data['class_name'] + \
            '  ('+str(round(char_data['proba'], 3))+')'
        EFFETS.append(text)
        TEF_ID.append(char_data["class"])

    df_result['Dysfonctionnements'], df_result['Consequences'], df_result['Effets'] = DYSFONCTIONNEMENT, CONSEQUENCES, EFFETS
    df_result['TDY_ID'], df_result['CDY_ID'], df_result['TEF_ID'] = TDY_ID, CDY_ID, TEF_ID

    df_result['post_Dysfonctionnements'] = POST_DYSFONCTIONNEMENT
    df_result['post_TDY_ID'] = post_TDY_ID
    
    
    logger.info('Done !')

    logger.info('inférence de la GRAVITE! ')
    X = df_data[['CLASSIFICATION', 'DESCRIPTION_INCIDENT', 'ETAT_PATIENT',
                 'ACTION_PATIENT', 'FABRICANT']].fillna('')
    X = X.applymap(str)

    GRAVITE_2, GRAVITE_5 = [], []

    Proba = Gravite_1234_model.predict(X)
    for prob in Proba:
        char_data = lib_plot.create_chart_data_Gravite(
            [prob], lib_plot.dec_di_multi).iloc[0]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + \
            '  ('+str(round(char_data['proba'], 3))+')'
        GRAVITE_5.append(text)

    Proba = Gravite_01_model.predict(X)

    for prob in Proba:
        char_data = lib_plot.create_chart_data_Gravite(
            [prob], lib_plot.dec_di_bin).iloc[0]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + \
            '  ('+str(round(char_data['proba'], 3))+')'
        GRAVITE_2.append(text)

    df_result['GRAVITE_5'], df_result['GRAVITE_2'] = GRAVITE_5, GRAVITE_2

    logger.info('Done !')

    if save:
        df_result.to_excel('resulats.xlsx')
    end = timer()
    logger.info("""Temps d'éxecution """+ str(end-start)+ 's')
    return(df_result)


d = main(TEST_PATH)
