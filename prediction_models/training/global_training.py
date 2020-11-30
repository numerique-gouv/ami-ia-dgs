"""
Auteur: 
    Quillivic Robin, Data Scientist chez StarClay, rquilivic@starclay.fr

Description: 
    Ce fichier contient principalement l'execution de l'entrainement et de la sauvegarde des modèles.
    Les paramètres à modifier avant utilisation sont: 
        - L'emplacement des données, ligne 35: DATA_PATH

"""


import clean_text
import pandas as pd
import numpy as numpy
import time
import joblib
import os

import warnings
warnings.filterwarnings('ignore')

import logging
import logging.config
# logging.basicConfig(filename='./inference.log',level=logging.DEBUG)
import yaml

import model_dco
import model_gravite
import model_gravite_binaire
import model_typologie

from timeit import default_timer as timer
import os


from datetime import date
today = date.today()
d4 = today.strftime("%b-%d-%Y")

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)


MODEL_PATH = os.path.join(config_data['training']['save_path'], 'model_'+str(d4))
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

DATA_PATH = os.path.abspath(
    config_data['training']['data']['mrv']['file_path'])

# load logging conf and create logger
with open(os.path.join(os.path.dirname(__file__), '../log_conf.yaml'), 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)

logger = logging.getLogger('inference')


def main() :
    """
    Fonction principale qui réentraine les modèles, les évalue (performance.csv) 
    et les enregistre dans ./models.
    L'avancé du script est dans le fichier inference.log à la racine du dossier Git

    Returns:
        None
    """
    df_result = pd.DataFrame(columns=['DCO', 'TDY', 'CDY', 'TEF', 'GRAVITE_2', 'GRAVITE_5'], index=[
                             'score global', 'score Citoyen', 'score Professionnel'])
    logger.info("Chargement des Données !")
    df_declaration_mrv = pd.read_csv(os.path.abspath(DATA_PATH))
    logger.info('Done ! ')
    logger.info('### DCO ###')
    start = timer()
    cv = 5
    df_DCO = df_declaration_mrv[['DESCRIPTION_INCIDENT', 'TYPE_VIGILANCE', 'LIBELLE_COMMERCIAL', 'REFERENCE_COMMERCIALE', 'ETAT_PATIENT', 'DCO', 'FABRICANT',
                                 'DCO_ID', 'CLASSIFICATION', "TYPE_DECLARANT"]]
    logger.info("0) Préparation des données")
    df_prep, le = model_dco.prepare_data(df_DCO, n=cv, clean=True)
    logger.info(
        '1) Calcul des scores de performances (sauvegardés dans result.csv)')
    score, score_citoyens, score_pro = model_dco.repro_result(
        df_prep, n=cv, citoyen=True, pro=True, clean=True)
    logger.info("2) Entrainement du modèle sur l'ensemble les données")
    model = model_dco.train_DCO(df_prep)
    logger.info('3) Sauvegarde des résultats et du modèle')
    joblib.dump(model, os.path.join(
        MODEL_PATH, config_data['training']['models']['model_dco']['model_filename']))
    joblib.dump(le, os.path.join(
        MODEL_PATH, config_data['training']['preparation']['encodeur']['dco']['model_filename']))
    df_result['DCO'] = [round(score, 3), round(
        score_citoyens, 3), round(score_pro, 3)]
    end = timer()
    logger.info("Temps écoulé, pour l'éxecution : "+str(end - start)+ "secondes")
    logger.info('### TYPOLOGIE ###')
    start = timer()
    df_TYPO = df_declaration_mrv[['NUMERO_DECLARATION', 'DESCRIPTION_INCIDENT', 'TYPE_VIGILANCE', 'LIBELLE_COMMERCIAL',
                                  'REFERENCE_COMMERCIALE', 'ETAT_PATIENT', 'ACTION_PATIENT', 'DCO', 'FABRICANT', 'TEF_ID', 'CDY_ID',
                                  'TDY_ID', 'CLASSIFICATION', "TYPE_DECLARANT"]]
    logger.info("0) Préparation des données")
    model_typologie.create_multilabel_data(df_TYPO)

    logger.info('#### TYPE DYSFONCTIONNEMENT ###')
    typo = 'TDY_ID'
    logger.info(' Calcul des poids de chaque classe pour gérer le déséquilibrage')
    weights = model_typologie.compute_class_weight(df_TYPO,typo= typo)
    logger.info(
        '1) Calcul des scores de performances (sauvegardés dans performances.csv)')
    score, score_citoyens, score_pro = model_typologie.repro_result(
        typo=typo, citoyen=True, pro=True,weight = weights)
    logger.info("2) Entrainement du modèle sur l'ensemble les données")
    mrv = pd.read_pickle('./multilabel_data.pkl')
    X_train_, y_train = model_typologie.prepare_data(
        mrv, typo, n=1000, split=False)
    model = model_typologie.train(X_train_, y_train, typo=typo,weight = weights)
    logger.info('3) Sauvegarde des résultats et du modèle')
    model_json = model.to_json()
    with open(os.path.join(MODEL_PATH, config_data['training']['models']['model_typologie']['tdy']['archi_filename']), "w") as json_file:
        json_file.write(model_json)
    model.save(os.path.join(
        MODEL_PATH, config_data['training']['models']['model_typologie']['tdy']['model_filename']))
    df_result['TDY'] = [round(score, 3), round(
        score_citoyens, 3), round(score_pro, 3)]

    logger.info('#### CONSÉQUENCE DYSFONCTIONNEMENT ###')
    typo = 'CDY_ID'
    logger.info(
        '1) Calcul des scores de performances (sauvegardés dans result.csv)')
    score, score_citoyens, score_pro = model_typologie.repro_result(
        typo=typo, citoyen=True, pro=True)
    logger.info("2) Entrainement du modèle sur l'ensemble les données")
    mrv = pd.read_pickle('./multilabel_data.pkl')
    X_train_, y_train = model_typologie.prepare_data( mrv, typo, n=1000, split=False)
    model = model_typologie.train(X_train_, y_train, typo=typo)
    logger.info('3) Sauvegarde des résultats et du modèle')
    model_json = model.to_json()
    with open(os.path.join(MODEL_PATH, config_data['training']['models']['model_typologie']['cdy']['archi_filename']), "w") as json_file:
        json_file.write(model_json)
    model.save(os.path.join(
        MODEL_PATH, config_data['training']['models']['model_typologie']['cdy']['model_filename']))

    df_result['CDY'] = [round(score, 3), round(
        score_citoyens, 3), round(score_pro, 3)]

    logger.info("#### TYPE D'EFFET ###")
    typo = 'TEF_ID'
    logger.info(' Calcul des poids de chaque classe pour gérer le déséquilibrage')
    weights = model_typologie.compute_class_weight(df_TYPO,typo= typo)
    
    logger.info(
        '1) Calcul des scores de performances (sauvegardés dans result.csv)')
    score, score_citoyens, score_pro = model_typologie.repro_result(
        typo=typo, citoyen=True, pro=True,weight = weights)
    
    logger.info("2) Entrainement du modèle sur l'ensemble les données")
    mrv = pd.read_pickle('./multilabel_data.pkl')
    X_train_, y_train = model_typologie.prepare_data(
        mrv, typo, n=1000, split=False)
    model = model_typologie.train(X_train_, y_train, typo=typo, weight = weights)
    
    logger.info('3) Sauvegarde des résultats et du modèle')
    model_json = model.to_json()
    with open(os.path.join(MODEL_PATH, config_data['training']['models']['model_typologie']['tef']['archi_filename']), "w") as json_file:
        json_file.write(model_json)
    model.save(os.path.join(
        MODEL_PATH, config_data['training']['models']['model_typologie']['tef']['model_filename']))

    df_result['TEF'] = [round(score, 3), round(
        score_citoyens, 3), round(score_pro, 3)]
    end = timer()
    logger.info("Temps écoulé, pour l'éxecution : "+ str(end - start)+ "secondes")
    logger.info('### GRAVITE ###')
    start = timer()
    logger.info('#### 5 classes ')
    cv = 3
    df_GRAVITE = df_declaration_mrv[['DESCRIPTION_INCIDENT', 'TYPE_VIGILANCE', 'ACTION_PATIENT',
                                     'ETAT_PATIENT', 'FABRICANT', 'CLASSIFICATION', "TYPE_DECLARANT", 'GRAVITE']]
    logger.info("0) Préparation des données")
    df_prep = model_gravite.prepare_data(df_GRAVITE, clean=True)
    logger.info(
        '1) Calcul des scores de performances (sauvegardés dans performances.csv)')
    score, score_citoyens, score_pro = model_gravite.repro_result(
        df_prep, n=cv, citoyen=True, pro=True, clean=False)
    logger.info("2) Entrainement du modèle sur l'ensemble les données")
    model = model_gravite.train_GRAVITE(df_prep)
    logger.info('3) Sauvegarde des résultats et du modèle')
    joblib.dump(model, os.path.join(
        MODEL_PATH, config_data['training']['models']['model_gravite']['non_binaire']['model_filename']))
    df_result['GRAVITE_5'] = [round(score, 3), round(
        score_citoyens, 3), round(score_pro, 3)]

    logger.info('#### 2 classes ')
    cv = 3
    df_GRAVITE = df_declaration_mrv[['DESCRIPTION_INCIDENT', 'TYPE_VIGILANCE', 'ACTION_PATIENT',
                                     'ETAT_PATIENT', 'FABRICANT', 'CLASSIFICATION', "TYPE_DECLARANT", 'GRAVITE']]
    logger.info("0) Préparation des données")
    df_prep = model_gravite_binaire.prepare_data(df_GRAVITE, clean=True)
    logger.info(
        '1) Calcul des scores de performances (sauvegardés dans result.csv)')
    score, score_citoyens, score_pro = model_gravite_binaire.repro_result(
        df_prep, n=cv, citoyen=True, pro=True, clean=False)
    logger.info("2) Entrainement du modèle sur l'ensemble les données")
    model = model_gravite_binaire.train_GRAVITE(df_prep)
    logger.info('3) Sauvegarde des résultats et du modèle')
    joblib.dump(model, os.path.join(
        MODEL_PATH, config_data['training']['models']['model_gravite']['binaire']['model_filename']))
    df_result['GRAVITE_2'] = [round(score, 3), round(
        score_citoyens, 3), round(score_pro, 3)]
    end = timer()

    logger.info(str("Temps écoulé, pour l'éxecution : " + str(end - start)+ "secondes"))
    df_result.to_csv(os.path.join(MODEL_PATH, 'performances.csv'))
    
    return 'Terminé !'


if __name__ == '__main__':
    main()
