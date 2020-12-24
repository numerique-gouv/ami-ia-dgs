"""
Module de chargement et de classification

:author: cpoulet@starclay.fr
"""
import os
import numpy as np
import logging
import pandas as pd

from . import prediction_context

from prediction_models.inference.inference_dco import DcoModel
from prediction_models.inference.inference_typologie import TypologieModel
from prediction_models.inference.inference_gravite import GraviteModel
from prediction_models.inference.encodeur import EncodeurModel

logger = logging.getLogger('prediction_models')


def load_classification_models(models_path):
    """
    Fonction permettant de charger les modèles

    :param models_path: chemin vers les data des modèles de classif
    :return: DCO_model, Dysfonctionnement_model, Consequence_model, Effet_model, \
           Gravite_1234_model, Gravite_01_model, classification_encoder
    """
    # Chargement du modèle des DCO
    logger.info('Loading DCO Model')
    DCO_model = DcoModel()
    DCO_model.load(models_path)

    # Chargement des modèles de typologie
    logger.info('Loading Effects model')
    Effet_model = TypologieModel()
    Effet_model.typo = 'tef'
    Effet_model.load(models_path)

    logger.info('Loading Dysfunctions Model')
    Dysfonctionnement_model = TypologieModel()
    Dysfonctionnement_model.typo = 'tdy'
    Dysfonctionnement_model.load(models_path)

    logger.info('Loading Consequences Model')
    Consequence_model = TypologieModel()
    Consequence_model.typo = 'cdy'
    Consequence_model.load(models_path)

    # Modèle de Gravité
    logger.info('Loading Gravity Model 1')
    Gravite_1234_model = GraviteModel()
    Gravite_1234_model.load(models_path)

    logger.info('Loading Gravity Model 2')
    Gravite_01_model = GraviteModel()
    Gravite_01_model.binaire = True
    Gravite_01_model.load(models_path)

    # Encodeur
    logger.info('Loading data encoder')
    classification_encoder = EncodeurModel()
    classification_encoder.load(models_path)

    return DCO_model, Dysfonctionnement_model, Consequence_model, Effet_model, \
           Gravite_1234_model, Gravite_01_model, classification_encoder


def predict_DCO(dco_model, df_data):
    """
    Prédiction de la DCO

    :param dco_model: modèle à utiliser
    :param df_data: dataframe à classifier
    :return: dataframe
    """
    # essayer de voir si on connait le libelle commercial dans le referentiel si le déclarant est un professionnel
    if isinstance(df_data['TYPE_DECLARANT'].loc[0], str) and 'professionnel' in df_data['TYPE_DECLARANT'].loc[0].lower():
        # TODO : generaliser pour N docs : ici ça ne fonctionne que pour 1 doc, le 0
        dco_id = prediction_context.get_dco_id_from_label(df_data['LIBELLE_COMMERCIAL'].loc[0])
        if dco_id is None:
            dco_id = prediction_context.get_dco_id_from_label(df_data['DM'].loc[0])
        if dco_id is not None:
            return pd.DataFrame({'class': [dco_id, -1],
                                 'class_name': [prediction_context.get_name(dco_id), ' '],
                                 'proba': [1, 0]})

    try:
        X = df_data[['DESCRIPTION_INCIDENT', 'FABRICANT',
                     'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']].fillna('')
        X['LIBELLE_COMMERCIAL'] = X['LIBELLE_COMMERCIAL'] + '. ' + df_data['DM']
    except:
        X = df_data[['DESCRIPTION_INCIDENT', 'FABRICANT',
                     'REFERENCE_COMMERCIALE', 'LIBELLE_COMMERCIAL']].fillna('')
    X = X.applymap(str)  # Pour éviter les erreurs de type
    # faire la prédiction
    Proba = dco_model.predict(X)
    return prediction_context.contextualize_prediction(Proba)


def predict_dysfonctionnements(dysfonctionnement_model, classification_encoder, df_data):
    """
    Prédiction du dysfonctionnement

    :param dysfonctionnement_model: modèle à utiliser
    :param classification_encoder: encodeur à utiliser
    :param df_data: dataframe à classifier
    :return: dataframe
    """
    X = df_data[['FABRICANT', 'CLASSIFICATION', 'DESCRIPTION_INCIDENT',
                 'ETAT_PATIENT', 'ACTION_PATIENT']].fillna('NON RENSEIGNE')
    X = X.applymap(str)  # Pour éviter les erreurs de type
    X['CLASSIFICATION'] = classification_encoder.transform(X['CLASSIFICATION'])
    X_dys = dysfonctionnement_model.transform(X)
    X_dys = np.reshape(X_dys, (X_dys.shape[0], 1, X_dys.shape[1]))

    Proba = dysfonctionnement_model.predict(X_dys)
    Proba = prediction_context.contextualize_prediction_typo(Proba,
                                                             prediction_context.get_name_dysfonctionnement,
                                                             le=prediction_context.le_dys)
    Proba = Proba[Proba['class_name'] != "INACTIF"]
    return Proba


def predict_consequences(consequence_model, classification_encoder, df_data):
    """
    Prédiction de la conséquence

    :param consequence_model: modèle à utiliser
    :param classification_encoder: encodeur à utiliser
    :param df_data: dataframe à classifier
    :return: dataframe
    """
    X = df_data[['FABRICANT', 'CLASSIFICATION', 'DESCRIPTION_INCIDENT',
                 'ETAT_PATIENT', 'ACTION_PATIENT']].fillna('NON RENSEIGNE')
    X = X.applymap(str)  # Pour éviter les erreurs de type
    X['CLASSIFICATION'] = classification_encoder.transform(X['CLASSIFICATION'])

    X_cons = consequence_model.transform(X)
    X_cons = np.reshape(X_cons, (X_cons.shape[0], 1, X_cons.shape[1]))
    Proba = consequence_model.predict(X_cons)
    return prediction_context.contextualize_prediction_typo(Proba,
                                                            prediction_context.get_name_consequence,
                                                            le=prediction_context.le_consequence)


def predict_effets(effet_model, classification_encoder, df_data):
    """
    Prédiction de l'effet

    :param effet_model: modèle à utiliser
    :param classification_encoder: encodeur à utiliser
    :param df_data: dataframe à classifier
    :return: dataframe
    """
    X = df_data[['FABRICANT', 'CLASSIFICATION', 'DESCRIPTION_INCIDENT',
                 'ETAT_PATIENT', 'ACTION_PATIENT']].fillna('NON RENSEIGNE')
    X = X.applymap(str)  # Pour éviter les erreurs de type
    X['CLASSIFICATION'] = classification_encoder.transform(X['CLASSIFICATION'])

    X_effet = effet_model.transform(X)
    X_effet = np.reshape(X_effet, (X_effet.shape[0], 1, X_effet.shape[1]))
    Proba = effet_model.predict(X_effet)
    return prediction_context.contextualize_prediction_typo(Proba,
                                                            prediction_context.get_name_effet,
                                                            le=prediction_context.le_effet)


def predict_gravites(gravite_1234_model, gravite_01_model, df_data):
    """
    Prédiction des gravités

    :param gravite_1234_model: modèle 1 à utiliser
    :param gravite_01_model: modèle 2 à utiliser
    :param df_data: dataframe à classifier
    :return: dataframe
    """
    # Test de la présence d'un mot ayant trait à la mort dans le texte
    df_data['text'] = df_data['DESCRIPTION_INCIDENT'] + ' ' + df_data['ETAT_PATIENT']
    deces = df_data['text'].map(lambda x : prediction_context.find_deces(x))
    if deces.loc[0] == 1:
        # TODO : generaliser pour N docs : ici ça ne fonctionne que pour 1 doc, le 0
        d1 = pd.DataFrame({'class_name': 'CRITI', 'proba': 1}, index=[0])
        d2 = pd.DataFrame({'class_name': 'CRITI', 'proba': 1}, index=[0])
        return [d1, d2]

    # Sinon (cas normal) on classifie
    X = df_data[['CLASSIFICATION', 'DESCRIPTION_INCIDENT', 'ETAT_PATIENT',
                 'ACTION_PATIENT', 'FABRICANT']].fillna('')
    X = X.applymap(str)
    Proba = gravite_1234_model.predict(X)

    d1 = prediction_context.contextualize_prediction_gravity(Proba, prediction_context.dec_di_multi)
    
    Proba = gravite_01_model.predict(X)
    d2 = prediction_context.contextualize_prediction_gravity(Proba, prediction_context.dec_di_bin)
    return [d1, d2]


def get_models_performances(models_path):
    perfs_df = pd.read_csv(os.path.join(models_path, 'performances.csv'))
    return perfs_df.to_dict()


def transversal_prediction_postprocess(f_data, f_results):
    """
    Fonctions permettant d'implémenter les post-process "transverses", c'est à dire basés sur des conditions concernant
    plusieurs modèles

    :param f_data: dataframe de donnée du document à post-processer (utilisée pour les prédictions)
    :param f_results: liste de résultats des modèles:
                        [{model_name: ..., predictions: dataframe contextualisée (voir fonctions de prédictions)}]
    :return: f_results modifiée
    """
    try:
        # si dco = 'virologie : ...' et conséquences = 'faux negatif', on booste les gravité à critique
        dco_index = [m['model_name'] for m in f_results].index('DCO')
        dco_preds = f_results[dco_index]['predictions']
        if 'virologie' == dco_preds['class_name'].iloc[0].lower()[:9]:
            try:
                cons_index = [m['model_name'] for m in f_results].index('consequence')
                cons_preds = f_results[cons_index]['predictions']
                if 'faux negatif' == cons_preds['class_name'].iloc[0].lower():
                    try:
                        grav_index = [m['model_name'] for m in f_results].index('gravité_ordinale')
                        f_results[grav_index]['predictions'] = pd.DataFrame({'class_name': 'CRITI', 'proba': 1},
                                                                            index=[0])
                    except ValueError:
                        pass

                    try:
                        gravbin_index = [m['model_name'] for m in f_results].index('gravité_binaire')
                        f_results[gravbin_index]['predictions'] = pd.DataFrame({'class_name': 'CRITI', 'proba': 1},
                                                                               index=[0])
                    except ValueError:
                        pass

            except ValueError:
                pass

    except ValueError:
        pass

    return f_results
