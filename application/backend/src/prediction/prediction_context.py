""" 
Auteur: 
    - Quillivic Robin, Data SCientist chez StarClay, rquilivic@starclay.fr
Description: 
    Fichier regroupant l'ensemble des fonctions utiles pour la visualisation des résultats de l'inférence des modèles: 
        - affichage du nom des différentes classes
        - construction de tableau de résultats pour un affichage lisible dans le démonstrateur. 

"""


import pandas as pd
import joblib
import numpy as np
import os
from gensim.utils import deaccent

from backend_utils.config_parser import get_local_file, parse_full_config

config = parse_full_config(get_local_file('../config.yaml'), get_local_file('../config_env.yaml'))

MODEL_PATH = config['models']['path']
DATA_PATH = config['data']['mrv']['path']


# dictionaire des id pour le DCO

id_to_dco = pd.read_csv(os.path.abspath(os.path.join(DATA_PATH, "referentiel_dispositif.csv")), delimiter=';', encoding='ISO-8859-1')
dco_to_id = {deaccent(v.lower().replace(' ', '')): k for (k, v) in id_to_dco.values.tolist()}

# Modèle d'encodage pour le DCO
le = joblib.load(os.path.abspath(os.path.join(MODEL_PATH, 'DCO_encoder.sav')))
le_effet = joblib.load(os.path.abspath(os.path.join(MODEL_PATH, 'TEF_ID_Encoder.sav')))
le_consequence = joblib.load(os.path.abspath(os.path.join(MODEL_PATH, 'CDY_ID_Encoder.sav')))
le_dys = joblib.load(os.path.abspath(os.path.join(MODEL_PATH, 'TDY_ID_Encoder.sav')))

# dictionaire pour la typologie
df_effets = pd.read_csv(os.path.abspath(os.path.join(DATA_PATH, "referentiel_dispositif_effets_connus.csv")),
                        delimiter=';', encoding='ISO-8859-1')
df_dys = pd.read_csv(os.path.abspath(os.path.join(DATA_PATH, "referentiel_dispositif_dysfonctionnement.csv")),
                     delimiter=';', encoding='ISO-8859-1')
df_csq = pd.read_csv(os.path.abspath(os.path.join(DATA_PATH, "referentiel_consequence.csv")))


def get_dco_id_from_label(label):
    try:
        return dco_to_id[deaccent(label.lower().replace(' ', ''))]
    except KeyError:
        return None


def get_name(x: int) -> (str):
    """
    Renvoie le nom du dispositif à partir du numéro précisé dans le référentiel. 
    Si le numéro n'existe pas, alors il ne renvoie un string vide.

    Args:
        x (int): Numéro de la classe de DCO

    Returns:
        dco (str): Nom de la classe du dispositif médical (DCO) 
    """
    try:
        dco = id_to_dco[id_to_dco['DCO_ID'] == int(x)]['LIBELLE'].iloc[0]
        return dco
    except:
        if x == '0':
            return ''
        return f"Libelle inconnu; DCO_ID = {x}"


def get_name_effet(x: str) -> (str):
    """
    Renvoie le nom de la classe d'effet à partir  du numéro précisé dans le référentiel. 
    Si le numéro n'existe pas, alors il ne renvoie un string vide.. 

    Args:
        x (str): Numéro de classe du TYPE_EFFET

    Returns:
        effet (str): Nom de la classe du TYPE_EFFET
    """
    try:
        effet = df_effets[df_effets['TEF_ID']
                          == int(x[1:])]['TYPE_EFFET'].iloc[0]
        return effet
    except:
        if x[1:] == '0':
            return ''
        return f"Libelle inconnu; TEF_ID = {x}"


def get_name_dysfonctionnement(x: str) -> (str):
    """
    Renvoie le nom de la classe du dysfonctionnement à partir  du numéro précisé dans le référentiel. 
    Si le numéro n'existe pas, alors il ne renvoie un string vide. 

    Args:
        x (str): Numéro de classe du TYPE_DYSFONCTIONNEMENT

    Returns:
        dys (str): Nom de la classe du TYPE_DYSFONCTIONNEMENT
    """
    try:
        dys = df_dys[df_dys['TDY_ID'] == int(x[1:])]
        if 'SI_ACTIF' in dys.columns and not dys['SI_ACTIF'].iloc[0]:
            return "INACTIF"
        return dys['LIBELLE_DYSFONCTIONNEMENT'].iloc[0]
    except:
        if x[1:] == '0':
            return ''
        return f"Libelle inconnu; TDY_ID = {x}"


def get_name_consequence(x: str) -> (str):
    """
    Renvoie le nom de la classe de la conséquence du dysfonctionnement à partir  du numéro précisé dans le référentiel. 
    Si le numéro n'existe pas, alors il ne renvoie un string vide. 

    Args:
        x (str): Numéro de classe du CONSEQUENCE_DYSFONCTIONNEMENTT

    Returns:
        csq (str): Nom de la classe du CONSEQUENCE_DYSFONCTIONNEMENT
    """
    try:
        csq = df_csq[df_csq['CDY_ID'] ==
                     x]['CONSEQUENCE_DYSFONCTIONNEMENT'].iloc[0]
        return csq
    except:
        if x == 'C0':
            return ''
        return f"Libelle inconnu; CDY_ID = {x}"


def contextualize_prediction(pred: np.array, name=get_name, le=le) -> (pd.DataFrame):
    """
    Reçoit en entrée le vecteur de prédiction de probabilités du modèle et
    renvoie une Dataframe avec le nom des classes associés à chaque probabilité.
    Permet de créer un affichage lisible facilement dans l'appli web.  


    Args:
        pred (np.array): Vecteur des probabilité sortie par le modèle d'inférence de classification. 
        name (function, optional): fonction associé à la variable à  inférer. Defaults to get_name.
            - get_name : DCO
        le (sk.preprocessing.labelEncoder, optional): [description]. Defaults to le.

    Returns: 
        df_r (pd.DataFrame): DataFrame contenant les probabilités et le nom des classes associées triées par ordre décroissant.
    """

    df_r = pd.DataFrame(pred[0])
    df_r["class"] = le.inverse_transform(df_r.index.values)
    df_r['class_name'] = df_r['class'].apply(lambda x: name(x))
    df_r['proba'] = df_r[0]
    df_r = df_r.drop(0, axis=1)
    df_r = df_r.sort_values('proba', ascending=False)
    return df_r


def multi_format(n: int, le):
    """
    Transforme le numéro d'une classe n en un vecteur de taille du nombre de classes avec un un en n ième position

    Args:
        n (int): Le numéro de classe
        le (sk.preprocessing.labelEncoder): le label encodeur fitté sur la variable visée

    Returns:
        y (np.array): vecteur avec un 1 en n ième position
    """
    dim = len(le.classes_)
    y = np.zeros(dim, dtype=int)
    y[n] = 1
    return y


def contextualize_prediction_typo(pred: np.array, name=get_name, le=le) -> (pd.DataFrame):
    """
    Reçoit en entrée le vecteur de prédiction de probabilités du modèle pour la TYPOLOGIE et
    renvoie une Dataframe avec le nom des classes associés à chaque proba.
    Permet de créer un affichage lisible facilement dans l'appli web.  


    Args:
        pred (np.array): Vecteur des probabilité sortie par le modèle d'inférence de classification. 
        name (function, optional): fonction associé à la variable à  inférer. Defaults to get_name.
            - get_name_effet: TYPE_EFFET
            - get_name_consequence: CONSEQUENCE_DYSFONCTIONNEMENT
            - get_name_dysfonctionnement: TYPE_DYSFONCTIONNEMENT
        le (sk.preprocessing.labelEncoder, optional): [description]. Defaults to le.

    Returns: 
        df_r (pd.DataFrame): DataFrame contenant les probabilités et le nom des classes associées triées par ordre décroissant.
    """
    df_r = pd.DataFrame(pred[0])
    df_r["multi"] = df_r.index.map(lambda x: multi_format(x, le))
    df_r['class'] = df_r.multi.map(
        lambda x: le.inverse_transform(np.array([x]))[0][0])
    df_r['class_name'] = df_r['class'].apply(lambda x: name(x))
    df_r['proba'] = df_r[0]
    df_r = df_r.drop(0, axis=1)
    df_r = df_r.sort_values('proba', ascending=False)
    return df_r


# Dictionnaire pour encoder la gravité
enc_di_multi = {
    'NULLE': 0,
    'MINEU': 1,
    'MOYEN': 2,
    'SEVER': 3,
    'CRITI': 4
}
# Dictionnaire pour décoder la gravité
dec_di_multi = {
    0: 'NULLE',
    1: 'MINEU',
    2: 'MOYEN',
    3: 'SEVER',
    4: 'CRITI'
}

# Dictionnaire pour encoder la gravité binaire
enc_di_bin = {
    'NULLE': 1,
    'MINEU': 1,
    'MOYEN': 1,
    'SEVER': 1,
    'CRITI': 0
}
# Dictionnaire pour décoder la gravité binaire
dec_di_bin = {
    1: 'NON CRITIQUE',
    0: 'CRITI'
}


def contextualize_prediction_gravity(pred: np.array, enc: dict) -> (pd.DataFrame):
    """
    Reçoit en entrée le vecteur de prédiction de probabilités du modèle pour la GRAVITE et
    renvoie une Dataframe avec le nom des classes associés à chaque proba.
    Permet de créer un affichage lisible facilement dans l'appli web.  


    Args:
        pred (np.array): Vecteur des probabilité sortie par le modèle d'inférence de classification. 
        enc (dict): dictionnaire de mapping entre l'encodage de la variable gravité et le nom de la classe

    Returns: 
        df_r (pd.DataFrame): DataFrame contenant les probabilités et le nom des classes associées triées par ordre décroissant.
    """
    df_r = pd.DataFrame(pred[0])
    df_r["class_name"] = df_r.index.map(enc)
    df_r['proba'] = df_r[0]
    df_r = df_r.drop(0, axis=1)
    df_r = df_r.sort_values('proba', ascending=False)
    return df_r


def find_deces(text):
    """Renvoie 0 ou 1 en fonction de la présence de vocabulaire lié à la mort dans le texte

    Args:
        text (str): texte assocé au signalement

    Returns:
        deces (int): 0 il n'y a pas de deces, 1 il y a deces
    """
    if not isinstance(text, str):
        return 0
    vocab = deaccent(text.lower()).split(' ')
    deces_list = ['mort','morts','meurt','mourir','mortel','deces','deceder','decede','decedee','decedes','decedait',"decedais",'dcd']
    for word in vocab:
        for elt in deces_list:
            if elt in word:
                return 1
    
    return 0