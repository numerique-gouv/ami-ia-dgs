""" 
Auteur: 
    - Quillivic Robin, Data SCientist chez StarClay, rquilivic@starclay.fr
Description: 
    Fichier regroupant l'ensemble des fonctions utiles pour l'extraction d'information depuis des fichiers pdf, csv et xml.
    Ainsi que pour la conversion au format MRveil qui a été utilisé pour entrainé les modèles 

"""

import pdfplumber
import re
import pandas as pd
import logging
import numpy as np
import xmltodict
import base64
import unidecode


def extract_vigilance_type_from_filename(filename):

    processed_filename = unidecode.unidecode(filename).lower()
    if 'materio' in processed_filename:
        return "Matériovigilance"
    if 'bacterio' in processed_filename:
        return 'Bactériovigilance'
    return ''


def plumber_df(my_file, from_path=False) -> pd.DataFrame:
    """Permet d'extraire l'information des tableaux présents dans un pdf et de le renvoyer
    sous forme de DataFrame pandas. 

    Args:
        my_file (IO.file): fichier pdf chargé
        from_path (bool, optional): Permet de choisir la méthode de chargement du fichier pdf. (Default to False)

    Returns:
        char_df (pd.DataFrame): DataFrame contenant les données de la table dans le pdf
    """
    if from_path:  
        pdf_object = pdfplumber.from_path(my_file)
    else: 
        pdf_object = pdfplumber.load(my_file)
        
    char_df = pd.DataFrame()
    tmp_lst = []
    for i, page in enumerate(pdf_object.pages):
        temp_df = None
        try:
            temp_df = page.extract_table()
        except:
            logging.warning(f"No Table found in page {i} of file {my_file}")
            pass
        if temp_df is not None:
            # Save to list
            tmp_lst.append(temp_df)
            try:
                char_df = char_df.append(temp_df)
            except:
                logging.warning(f"Failed to concatenate table of page {i}")
                pass

    char_df.columns = ["Column0", "Column1"]
    if 'matériovigilance' in pdf_object.pages[-1].extract_text(x_tolerance=1, y_tolerance=1):
        TYPE_VIGILANCE = 'Matériovigilance'
    elif 'réactovigilance' in pdf_object.pages[-1].extract_text(x_tolerance=1, y_tolerance=1):
        TYPE_VIGILANCE = 'Réactovigilance'
    else:
        TYPE_VIGILANCE = ' '
    try:
        NUMERO_DECLARATION = re.findall(
            'Référence du signalement : \d+', pdf_object.pages[0].extract_text(x_tolerance=1, y_tolerance=1))[0].split(' ')[-1]
    except:
        NUMERO_DECLARATION = ' '

    df_2 = pd.DataFrame({'Column0': ['TYPE_VIGILANCE', 'NUMERO_DECLARATION'], 'Column1': [
                        TYPE_VIGILANCE, NUMERO_DECLARATION]})
    char_df = pd.concat([char_df, df_2])
    
    char_df.Column1 = char_df.Column1.map(lambda x: str(x).replace('\n', ' '))

    return char_df


def create_fus(df_table: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Permet de faire la jonction entre les champs de MRveil et les champs des PDF/XML

    Args:
        df_table (pd.DataFrame): DataFrame contenant les données des pdf avec les champs non uniformisés
        mapping (dict): Dictionnaire faisant le lien entre les colonnes de la base MRveil et l'ensemble des noms de colonnes pour les pdf/XML
    Returns: 
        df_fus (pd.DataFrame): DataFrame contenant les même données qu'en entrée mais regroupé selon les champs de la base MRveil
    """
    E = []
    df_fus = pd.DataFrame(index=df_table.index, columns=mapping.keys())
    for key in mapping.keys():
        try:
            if len(mapping[key]) > 0:
                for elt in mapping[key]:
                    # On rassemble les colonnes, mais le NaN écrase tout donc on les remplace par '
                    if str(df_table[elt].values[0]) != 'nan':
                        df_fus[key] = df_fus[key].map(lambda x: str(x).replace(
                            'nan', '')) + df_table[elt].map(lambda x: str(x))
        except:
            E.append(key)

    # Une fois la fusion terminé, on peut remettre les NaN la ou il n'y a pas de donnée
    df_fus = df_fus.replace('', np.NaN)
    # df_fus['file_name'] = df_table['Unnamed: 0.1']# on remet les noms des fichiers qui nous permet de faire la jointure
    return df_fus


def from_pdf_to_mrv_format(file, colonnes: list, mapping: dict, from_path=False) -> pd.DataFrame:
    """
    Synthetise l'ensemble des opérations necessaires pour transformer un fichier pdf au  formet de la base de donnée MRveil

    Args:
        file (IO.file): le fichier pdf chargé apr l'utilisateur
        colonnes (list): Listes des colonnes existants dans tout les types de questionnaires
        mapping (dict): Dictionnaire de mapping entre les Colonnes et les champs de réference de la base mrv
        from_path (bool, optional): Faut-il charger les pdf depuis un chemin ou bien depuis une IO.file. Default to False

    Returns: 
        df_mrv (pd.DataFrame): Une dataframe pandas de longueur une ligne au format de la base MRveil
    """
    df = plumber_df(file,from_path=from_path)
    df = df.mask(df.eq('None')).dropna()  # on supprime les None et les nan
    df_2 = df.drop_duplicates(subset='Column0', keep='last')
    df_3 = df_2.set_index('Column0', verify_integrity=True)
    df_4 = df_3.T.reset_index()
    Cols = df_4.columns
    # On créer un Table avec toute les collones déja rencontré et on utilise les données du pdf pour la remplir
    df_table = pd.DataFrame(columns=colonnes, index=[0])
    df_table[df_4.columns] = df_4

    # On applique le mapping
    df_mrv = create_fus(df_table, mapping)
    if not df_mrv['TYPE_VIGILANCE'].iloc[0] \
            or (not isinstance(df_mrv['TYPE_VIGILANCE'].iloc[0], str) and  np.isnan(df_mrv['TYPE_VIGILANCE'].iloc[0])):
        df_mrv['TYPE_VIGILANCE'].iloc[0] = extract_vigilance_type_from_filename(file)
    return df_mrv


def from_csv_to_mrv_format(file, colonnes: list, mapping: dict) -> pd.DataFrame:
    """
    Synthetise l'ensemble des opérations necessaires pour transformer un fichier csv au  format de la base de donnée MRveil

    Args:
        file (IO.file): le fichier pdf chargé apr l'utilisateur
        colonnes (list): Listes des colonnes existants dans tout les types de questionnaires
        mapping (dict): Dictionnaire de mapping entre les Colonnes et les champs de réference de la base mrv

    Returns: 
        df_mrv (pd.DataFrame): Une dataframe pandas de longueure 1 ligne au format de la base mrv
    """
    df = pd.read_csv(file)  # ,delimiter=',',encoding='utf-8',engine='python')
    df = df.replace('\n', ' ', regex=True)   # il y a parfois des retours à la ligne qui s'ajoutent sur certains os à l'ouverture du fichier
    df = df.mask(df.eq('None')).dropna()  # on supprime les None et les nan
    df_2 = df.drop_duplicates(subset='Column0', keep='last')
    df_3 = df_2.set_index('Column0', verify_integrity=True)
    df_4 = df_3.T.reset_index()
    Cols = df_4.columns
    # On créer un Table avec toute les collones déja rencontré et on utilise les données du pdf pour la remplir
    df_table = pd.DataFrame(columns=colonnes, index=[0])
    df_table[df_4.columns] = df_4.iloc[-1].values  # 0 is the original index

    # On applique le mapping
    df_mrv = create_fus(df_table, mapping)
    if not df_mrv['TYPE_VIGILANCE'].iloc[0] \
            or (not isinstance(df_mrv['TYPE_VIGILANCE'].iloc[0], str) and np.isnan(df_mrv['TYPE_VIGILANCE'].iloc[0])):
        df_mrv['TYPE_VIGILANCE'].iloc[0] = extract_vigilance_type_from_filename(file)
    return df_mrv


def from_xml_to_mrv_format(file_path, colonnes, mapping):
    """
    Synthetise l'ensemble des opérations necessaires pour transformer un fichier xml au  format de la base de donnée MRveil
    Cette fonction permet de gérer l'encodage 64 ou non du fichier XML. Elle repose sur la librairie xmltodict

    Args:
        file_path (str): le chemin du fichier xml à transformer
        colonnes (list): liste des colonnes existants dans tout les types de questionnaires
        mapping (dict): Dictionnaire de mapping entre les Colonnes et les champs de réference de la base mrv

    Returns: 
        df_mrv (pd.DataFrame): Une dataframe pandas de longueure 1 au format de la base MRveil
    """
    try:
        with open(file_path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
            E = []
            res = pd.DataFrame()
            for elt in data_dict['ClinicalDocument']['component']['structuredBody']['component']:
                for el in elt['section']['text']['table']['tbody']['tr']:
                    for i in range(len(el['td'])):
                        try:
                            res[el['td'][2*i]['#text']
                                ] = [el['td'][2*i+1]['#text']]
                            # print('id:', el['td'][2*i]['#text'])
                            # print('values:', el['td'][2*i+1]['#text'])

                        except:
                            try:
                                E.append(el['td'][i]['#text'])
                            except:
                                E.append('erreur !')
            try:
                if 'Réactovigilance' in data_dict['ClinicalDocument']['title'] or 'Reactovigilance' in data_dict['ClinicalDocument']['title']:
                    TYPE = 'Réactovigilance'
                elif 'Matériovigilance' in data_dict['ClinicalDocument']['title'] or 'Materiovigilance' in data_dict['ClinicalDocument']['title']:
                    TYPE = 'Matériovigilance'
                else:
                    TYPE = ''

                NUMERO = data_dict['ClinicalDocument']['id']['@extension']
            except:
                TYPE = ''
                NUMERO = ''

    except:  # on gère le cas de l'encodage avec une exception de manière provisoire
        with open(file_path, 'r') as f:
            xml_file = f.read()
            content = base64.b64decode(xmltodict.parse(xml_file)[
                                       'soap:Envelope']['soap:Body']['ConsulterSignalementResponseV3']['ns4:signalementRetourList']['ns4:signalement'][0]['ns8:cda'])
            data_dict = xmltodict.parse(content)
            E = []
            res = pd.DataFrame()
            for elt in data_dict['ClinicalDocument']['component']['structuredBody']['component']:
                for el in elt['section']['text']['table']['tbody']['tr']:
                    for i in range(len(el['td'])):
                        try:
                            res[el['td'][2*i]['#text']
                                ] = [el['td'][2*i+1]['#text']]
                            # print('id:', el['td'][2*i]['#text'])
                            # print('values:', el['td'][2*i+1]['#text'])

                        except:
                            try:
                                E.append(el['td'][i]['#text'])
                            except:
                                E.append('erreur !')
        with open(file_path, 'r') as f:
            xml_file = f.read()
            TYPE = xmltodict.parse(xml_file)[
                'soap:Envelope']['soap:Body']['ConsulterSignalementResponseV3']['ns4:signalementRetourList']['ns4:signalement'][0]['ns8:typeSignalement']
            NUMERO = xmltodict.parse(xml_file)['soap:Envelope']['soap:Body']['ConsulterSignalementResponseV3'][
                'ns4:signalementRetourList']['ns4:signalement'][0]['ns8:declaration']['ns14:referenceDeclaration']

    df_table = pd.DataFrame(columns=colonnes, index=[0])
    df_table[res.columns] = res

    # On applique le mapping
    df_mrv = create_fus(df_table, mapping)

    df_mrv['NUMERO_DECLARATION'] = NUMERO.lower()
    df_mrv['TYPE_VIGILANCE'] = TYPE.lower()

    if not df_mrv['TYPE_VIGILANCE'].iloc[0] \
            or (not isinstance(df_mrv['TYPE_VIGILANCE'].iloc[0], str) and np.isnan(df_mrv['TYPE_VIGILANCE'].iloc[0])):
        df_mrv['TYPE_VIGILANCE'].iloc[0] = extract_vigilance_type_from_filename(file_path)
    return df_mrv
