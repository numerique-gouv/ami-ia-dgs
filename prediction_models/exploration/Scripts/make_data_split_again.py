"""
Auteur: Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
Description: Création et sauvegarde des daraset multilabel pour la typologie
"""


import pandas as pd
import joblib
import numpy as np
import clean_text

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder
import os

directory = './data_clean/'
if not os.path.exists(directory):
    os.makedirs(directory)

# Chargement des données
mrv = pd.read_csv('data/data_mrv/declaration_mrv_complet.csv',
                 usecols=['NUMERO_DECLARATION',"DESCRIPTION_INCIDENT","ETAT_PATIENT","FABRICANT", "CLASSIFICATION", 'ACTION_PATIENT',
                          'DCO_ID','LIBELLE_COMMERCIAL','REFERENCE_COMMERCIALE','TEF_ID',
                          'TDY_ID','CDY_ID','GRAVITE'])

# 1) On gère nan pour les colonnes contenant du text
text_columns =  ["DESCRIPTION_INCIDENT","ETAT_PATIENT","FABRICANT", 'ACTION_PATIENT','LIBELLE_COMMERCIAL']


mrv[text_columns] = mrv[text_columns].fillna('NON RENSEIGNE')

# 2) on gère les NaN pour chaque Collones
mrv['NUMERO_DECLARATION'] = mrv['NUMERO_DECLARATION'].fillna('NON RENSEIGNE')
mrv['DESCRIPTION_INCIDENT'] = mrv['DESCRIPTION_INCIDENT'].fillna('NON RENSEIGNE')
mrv['ETAT_PATIENT'] = mrv['ETAT_PATIENT'].fillna('NON RENSEIGNE')              
mrv['CLASSIFICATION'] = mrv['CLASSIFICATION'].fillna('NON RENSEIGNE')
mrv['FABRICANT'] = mrv['FABRICANT'].fillna('NON RENSEIGNE')
mrv['LIBELLE_COMMERCIAL'] = mrv['LIBELLE_COMMERCIAL'].fillna('NON RENSEIGNE')
mrv['REFERENCE_COMMERCIALE'] = mrv['REFERENCE_COMMERCIALE'].fillna('NON RENSEIGNE')



#Effet
mrv['TEF_ID']= mrv['TEF_ID'].fillna('E1213')
#DYSFOCNTIONNEMENT
mrv['TDY_ID']  = mrv['TDY_ID'].fillna("D0")
#CONSEQUENCES
mrv['CDY_ID']  = mrv['CDY_ID'].fillna("C0")

#Construction de nouvelles variables

mrv["text"] = mrv['DESCRIPTION_INCIDENT']+' '+mrv['ETAT_PATIENT']


# Encodage des variables catégorielles et sauvegarde des données associé

le = LabelEncoder()
mrv["CLASSIFICATION"] = le.fit_transform(mrv["CLASSIFICATION"])
joblib.dump(le, './data_clean/classification_encodeur.sav')


le = LabelEncoder()
mrv["CDY_ID"] = le.fit_transform(mrv["CDY_ID"])
joblib.dump(le, './data_clean/CDY_ID_encodeur.sav')


le = LabelEncoder()
mrv["TEF_ID"] = le.fit_transform(mrv["TEF_ID"])
joblib.dump(le, './data_clean/TEF_ID_encodeur.sav')


le = LabelEncoder()
mrv["TDY_ID"] = le.fit_transform(mrv["TDY_ID"])
joblib.dump(le, './data_clean/TDY_ID_encodeur.sav')



# La typologie est en génrale multilabel

mrv_id = mrv.groupby('text').agg({'TEF_ID':lambda x: list(set(x)),
                               'CDY_ID':lambda x: list(set(x)),
                               'TDY_ID':lambda x:list(set(x))
                            })

#On fusionne les multilabels et on supprime les doublons
mrv = mrv.drop(['TEF_ID','CDY_ID','TDY_ID'],axis=1)
mrv = pd.merge(mrv,mrv_id, on = 'text')
mrv = mrv.drop_duplicates('text')
mrv = mrv.reset_index(drop=True)


#On remplace les strings vide
mrv.replace('', 'NON RENSEIGNE', inplace=True)
# Sauvegarde des données nettoyées
mrv.to_pickle('./data_clean/split_data.pkl')


# Split data

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1029)

mlb = MultiLabelBinarizer()

for train_index, test_index in msss.split(mrv['text'], mlb.fit_transform(mrv['TEF_ID'])):
    train, test = mrv.loc[train_index],  mrv.loc[test_index]

train.to_pickle('./data_clean/train.pkl')
test.to_pickle('./data_clean/test.pkl')

print(train.shape, test.shape)