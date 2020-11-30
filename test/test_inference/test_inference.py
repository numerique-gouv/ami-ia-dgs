"""
Auteur : Quillivic Robin, Data Scientist

Fichier pour tester le module inférence
"""

import warnings
warnings.filterwarnings('ignore')

import unittest
import sys
sys.path.insert(0, '../../prediction_models/inference')


import pandas as pd
import os
import numpy as np

import TYPOLOGIE
import DCO
import GRAVITE
import GRAVITE_Binaire



class DCOTest(unittest.TestCase):
    def setUp(self):
        self.data_test = pd.read_csv('./data_test_1000.csv')
        self.columns = self.data_test.columns
        
    def test_prepare_data(self) : 
        df_prep,_ = DCO.prepare_data(self.data_test ,clean=True,n=3)
        self.assertTrue(df_prep[['NUMERO_DECLARATION',
'TYPE_DECLARANT','DESCRIPTION_INCIDENT','ETAT_PATIENT','LIBELLE_COMMERCIAL','REFERENCE_COMMERCIALE','FABRICANT','TYPE_VIGILANCE','CLASSIFICATION']].isna().sum().sum()==0)
        
    def test_train_DCO(self):
        df_prep,_ = DCO.prepare_data(self.data_test ,clean=True,n=3)
        pipeline = DCO.train_DCO(df_prep)
        self.assertTrue(len(pipeline['clf'].classes_)>1)
        
        
class GraviteTest(unittest.TestCase):
    def setUp(self):
        self.data_test = pd.read_csv('./data_test_1000.csv')
        self.columns = self.data_test.columns
        
    def test_prepare_data(self) : 
        df_prep = GRAVITE.prepare_data(self.data_test ,clean=True,n=3)
        self.assertTrue(df_prep[['NUMERO_DECLARATION',
'TYPE_DECLARANT','DESCRIPTION_INCIDENT','ETAT_PATIENT','ACTION_PATIENT','FABRICANT','TYPE_VIGILANCE','CLASSIFICATION']].isna().sum().sum()==0)
        y = df_prep.GRAVITE
        self.assertEqual(len(set(y)),5)
        
    def test_train_GRAVITE(self):
        df_prep = GRAVITE.prepare_data(self.data_test ,clean=True,n=3)
        #pipeline = GRAVITE.train_G(df_prep)
        #self.assertTrue(len(pipeline['clf'].classes_)>1)

        
class GraviteBinaireTest(unittest.TestCase):
    def setUp(self):
        self.data_test = pd.read_csv('./data_test_1000.csv')
        self.columns = self.data_test.columns
        
    def test_prepare_data(self) : 
        df_prep = GRAVITE_Binaire.prepare_data(self.data_test ,clean=True,n=3)
        self.assertTrue(df_prep[['NUMERO_DECLARATION',
'TYPE_DECLARANT','DESCRIPTION_INCIDENT','ETAT_PATIENT','ACTION_PATIENT','FABRICANT','TYPE_VIGILANCE','CLASSIFICATION']].isna().sum().sum()==0)
        y = df_prep.GRAVITE
        self.assertEqual(len(set(y)),2)
        
    def test_train_GRAVITE(self):
        df_prep = GRAVITE_Binaire.prepare_data(self.data_test ,clean=True,n=3)
        #pipeline = GRAVITE.train_G(df_prep)
        #self.assertTrue(len(pipeline['clf'].classes_)>1)
        
class TypologieTest(unittest.TestCase):
    def setUp(self):
        self.data_test = pd.read_csv('./data_test_1000.csv')
        self.svd = 1000
        self.columns = self.data_test.columns
        
    def test_crate_multilabel(self):
        TYPOLOGIE.create_multilabel_data(self.data_test)
        self.assertTrue(os.path.isfile('multilabel_data.pkl'))
        
        df = pd.read_pickle('./multilabel_data.pkl')
        self.assertTrue(set(df.columns).difference(self.columns)=={'text'})
        self.assertTrue(type(df['TEF_ID'][0])==list)
        self.assertTrue(type(df['CDY_ID'][0])==list)
        self.assertTrue(type(df['TDY_ID'][0])==list)
        
        self.assertTrue(df[['NUMERO_DECLARATION',
'TYPE_DECLARANT','DESCRIPTION_INCIDENT','ETAT_PATIENT','ACTION_PATIENT','LIBELLE_COMMERCIAL','REFERENCE_COMMERCIALE','FABRICANT','TYPE_VIGILANCE','CLASSIFICATION']].isna().sum().sum()==0)
       
           
    def test_prepare_data(self):
        df = pd.read_pickle('./multilabel_data.pkl')
        for typo in ['TEF_ID','CDY_ID','TDY_ID']:
            X_train, y_train = TYPOLOGIE.prepare_data(df,typo,n=50,split=False)
            self.assertEqual(X_train.shape[2],50)
            
            self.assertTrue(type(X_train[np.random.randint(0,99)][0][np.random.randint(0,49)]==float))
     
    def test_train(self):
        df = pd.read_pickle('./multilabel_data.pkl')
        for typo in ['TEF_ID','CDY_ID','TDY_ID']:
            X_train,y_train, X_test,y_test= TYPOLOGIE.prepare_data(df,typo,n=50,split=True)
            model = TYPOLOGIE.train(X_train,y_train,typo, save=False)
            self.assertTrue(len(model.trainable_variables[0][0])==800)
            f1 = TYPOLOGIE.evaluate_model(model, X_test,y_test,typo)
            self.assertTrue(f1>0)
        

    
if __name__ == "__main__":
    unittest.main()