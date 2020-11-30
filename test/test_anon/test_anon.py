import sys
import os
import unittest
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from pathlib import Path
import pandas as pd
import anon
logging.basicConfig(level=logging.ERROR)

class AnonTest(unittest.TestCase):
    
    def setUp(self):

        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')

        self.ano_champs =  ["Nom","Prénom","Téléphone","Adresse électronique","Adresse postale","Code postal / Commune",
                 "Nom (3 premières lettres)","Sexe","Date de naissance","ou âge (réel ou estimé)","Poids",
                 "Date de survenue","Période de survenue","Type de lieu de survenue","Date de survenue de l'incident",
                 "Date de détection de l'incident","Profession","Complément profession",
                 "Téléphone ou pour les correspondants locaux : Télécopie",
                 "Nom de l'établissement ou de l'organisme", "Age (réel ou estimé)",
                 "Êtes vous la personne ayant présenté les symptômes  ?",
                 "Veuillez préciser votre lien avec la personne",
                 "Nom de l'établissement ou de l'organisme",
                 "Êtes-vous le correspondant de matériovigilance ou son suppléant ?",
                 "Êtes-vous le correspondant de réactovigilance ou son suppléant ?",
                 "Si vous êtes dans un établissement de santé, il est préférable de prendre contact avec votre correspondant local de matériovigilance qui a pour rôle de recevoir et transmettre ensuite les déclarations à l'ansm. Sinon, vous pouvez les transmettre à l'ansm vous-même via ce portail."
                 ]
    
    def test_find_files(self):
        file_list =anon.findfiles('*.csv', self.data_dir)
        
        self.assertEqual(len(file_list), 5)
    
    def test_anon_extract(self):
        extract = []
        filenames = []
        list_of_csv =anon.findfiles('*.csv', self.data_dir)
        for file in list_of_csv:
            filename_check = Path(file).name
            df = pd.read_csv(file, sep="|", index_col=0)
            df['Column0'] = df['Column0'].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=[" "," "], regex=True)
            df = df[~df['Column0'].str.lower().isin([x.lower() for x in self.ano_champs])]
            extract.append(df)
            filenames.append(filename_check)

        for name, table in zip(filenames,extract):
            if name == '2dm_MATERIOVIGILANCE_20200414150210003.pdf.csv':
                print(name, table.shape)
                self.assertEqual(table.shape, (42,2))
            elif name == 'citoyen_MATERIOVIGILANCE_20200414143453744.pdf.csv':
                print(name, table.shape)
                self.assertEqual(table.shape, (30,2))
            elif name == 'citoyen_REACTOVIGILANCE_20200414143914639.pdf.csv':
                print(name, table.shape)
                self.assertEqual(table.shape, (22,2))
            elif name == 'pro_MATERIOVIGILANCE_20200414145304272.pdf.csv':
                print(name, table.shape)
                self.assertEqual(table.shape, (37,2))
            elif name == 'pro_REACTOVIGILANCE_20200414145808560.pdf.csv':
                print(name, table.shape)
                self.assertEqual(table.shape, (33,2))

        
if __name__ == "__main__":
    unittest.main()
