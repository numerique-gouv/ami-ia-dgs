import sys
import os
import unittest
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import main_plumber

logging.basicConfig(level=logging.ERROR)

class PlumberTest(unittest.TestCase):
    
    def setUp(self):

        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    def test_find_files(self):
        file_list =main_plumber.findfiles('*.pdf', self.data_dir)
        
        self.assertEqual(len(file_list), 5)
    
    def test_plumber_extract(self):
        extract = []
        filenames = []
        file_list =main_plumber.findfiles('*.pdf', self.data_dir)
        for file in file_list:
            _, char_df =main_plumber.plumber_df(str(file))
            extract.append(char_df)
            filenames.append(file.stem)

        for name, table in zip(filenames,extract):
            if name == '2dm_MATERIOVIGILANCE_20200414150210003':
                print(name, table.shape)
                self.assertEqual(table.shape, (58,2))
            elif name == 'citoyen_MATERIOVIGILANCE_20200414143453744':
                print(name, table.shape)
                self.assertEqual(table.shape, (46,2))
            elif name == 'citoyen_REACTOVIGILANCE_20200414143914639':
                print(name, table.shape)
                self.assertEqual(table.shape, (38,2))
            elif name == 'pro_MATERIOVIGILANCE_20200414145304272':
                print(name, table.shape)
                self.assertEqual(table.shape, (55,2))
            elif name == 'pro_REACTOVIGILANCE_20200414145808560':
                print(name, table.shape)
                self.assertEqual(table.shape, (51,2))

        
if __name__ == "__main__":
    unittest.main()
