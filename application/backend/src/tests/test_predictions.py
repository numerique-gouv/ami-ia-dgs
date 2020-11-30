import sys, os
sys.path.append(os.path.abspath('..'))
import unittest
import requests
from pprint import pprint


class PredictionsTest(unittest.TestCase):

    def setUp(self):
        self.end_point = '/predict/{model_name}'
        self.app = None
        self.docker_adress = os.environ.get('TEST_DOCKER_ADRESS', None)
        if self.docker_adress is None:
            from src import main_server
            self.app = main_server.app.test_client()
        self.verbose = os.environ.get('TEST_VERBOSE', False)

    def _send_file(self, api_path, test_file):
        if self.docker_adress is not None:
            with open(test_file, 'rb') as f:
                r = requests.post(f'http://{self.docker_adress}{api_path}', files={'file': f})
                r.json = r.json()
                return r
        else:
            with open(test_file, 'rb') as f:
                return self.app.post(api_path, files={'file': f})

    def _check_prediction(self, doc_id, model_names, json_response):
        self.assertIn(doc_id, json_response)
        self.assertIsInstance(json_response[doc_id], list)
        for i, res in enumerate(json_response[doc_id]):
            self.assertIn('model_name', res)
            self.assertIsInstance(res['model_name'], str)
            self.assertEqual(res['model_name'], model_names[i])
            self.assertIn('predictions', res)
            self.assertIsInstance(res['predictions'], dict)

            self.assertIn('columns', res['predictions'])
            self.assertIsInstance(res['predictions']['columns'], list)
            for c in res['predictions']['columns']:
                self.assertIsInstance(c, dict)
                self.assertIn('title', c)
                self.assertIsInstance(c['title'], str)
                self.assertIn('dataIndex', c)
                self.assertIsInstance(c['dataIndex'], str)
                self.assertIn('key', c)
                self.assertIsInstance(c['key'], str)

            self.assertIn('datasource', res['predictions'])
            self.assertIsInstance(res['predictions']['datasource'], list)
            for d in res['predictions']['datasource']:
                self.assertIsInstance(d, dict)
                self.assertIn('key', d)
                self.assertIsInstance(d['key'], int)
                self.assertIn('cat', d)
                self.assertIsInstance(d['cat'], str)
                self.assertIn('proba', d)
                self.assertIsInstance(d['proba'], float)

    def test_classification_ok(self):
        # When
        model_names = {
            'dco': ['DCO'],
            'dysfonctionnement': ['dysfonctionnement'],
            'consequence': ['consequence'],
            'effet': ['effet'],
            'gravite': ['gravité_ordinale', 'gravité_binaire']
        }
        test_file = os.path.abspath(
                        os.path.join(os.path.abspath(__file__),
                                     '../../data/test/2dm_MATERIOVIGILANCE_20200414150210003.pdf'))
        file_id = os.path.basename(test_file)
        for model in model_names:
            response = self._send_file(self.end_point.format(model_name=model), test_file)

            # Then
            if self.verbose:
                pprint(response.json)

            self.assertEqual(200, response.status_code)
            self.assertIsInstance(response.json, dict)
            self._check_prediction(file_id, model_names[model], response.json)

    def test_clusterization_ok(self):
        # When
        test_file = os.path.abspath(
                        os.path.join(os.path.abspath(__file__),
                                     '../../data/test/2dm_MATERIOVIGILANCE_20200414150210003.pdf'))
        file_id = os.path.basename(test_file)
        response = self._send_file(self.end_point.format(model_name='clustering'), test_file)

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        # self._check_clustering(file_id, response.json)