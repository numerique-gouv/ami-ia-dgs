import sys, os
sys.path.append(os.path.abspath('..'))
import unittest
import requests
from pprint import pprint


class DocumentTest(unittest.TestCase):

    def setUp(self):
        self.end_point = '/documents/{doc_id}'
        self.app = None
        self.docker_adress = os.environ.get('TEST_DOCKER_ADRESS', None)
        if self.docker_adress is None:
            from src import main_server
            self.app = main_server.app.test_client()
        self.verbose = os.environ.get('TEST_VERBOSE', False)

    def _get_route(self, api_path):
        if self.docker_adress is not None:
            res = requests.get(f'http://{self.docker_adress}{api_path}')
            res.json = res.json()
            return res
        else:
            return self.app.get(api_path)

    def _check_doc_info(self, doc_id, json_response):
        self.assertIn('doc_id', json_response)
        self.assertIsInstance(json_response['doc_id'], str)
        self.assertEqual(json_response['doc_id'], doc_id)
        self.assertIn('DCO', json_response)
        self.assertIsInstance(json_response['DCO'], str)
        self.assertIn('Description incident', json_response)
        self.assertIsInstance(json_response['Description incident'], str)
        self.assertIn('Etat patient', json_response)
        self.assertIsInstance(json_response['Etat patient'], str)
        self.assertIn('Numero de déclaration', json_response)
        self.assertIsInstance(json_response['Numero de déclaration'], str)
        self.assertEqual(json_response['Numero de déclaration'], doc_id)

    def _check_doc_cluster(self, doc_id, json_response):
        self.assertIn('doc_id', json_response)
        self.assertIsInstance(json_response['doc_id'], str)
        self.assertEqual(json_response['doc_id'], doc_id)
        self.assertIn('cluster', json_response)
        self.assertIsInstance(json_response['cluster'], int)

    def _check_doc_topics(self, doc_id, json_response):
        self.assertIn('doc_id', json_response)
        self.assertIsInstance(json_response['doc_id'], str)
        self.assertEqual(json_response['doc_id'], doc_id)
        self.assertIn('topics', json_response)
        self.assertIsInstance(json_response['topics'], list)
        for elt in json_response['topics']:
            self.assertIsInstance(elt, dict)
            self.assertIn('label', elt)
            self.assertIsInstance(elt['label'], float)
            self.assertIn('topic', elt)
            self.assertIsInstance(elt['topic'], str)
            self.assertIn('value', elt)
            self.assertIsInstance(elt['value'], float)
            self.assertIn('tooltip', elt)
            self.assertIsInstance(elt['tooltip'], str)

    def test_all_docs_ok(self):
        # When
        response = self._get_route(self.end_point.format(doc_id='all_ids'))

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self.assertIn('doc_ids', response.json)
        self.assertIsInstance(response.json['doc_ids'], list)
        self.assertIn('nb_docs', response.json)
        self.assertIsInstance(response.json['nb_docs'], int)
        self.assertEqual(len(response.json['doc_ids']), response.json['nb_docs'])

    def test_content_ok(self):
        # When
        doc_id = 'R1700004'
        response = self._get_route(self.end_point.format(doc_id=doc_id)+'/content')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_doc_info(doc_id, response.json)

    def test_cluster_ok(self):
        # When
        doc_id = 'R1700004'
        response = self._get_route(self.end_point.format(doc_id=doc_id)+'/cluster')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_doc_cluster(doc_id, response.json)

    def test_topics_ok(self):
        # When
        doc_id = 'R1700004'
        response = self._get_route(self.end_point.format(doc_id=doc_id)+'/topics')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_doc_topics(doc_id, response.json)

    def test_complete_ok(self):
        # When
        doc_id = 'R1700004'
        response = self._get_route(self.end_point.format(doc_id=doc_id))

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_doc_info(doc_id, response.json)
        self._check_doc_cluster(doc_id, response.json)
        self._check_doc_topics(doc_id, response.json)

    def test_ressource_failure(self):
        # When
        for endpoint in ['', '/info', '/cluster', '/topics']:
            response = self._get_route(self.end_point.format(doc_id="test")+endpoint)
            self.assertEqual(404, response.status_code)

            response = self._get_route(self.end_point.format(doc_id='')+endpoint)
            self.assertEqual(404, response.status_code)

    # def tearDown(self):
    #     pass
