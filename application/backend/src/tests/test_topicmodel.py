import sys, os
sys.path.append(os.path.abspath('..'))
import unittest
import requests
from pprint import pprint


class TopicModelTest(unittest.TestCase):

    def setUp(self):
        self.end_point = '/topics/model'
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

    def _check_topicmodel_nbtopics(self, json_response):
        self.assertIn('nb_topics', json_response)
        self.assertIsInstance(json_response['nb_topics'], int)

    def _check_topicmodel_coherence(self, json_response):
        self.assertIn('coherence_score', json_response)
        self.assertIsInstance(json_response['coherence_score'], float)

    def _check_topicmodel_distmat(self, json_response):
        self.assertIn('distances_matrix', json_response)
        self.assertIsInstance(json_response['distances_matrix'], list)
        self.assertIsInstance(json_response['distances_matrix'][0], list)
        self.assertIsInstance(json_response['distances_matrix'][0][0], float)
        for i in range(len(json_response['distances_matrix'])):
            self.assertEqual(len(json_response['distances_matrix']), len(json_response['distances_matrix'][i]))
            
    def _check_topicmodel_pca(self, json_response):
        self.assertIn('pca', json_response)
        self.assertIsInstance(json_response['pca'], str)

    def test_topicmodel_nbtopics_ok(self):
        # When
        response = self._get_route(self.end_point + '/nb_topics')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topicmodel_nbtopics(response.json)

    def test_topicmodel_coherence_ok(self):
        # When
        response = self._get_route(self.end_point + '/coherence')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topicmodel_coherence(response.json)

    def test_topicmodel_distmat_ok(self):
        # When
        response = self._get_route(self.end_point + '/distances')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topicmodel_distmat(response.json)
        
    def test_topicmodel_pca_ok(self):
        # When
        response = self._get_route(self.end_point + '/pca')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topicmodel_pca(response.json)

    def test_topicmodel_complete_ok(self):
        # When
        response = self._get_route(self.end_point)

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topicmodel_nbtopics(response.json)
        self._check_topicmodel_coherence(response.json)
        self._check_topicmodel_distmat(response.json)
        self._check_topicmodel_pca(response.json)

    # def tearDown(self):
    #     pass
