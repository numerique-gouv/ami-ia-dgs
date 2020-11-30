import sys, os
sys.path.append(os.path.abspath('..'))
import unittest
import requests
from pprint import pprint


class ClusterModelTest(unittest.TestCase):

    def setUp(self):
        self.end_point = '/clusters/model'
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

    def _check_clustermodel_nbtopics(self, json_response):
        self.assertIn('nb_clusters', json_response)
        self.assertIsInstance(json_response['nb_clusters'], int)

    def _check_clustermodel_scores(self, json_response):
        self.assertIn('scores', json_response)
        self.assertIsInstance(json_response['scores'], dict)
        for k, v in json_response['scores'].items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, float)

    def _check_clustermodel_distmat(self, json_response):
        self.assertIn('distances_matrix', json_response)
        self.assertIsInstance(json_response['distances_matrix'], list)
        self.assertIsInstance(json_response['distances_matrix'][0], list)
        self.assertIsInstance(json_response['distances_matrix'][0][0], float)
        for i in range(len(json_response['distances_matrix'])):
            self.assertEqual(len(json_response['distances_matrix']), len(json_response['distances_matrix'][i]))

    def _check_clustermodel_pca(self, json_response):
        self.assertIn('pca', json_response)
        self.assertIsInstance(json_response['pca'], dict)
        self.assertIn('clusters_ind', json_response['pca'])
        self.assertIsInstance(json_response['pca']['clusters_ind'], list)
        self.assertIsInstance(json_response['pca']['clusters_ind'][0], int)
        self.assertIn('X', json_response['pca'])
        self.assertIsInstance(json_response['pca']['X'], list)
        self.assertIsInstance(json_response['pca']['X'][0], float)
        self.assertIn('Y', json_response['pca'])
        self.assertIsInstance(json_response['pca']['Y'], list)
        self.assertIsInstance(json_response['pca']['Y'][0], float)
        self.assertIn('weights', json_response['pca'])
        self.assertIsInstance(json_response['pca']['weights'], list)
        self.assertIsInstance(json_response['pca']['weights'][0], float)

    def test_clustermodel_nbclusters_ok(self):
        # When
        response = self._get_route(self.end_point + '/nb_clusters')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_clustermodel_nbtopics(response.json)

    def test_clustermodel_scores_ok(self):
        # When
        response = self._get_route(self.end_point + '/scores')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_clustermodel_scores(response.json)

    def test_clustermodel_distmat_ok(self):
        # When
        response = self._get_route(self.end_point + '/distances')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_clustermodel_distmat(response.json)

    def test_clustermodel_pca_ok(self):
        # When
        response = self._get_route(self.end_point + '/pca')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_clustermodel_pca(response.json)

    def test_clustermodel_complete_ok(self):
        # When
        response = self._get_route(self.end_point)

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_clustermodel_nbtopics(response.json)
        self._check_clustermodel_scores(response.json)
        self._check_clustermodel_distmat(response.json)
        self._check_clustermodel_pca(response.json)

    # def tearDown(self):
    #     pass

