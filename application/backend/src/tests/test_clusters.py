import sys, os
sys.path.append(os.path.abspath('..'))
import unittest
import requests
from pprint import pprint


class ClustersTest(unittest.TestCase):

    def setUp(self):
        self.end_point = '/clusters/{cluster_id}'
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

    def _check_cluster_weight(self, cluster_id, json_response):
        self.assertIn('cluster_id', json_response)
        self.assertIsInstance(json_response['cluster_id'], int)
        self.assertEqual(json_response['cluster_id'], cluster_id)
        self.assertIn('weight', json_response)
        self.assertIsInstance(json_response['weight'], dict)
        self.assertIn('weight', json_response['weight'])
        self.assertIsInstance(json_response['weight']['weight'], float)
        self.assertIn('nb_docs', json_response['weight'])
        self.assertIsInstance(json_response['weight']['nb_docs'], int)

    def _check_cluster_documents(self, cluster_id, json_response):
        self.assertIn('cluster_id', json_response)
        self.assertIsInstance(json_response['cluster_id'], int)
        self.assertEqual(json_response['cluster_id'], cluster_id)
        self.assertIn('documents', json_response)
        self.assertIsInstance(json_response['documents'], list)
        for elt in json_response['documents']:
            self.assertIsInstance(elt, dict)
            self.assertIn('doc_name', elt)
            self.assertIsInstance(elt['doc_name'], str)
            self.assertIn('document_similarity', elt)
            self.assertIsInstance(elt['document_similarity'], float)

    def _check_cluster_topics(self, cluster_id, json_response):
        self.assertIn('cluster_id', json_response)
        self.assertIsInstance(json_response['cluster_id'], int)
        self.assertEqual(json_response['cluster_id'], cluster_id)
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

    def _check_cluster_dcos(self, cluster_id, json_response):
        self.assertIn('cluster_id', json_response)
        self.assertIsInstance(json_response['cluster_id'], int)
        self.assertEqual(json_response['cluster_id'], cluster_id)
        self.assertIn('dcos', json_response)
        self.assertIsInstance(json_response['dcos'], list)
        for elt in json_response['dcos']:
            self.assertIsInstance(elt, dict)
            self.assertIn('label', elt)
            self.assertIsInstance(elt['label'], float)
            self.assertIn('topic', elt)
            self.assertIsInstance(elt['topic'], str)
            self.assertIn('value', elt)
            self.assertIsInstance(elt['value'], float)
            self.assertIn('tooltip', elt)
            self.assertIsInstance(elt['tooltip'], float)

    def _check_cluster_wordcloud(self, cluster_id, json_response):
        self.assertIn('cluster_id', json_response)
        self.assertIsInstance(json_response['cluster_id'], int)
        self.assertEqual(json_response['cluster_id'], cluster_id)
        self.assertIn('wordcloud', json_response)
        self.assertIsInstance(json_response['wordcloud'], list)
        for elt in json_response['wordcloud']:
            self.assertIsInstance(elt, dict)
            self.assertIn('word', elt)
            self.assertIsInstance(elt['word'], str)
            self.assertIn('weight', elt)
            self.assertIsInstance(elt['weight'], float)

    def test_weight_ok(self):
        # When
        cluster_id = 1
        response = self._get_route(self.end_point.format(cluster_id=cluster_id)+'/weight')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_cluster_weight(cluster_id, response.json)

    def test_documents_ok(self):
        # When
        cluster_id = 1
        response = self._get_route(self.end_point.format(cluster_id=cluster_id)+'/documents')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_cluster_documents(cluster_id, response.json)

    def test_topics_ok(self):
        # When
        cluster_id = 1
        response = self._get_route(self.end_point.format(cluster_id=cluster_id)+'/topics')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_cluster_topics(cluster_id, response.json)

    def test_dcos_ok(self):
        # When
        cluster_id = 1
        response = self._get_route(self.end_point.format(cluster_id=cluster_id)+'/dcos')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_cluster_dcos(cluster_id, response.json)

    def test_wordcloud_ok(self):
        # When
        cluster_id = 1
        response = self._get_route(self.end_point.format(cluster_id=cluster_id)+'/wordcloud')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_cluster_wordcloud(cluster_id, response.json)

    def test_complete_ok(self):
        # When
        cluster_id = 1
        response = self._get_route(self.end_point.format(cluster_id=cluster_id))

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_cluster_weight(cluster_id, response.json)
        self._check_cluster_documents(cluster_id, response.json)
        self._check_cluster_topics(cluster_id, response.json)
        self._check_cluster_dcos(cluster_id, response.json)
        self._check_cluster_wordcloud(cluster_id, response.json)

    def test_ressource_failure(self):
        # When
        for endpoint in ['', '/weight', '/documents', '/topics', '/dcos', '/wordcloud']:
            # response = self._get_route(self.end_point.format(cluster_id=0) + endpoint)
            # self.assertEqual(404, response.status_code)

            response = self._get_route(self.end_point.format(cluster_id="test")+endpoint)
            self.assertEqual(404, response.status_code)

            response = self._get_route(self.end_point.format(cluster_id='')+endpoint)
            self.assertEqual(404, response.status_code)

    # def tearDown(self):
    #     pass
