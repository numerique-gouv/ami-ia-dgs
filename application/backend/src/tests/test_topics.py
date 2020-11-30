import sys, os
sys.path.append(os.path.abspath('..'))
import unittest
import requests
from pprint import pprint


class TopicsTest(unittest.TestCase):

    def setUp(self):
        self.end_point = '/topics/{topic_ind}'
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

    def _check_topic_weight(self, topic_id, json_response):
        self.assertIn('topic_id', json_response)
        self.assertIsInstance(json_response['topic_id'], int)
        self.assertEqual(json_response['topic_id'], topic_id)
        self.assertIn('weight', json_response)
        self.assertIsInstance(json_response['weight'], float)

    def _check_topic_documents(self, topic_id, json_response):
        self.assertIn('topic_id', json_response)
        self.assertIsInstance(json_response['topic_id'], int)
        self.assertEqual(json_response['topic_id'], topic_id)
        self.assertIn('documents', json_response)
        self.assertIsInstance(json_response['documents'], list)
        for elt in json_response['documents']:
            self.assertIsInstance(elt, dict)
            self.assertIn('doc_name', elt)
            self.assertIsInstance(elt['doc_name'], str)
            self.assertIn('topic_score', elt)
            self.assertIsInstance(elt['topic_score'], float)

    def _check_topic_wordcloud(self, topic_id, json_response):
        self.assertIn('topic_id', json_response)
        self.assertIsInstance(json_response['topic_id'], int)
        self.assertEqual(json_response['topic_id'], topic_id)
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
        topic_ind = 1
        response = self._get_route(self.end_point.format(topic_ind=topic_ind)+'/weight')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topic_weight(topic_ind, response.json)

    def test_documents_ok(self):
        # When
        topic_ind = 1
        response = self._get_route(self.end_point.format(topic_ind=topic_ind)+'/documents')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topic_documents(topic_ind, response.json)

    def test_wordcloud_ok(self):
        # When
        topic_ind = 1
        response = self._get_route(self.end_point.format(topic_ind=topic_ind)+'/wordcloud')

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topic_wordcloud(topic_ind, response.json)

    def test_complete_ok(self):
        # When
        topic_ind = 1
        response = self._get_route(self.end_point.format(topic_ind=topic_ind))

        # Then
        if self.verbose:
            pprint(response.json)

        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json, dict)
        self._check_topic_weight(topic_ind, response.json)
        self._check_topic_documents(topic_ind, response.json)
        self._check_topic_wordcloud(topic_ind, response.json)

    def test_ressource_failure(self):
        # When
        for endpoint in ['', '/weight', '/documents', '/wordcloud']:
            response = self._get_route(self.end_point.format(topic_ind=0) + endpoint)
            self.assertEqual(404, response.status_code)

            response = self._get_route(self.end_point.format(topic_ind="test")+endpoint)
            self.assertEqual(404, response.status_code)

            response = self._get_route(self.end_point.format(topic_ind='')+endpoint)
            self.assertEqual(404, response.status_code)

    # def tearDown(self):
    #     pass
