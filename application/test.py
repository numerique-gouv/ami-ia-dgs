import requests
from requests.auth import HTTPBasicAuth
from pprint import pprint

adresse_api = 'http://localhost:5000/dgs-api'


def test_preditions():
    test_file = '../test/test_plumber/data/2dm_MATERIOVIGILANCE_20200414150210003.pdf'
    with open(test_file, 'rb') as f:
        r = requests.post(adresse_api + '/predict/dco', files={'file': f})
        pprint(r.json())

    with open(test_file, 'rb') as f:
        r = requests.post(adresse_api + '/predict/dysfonctionnement', files={'file': f})
        pprint(r.json())

    with open(test_file, 'rb') as f:
        r = requests.post(adresse_api + '/predict/consequence', files={'file': f})
        pprint(r.json())

    with open(test_file, 'rb') as f:
        r = requests.post(adresse_api + '/predict/effet', files={'file': f})
        pprint(r.json())

    with open(test_file, 'rb') as f:
        r = requests.post(adresse_api + '/predict/gravite', files={'file': f})
        pprint(r.json())

    with open(test_file, 'rb') as f:
        r = requests.post(adresse_api + '/predict/all_models', files={'file': f})
        pprint(r.json())

# requests.post(adresse_api + '/users', json={'username': 'toto', 'password': 'tata'})

res = requests.get(adresse_api + '/documents/{}'.format('R1700004'),
                   auth=HTTPBasicAuth('toto', 'titi'))
pprint(res.json())