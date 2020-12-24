import requests
from requests.auth import HTTPBasicAuth
from pprint import pprint
import os
import json

adresse_api = 'http://localhost:5000/dgs-api'


def test_predictions():
    keys = []
    for f in os.listdir('bugs'):
        print(f)
        with open(os.path.join('bugs', f), 'r') as f:
            r = requests.post(adresse_api + '/predict/all_models', files={'file': f},
                              auth=HTTPBasicAuth('dgs_admin', 'Q0a*1dYjD5LO'))
            keys.append(r.json()['last_results_key'])
            pprint(r.json())

    r = requests.post(adresse_api + '/predict/last_results/csv', data=json.dumps({'last_results_keys': keys}),
                      auth=HTTPBasicAuth('dgs_admin', 'Q0a*1dYjD5LO'))
    with open('output.csv', 'w') as out:
        out.write(r.text)



test_predictions()
# res = requests.get(adresse_api + '/documents/{}'.format('R1916936%20-%20VENTOUSE%20(%20UROLOGIE-GYNECOLOGIE%20)'),
#                    auth=HTTPBasicAuth('dgs_admin', 'Q0a*1dYjD5LO'))
# pprint(res.json())