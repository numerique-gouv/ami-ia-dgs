import requests
from requests.auth import HTTPBasicAuth
from pprint import pprint
import os
import json

adresse_api = 'http://localhost:5000/dgs-api'


def test_predictions():
    keys = []
    N = len(os.listdir('bugs')[:60])
    for i, f in enumerate(os.listdir('bugs')[:60]):
        # if f != 'PS_MATERIOVIGILANCE_20201006172135034.pdf.csv':
        #     continue
        print(f'{i}/{N} - {f}')
        with open(os.path.join('bugs', f), 'r') as f:
            r = requests.post(adresse_api + '/predict/all_models', files={'file': f},
                              auth=HTTPBasicAuth('titi', 'toto'))
            keys.append(r.json()['last_results_key'])
            pprint(r.json())

    r = requests.post(adresse_api + '/predict/last_results/csv', data=json.dumps({'last_results_keys': keys}),
                      auth=HTTPBasicAuth('titi', 'toto'))
    with open('output_direct.csv', 'w') as out:
        out.write(r.text)



test_predictions()
# res = requests.get(adresse_api + '/documents/{}'.format('R2000471'),
#                    auth=HTTPBasicAuth('titi', 'toto'))
# pprint(res.json())