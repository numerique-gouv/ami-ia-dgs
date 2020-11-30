"""
Script de création d'utilisateurs. Il vous faut le login/mdp admin pour cela
"""
import requests
from requests.auth import HTTPBasicAuth

# adresse du backend
adresse_api = 'http://dgcl-aclia.starclay.fr'
# login admin
admin_username = 'login_admin'
admin_password = 'mdp_admin'

users = {
    'user1': 'mdp1',
    ...
}


for user, mdp in users.items():
    res = requests.post(adresse_api + '/dgs-api/users',
                        json={'username': user, 'password': mdp},
                        auth=HTTPBasicAuth(admin_username, admin_password))
    if res.status_code == 201:
        print(f'User {user} created')
    else:
        print(f'Error creating user {user}')