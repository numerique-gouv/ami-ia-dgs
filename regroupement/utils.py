"""
Auteur: Quillivic Robin, Data scientist chez Starclay, rquillivic@straclay.fr 
"""


import logging
import os
import ast
import yaml




def loading_function(path, method,obj,logger):
    """Fonction générique pour charger un objet en Python

    Args:
        path (str): emplacement du fichier
        method (function): méthode pour charger un l'objet obj
        obj (): objet à charger
        logger (logging.logger): logger associé 

    Raises:
        ValueError: [description]
        ValueError: [description]
    """
    if type(path)==str:
        if not os.path.exists(path):
            logger.error(f'path {path} does not exists')
            raise ValueError('file does not exist')
        else : 
            logger.info(f'path {path} already exists')
    try :
        obj = method(path)
        logger.info(f'loaded object from {path}')
        return obj
        
    except Exception as e : 
        logger.error(f'Error loading {path}: {e}')
        raise ValueError(f'{path} : {e}') 
