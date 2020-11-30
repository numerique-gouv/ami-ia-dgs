import os
import joblib
import logging
import yaml

import sys
sys.path.append(os.path.dirname((os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))

                

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)


class GraviteModel:
    def __init__(self):
        self.model = None
        self.binaire = False
        self._logger = logging.getLogger(self.__class__.__name__)

    def load(self, path_to_models):
        """
        Charge les modèles depuis le chemin indiqué: path_to_model

        Args:
            path_to_models (str): cheminvers les modèles

        Returns: 
            None

        Raises:
            ValueError: value error si le modèle ne peut être chargé
            ValueError: value error si le fichier n'existe pas
        """
        if self.binaire == False:
            model_filename = config_data['training']['models']['model_gravite']['non_binaire']['model_filename']
        else:
            model_filename = config_data['training']['models']['model_gravite']['binaire']['model_filename']

        filename = os.path.join(path_to_models, model_filename)
        if not os.path.exists(filename):
            self._logger.error(f'path {filename} does not exists')
            raise ValueError('Model file does not exist')
        try:
            self.model = joblib.load(filename)
            self._logger.info(f'loaded model from {filename}')
        except Exception as e:
            self._logger.error(f'Error loading {filename}: {e}')
            raise ValueError(f'{filename} is not a model: {e}')

    def predict(self, X):
        if self.model is None:
            raise RuntimeError('No model loaded')

        return self.model.predict_proba(X)
