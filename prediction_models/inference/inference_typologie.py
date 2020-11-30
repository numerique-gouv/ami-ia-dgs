import os
import joblib
import logging
import yaml
import tensorflow as tf


with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'), 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.FullLoader)


class TypologieModel:

    def __init__(self):
        self.model = None
        self.pipeline = None
        self.typo = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def load(self, path_to_models):
        """
        Charge les modèles et les pipelines depuis les chemin indiqué: path_to_model

        Args:
            path_to_models (str): chemin vers les modèles
        
        Returns: 
            None

        Raises:
            ValueError: value error si le modèle ne peut être chargé
            ValueError: [description]
        """
        try: 
            model_filename = config_data['training']['models']['model_typologie'][self.typo]['model_filename']
            pipeline_filename =  config_data['training']['models']['model_typologie'][self.typo]['pipeline_filename']
        except: 
            self._logger.error(f'{typo} is None, need to be tef or cdy or tdy')
            raise ValueError(f'{typo} is None, need to be tef or cdy or tdy') 
            
        model_filename = os.path.join(path_to_models, model_filename)
        pipeline_filename = os.path.join(path_to_models, pipeline_filename)
        
        if not os.path.exists(model_filename):
            self._logger.error(f'path {model_filename} does not exists')
            raise ValueError('Model file does not exist')
        
        if not os.path.exists(pipeline_filename):
            self._logger.error(f'path {pipeline_filename} does not exists')
            raise ValueError('Pipeline file does not exist')
        
        try:
            self.pipeline = joblib.load(pipeline_filename)
            self._logger.info(f'loaded pipeline from {pipeline_filename}')
        except Exception as e:
            self._logger.error(f'Error loading {pipeline_filename}: {e}')
            raise ValueError(f'{pipeline_filename} is not a model: {e}')
        
        try:
            self.model = tf.keras.models.load_model(model_filename)
            self._logger.info(f'loaded model from {model_filename}')
        except Exception as e:
            self._logger.error(f'Error loading {model_filename}: {e}')
            raise ValueError(f'{model_filename} is not a model: {e}')
        
    def transform(self,X): 
        if self.pipeline is None: 
            raise RuntimeError('No pipeline loaded')
        
        return self.pipeline.transform(X)
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError('No model loaded')

        return self.model.predict(X)
