"""
Auteur: 
    Quillivic Robin,  Data Scientist chez StarClay, rquilivic@starclay.fr
    Poulet Cyril, Data Scientist chez StarClay, cpoulet@starclay.fr
Description:
    fichier permetant l'optimisation des des hyper-paramètres du topic-model :
    - nombre de topics
    - choix des colonnes

"""

import warnings

warnings.filterwarnings('ignore')

import os, sys
import yaml
import pandas as pd

from train_topic import TopicModel

import optuna
from optuna import Trial

current_dir = os.path.dirname(__file__)
config_file = 'training_config.yaml'

if len(sys.argv) == 2:
    config_file = sys.argv[1]

if not os.path.isabs(config_file):
    config_file = os.path.abspath(os.path.join(current_dir, config_file))

with open(config_file, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


filename = os.path.join(config['data']['path'], config['data']['filename'])
if not os.path.isabs(filename):
    filename = os.path.abspath(os.path.join(current_dir, filename))
n_ligne = None
try:
    df = pd.read_pickle(filename)
    if n_ligne is not None:
        df = df.iloc[:n_ligne, :]
    print('Chargement des données ! Ok !')
except Exception as e:
    print(f'Error loading {filename}: {e}')
    raise ValueError(e)


def objective(trial: Trial):
    """Fonction objectif pour l'optimiseur Optuna

    Args:
        trial (Trial): Un essai de paramètre

    Returns:
        f1 (float): score f1 sample pour le TEF_ID
    """
    try_name = os.path.join(config['config_name'], str(trial.number))
    save_dir = os.path.join(config['path_to_save'], try_name)
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(os.path.join(current_dir, save_dir))
    os.makedirs(save_dir, exist_ok=True)

    current_config = config['topic']
    current_config['model']['num_topic'] = trial.suggest_int("num_topic", 150, 160)
    current_config['dictionary']['used_columns'] = trial.suggest_categorical("used_columns",
                [
                    ['text_lem', 'rake_kw'],
                    ['text_lem', 'med_term'],
                    ['rake_kw','med_term'],
                    ['text_lem', 'rake_kw','med_term_uniq'],
                    ['rake_kw','med_term_uniq'],
                    ['text_lem','med_term_uniq'],
                    ['trigram', 'rake_kw','med_term'],
                    ['trigram', 'rake_kw','med_term_uniq'],
                    ['trigram', 'med_term_uniq']
                ])

    topic_model = TopicModel(os.path.basename(try_name), current_config, save_dir=save_dir)
    topic_model.build_dictionary(df)
    topic_model.build_corpus()
    topic_model.build_model()
    topic_model.build_doc_topic_mat()
    score = topic_model.evaluate(save=True)
    topic_model.logger.info(f'Coherence score : {score}')

    trial.report(score, step=current_config['model']['passes'])
    return score


# Optimisation
studyName = 'topic_study'
maximum_time = 30 * 60 * 60  # second
number_of_random_points = 50


optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
study = optuna.create_study(study_name=studyName, direction="maximize")
study.optimize(objective, n_trials=number_of_random_points, timeout=maximum_time)  

# Sauvegarde du resultat
df = study.trials_dataframe()
df.to_json(os.path.join(config['path_to_save'], config['config_name'], config['config_name'] + '.json'))
print(study.best_trial)