"""
Script principal de l'API

:author: cpoulet@starclay.fr
"""
import os
from flask import Flask, flash, request, redirect, url_for, abort, jsonify, send_file, after_this_request, Response
from flask_cors import CORS, cross_origin
from flask_httpauth import HTTPBasicAuth
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
import json
import logging.config
import magic
from functools import lru_cache
import secrets
from multiprocessing import Lock
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import copy
import pandas as pd

from users.user import User
from prediction.models import *
from clustering.prediction import clusterize_doc
from clustering.clustering_models import ClusteringModels
import clustering.analysis as clusters
from backend_utils.config_parser import get_local_file, parse_full_config
import backend_utils.input_handling as input_handling
import backend_utils.antpro_formatting as antpro_formatting
from backend_utils.output_handling import results_to_file

# Chargement de la config
config = parse_full_config(get_local_file('config.yaml'), get_local_file('config_env.yaml'))
logging.config.fileConfig(get_local_file('logging.ini'))
logging.getLogger().setLevel(config['app']['log_level'])

logger = logging.getLogger('Flask')
logger.debug('Creating app')

# Création de l'app, déclaration des variables principales
# base port is 5000
app = Flask(__name__)
auth = HTTPBasicAuth()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xml', 'pdf', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# on regarde si on active les fonctions utilisées par le front ou seulement la prédiction
ACTIVATE_FRONT = config['app']['activate_front'] in [True, 'True']

# load data for preprocessing
with open(os.path.abspath(os.path.join(config['data']['mrv']['path'], 'colonnes.json')), 'r') as file:
    colonnes = json.load(file)
with open(os.path.abspath(os.path.join(config['data']['mrv']['path'], 'mapping.json')), 'r') as file:
    mapping = json.load(file)

m = magic.Magic()


##################
# Gestion du stockage des derniers résultats pour export csv/excel
#
# La récupération se fait une seule fois, et les résultats non récupérés sont nettoyés au bout d'un moment
##################
last_results = {}
last_results_lock = Lock()
last_result_lifetime = timedelta(minutes=10)
last_results_cleaning_period_in_min = config['app']['clean_results_in_min']
max_nb_downloads = config['app']['max_nb_downloads']


def add_last_results(loaded_data, res):
    """
    Stockage des data et des résultats. Renvoie un token d'identification

    :param loaded_data: {f: datafram}
    :param res: dict {f: [{model_name: mn, predictions: dataframe}]}
    :return: token str
    """
    global last_results, last_results_lock
    key = secrets.token_urlsafe(32)
    with last_results_lock:
        last_results[key] = {'data': loaded_data, 'res': res, 'datetime': datetime.now(), 'nb_downloads': 0}
    return key


def get_last_results(key):
    """
    Récupération des data et des résultats associées à un token

    :param key: token str
    :return: loaded_data: {f: datafram}, res: dict {f: [{model_name: mn, predictions: dataframe}]}
    """
    global last_results, last_results_lock
    with last_results_lock:
        saved = last_results.get(key, {})
        data = saved.get('data', {})
        res = saved.get('res', {})
        if saved:
            last_results[key]['nb_downloads'] += 1
            if last_results[key]['nb_downloads'] >= max_nb_downloads:
                del last_results[key]
    return data, res


def clean_old_results():
    """
    Nettoyage des résultats stockés depuis trop longtemps. Paramêtré par config['app']['clean_results_in_min']

    :return: None
    """
    global last_results, last_results_lock, last_result_lifetime
    logger.info('starting to clean')
    now = datetime.now()
    keys_to_clean = []
    with last_results_lock:
        for k, v in last_results.items():
            if v['datetime'] < now - last_result_lifetime:
                keys_to_clean.append(k)
        logger.info(f'cleaning {keys_to_clean}')
        for k in keys_to_clean:
            del last_results[k]


#####################
#  Gestion des inputs
#####################

class NoFileException(Exception):
    pass


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_request_files(request):
    """
    Récupération des données dans une request

    :param request: la requète entrante
    :return: list de paths de fichiers récupérés
    """
    if not len(request.files):
        raise NoFileException("No file sent")
    files = []
    for _, descriptor in request.files.items():
        # if user does not select file, browser also
        # submit an empty part without filename
        if descriptor.filename == '':
            raise NoFileException("No file selected")
        if descriptor and allowed_file(descriptor.filename):
            filename = secure_filename(descriptor.filename)
            descriptor.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            files.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return files


##########
# Handle errors
##########

def handle_bad_request(e):
    logger.debug(str(e))
    return f'bad request : {e}', 400


app.register_error_handler(400, handle_bad_request)


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        # "description": e.description,
    })
    response.content_type = "application/json"
    logger.error(str(e))
    return response


@app.errorhandler(404)
def resource_not_found(e):
    logger.debug(str(e))
    return jsonify({'error': str(e)}), 404


########
# request processing
#######

def process_data(ressource_desc, func, *args):
    """
    Fonction d'encapsulation pour appliquer func à args en gérant les erreurs

    :param ressource_desc: str (utilisé dans le texte de l'erreur)
    :param func: fonction
    :param args: arguments pour la fonction
    :return: réponse à la requète
    """
    try:
        res = func(*args)
        if isinstance(res, str):
            # cas particulier de la pca TopicModel qu'on affiche en html
            return res
        return jsonify(res)
    except ValueError as e:
        logger.debug(e)
        abort(404, description=f"Resource not found : {ressource_desc}")
    except Exception as e:
        logger.error(e)
        abort(500, description=f"Unknow error : {e}")


def get_dataframe_from_filename(filename):
    """
    Extraction des données depuis les fichiers inputs

    :param filename: fichier à lire
    :return: dataframe
    """
    with open(filename, 'rb') as f:
        is_pdf = ('PDF' in m.from_buffer(f.read(1024)))
    if is_pdf:
        with open(filename, 'rb') as f:
            df_data = input_handling.from_pdf_to_mrv_format(f, colonnes, mapping)
    elif '.xml' in filename:
        df_data = input_handling.from_xml_to_mrv_format(filename, colonnes, mapping)
    elif '.csv' in filename:
        df_data = input_handling.from_csv_to_mrv_format(filename, colonnes, mapping)
    else:
        raise RuntimeError('Bad input format')
    return df_data


# déclaration des fonctions de formattage des outputs de prédiction
formatting_functions = {
    'default': antpro_formatting.format_prediction
}


def apply_prediction_formatting(results):
    """
    Formattage des résultats de prédiction en réponse de requètes

    :param results: dict {f: [{model_name: mn, predictions: dataframe}]}
    :return: dict
    """
    global formatting_functions
    res = copy.deepcopy(results)
    for f in res:
        if not isinstance(res[f], list):
            continue
        for i in range(len(res[f])):
            formatting_function = formatting_functions['default']
            if res[f][i]['model_name'] in formatting_functions:
                formatting_function = formatting_functions[res[f][i]['model_name']]
            res[f][i]['predictions'] = formatting_function(res[f][i]['model_name'],
                                                           res[f][i]['predictions'])
    return res


def handle_prediction_request(request, prediction_func, model_name, loaded_data=None):
    """
    Traitement complet d'une requète de classification mono-modèle

    :param request: requète entrante
    :param prediction_func: fonction de prédiction
    :param model_name: nom du modèle (ou liste de noms)
    :param loaded_data: data extraite du fichier input si dicponible (sinon, extrait en début de fonction)
                        {f: dataframe}
    :return: loaded_data: {f: dataframe}, dict {f: [{model_name: mn, predictions: dataframe}]}
    """
    results = {}
    if loaded_data is None:
        # extraction des fichiers de la requète
        try:
            uploaded_files = get_request_files(request)
        except NoFileException as e:
            return '', handle_bad_request(e)
        except Exception as e:
            return '', handle_exception(HTTPException(str(e)))
        # extraction des données des fichiers
        loaded_data = {}
        for filename in uploaded_files:
            try:
                df_data = get_dataframe_from_filename(filename)
                loaded_data[filename] = df_data
            except Exception as e:
                return '', handle_bad_request(e)
    for filename, df_data in loaded_data.items():
        # application du modèle
        res_df = prediction_func(df_data)
        if not isinstance(model_name, list):
            model_name = [model_name]
            res_df = [res_df]
        file_key = os.path.basename(filename)
        results[file_key] = []
        for i, mn in enumerate(model_name):
            file_res = {
                'model_name': mn,
                'predictions': res_df[i]
            }
            results[file_key].append(file_res)
    # nettoyage des fichiers
    for filename in loaded_data:
        try:
            os.remove(filename)
        except OSError:
            pass

    return loaded_data, results


##############
# Load models
#############

logger.info('Loading Prediction models')
DCO_model, Dysfonctionnement_model, \
    consequence_model, Effet_model, \
    gravite_1234_model, gravite_01_model, classification_encoder = load_classification_models(config['models']['path'])

logger.info('Loading topics and clusters data')
if ACTIVATE_FRONT:
    ClusteringModels(with_preprocess=True)
else:
    ClusteringModels(with_preprocess=False)


def init_models():
    init_file = config['models']['test_file']
    if not os.path.exists(init_file):
        return
    logger.info('Initializing models...')
    df_data = get_dataframe_from_filename(init_file)
    results = []
    results.append({'model_name': 'DCO', 'predictions': predict_DCO(DCO_model, df_data)})
    results.append({'model_name': 'dysfonctionnement', 'predictions': predict_dysfonctionnements(Dysfonctionnement_model, classification_encoder, df_data)})
    results.append({'model_name': 'consequence', 'predictions': predict_consequences(consequence_model, classification_encoder, df_data)})
    results.append({'model_name': 'effet', 'predictions': predict_effets(Effet_model, classification_encoder, df_data)})
    res_df = predict_gravites(gravite_1234_model, gravite_01_model, df_data)
    topics, cluster = clusterize_doc(df_data, results)
    logger.info('... initialized')


init_models()

#############
# declare security routes
#############

password_file = config['app']['password_file']
try:
    with open(password_file) as f:
        passwords = json.load(f)
except FileNotFoundError:
    passwords = {}


def add_password(user, hashed_pwd):
    global passwords
    passwords[user] = hashed_pwd
    with open(password_file, "w") as f:
        json.dump(passwords, f)


@app.route('/dgs-api/users', methods=['POST'])
@cross_origin()
@auth.login_required
def new_user():
    global passwords
    username = request.json.get('username')
    password = request.json.get('password')
    if username is None or password is None:
        return handle_bad_request(RuntimeError('Missing username or password')) # missing arguments
    if request.authorization['username'] != 'dgs_admin':
        return handle_bad_request(RuntimeError('Only the admin account can create new users'))  # missing arguments
    user = User(username, password)
    user.hash_password(password)
    add_password(user.username, user.password_hash)
    return jsonify({ 'username': user.username }), 201


@auth.verify_password
def verify_password(username, password):
    global passwords
    if username not in passwords:
        return False
    user = User(username, passwords[username])
    if not user.verify_password(password):
        return False
    return True


################
# Declare data routes
################

@app.route('/dgs-api/predict/dco', methods=['POST'])
@cross_origin()
@auth.login_required
def serve_predict_dco():
    if request.method == 'POST':
        logger.debug('/predict/dco - POST')
        data, res = handle_prediction_request(request, lambda x: predict_DCO(DCO_model, x), 'DCO')
        if isinstance(data, str):
            # error
            return res
        key = add_last_results(data, res)
        res['last_results_key'] = key
        return apply_prediction_formatting(res)


@app.route('/dgs-api/predict/dysfonctionnement', methods=['POST'])
@cross_origin()
@auth.login_required
def serve_predict_dysfonctionnement():
    if request.method == 'POST':
        logger.debug('/predict/dysfonctionnement - POST')
        data, res = handle_prediction_request(request,
                                         lambda x: predict_dysfonctionnements(Dysfonctionnement_model, classification_encoder, x),
                                         'dysfonctionnement')
        if isinstance(data, str):
            # error
            return res
        key = add_last_results(data, res)
        res['last_results_key'] = key
        return apply_prediction_formatting(res)


@app.route('/dgs-api/predict/consequence', methods=['POST'])
@cross_origin()
@auth.login_required
def serve_predict_consequence():
    if request.method == 'POST':
        logger.debug('/predict/consequence - POST')
        data, res = handle_prediction_request(request,
                                         lambda x: predict_consequences(consequence_model, classification_encoder, x),
                                         'consequence')
        if isinstance(data, str):
            # error
            return res
        key = add_last_results(data, res)
        res['last_results_key'] = key
        return apply_prediction_formatting(res)


@app.route('/dgs-api/predict/effet', methods=['POST'])
@cross_origin()
@auth.login_required
def serve_predict_effet():
    if request.method == 'POST':
        logger.debug('/predict/effet - POST')
        data, res = handle_prediction_request(request,
                                         lambda x: predict_effets(Effet_model, classification_encoder, x), 'effet')
        if isinstance(data, str):
            # error
            return res
        key = add_last_results(data, res)
        res['last_results_key'] = key
        return apply_prediction_formatting(res)


@app.route('/dgs-api/predict/gravite', methods=['POST'])
@cross_origin()
@auth.login_required
def serve_predict_gravite():
    if request.method == 'POST':
        logger.debug('/predict/gravite - POST')
        data, res = handle_prediction_request(request,
                                         lambda x: predict_gravites(gravite_1234_model, gravite_01_model, x),
                                         ['gravité_ordinale', 'gravité_binaire'])
        if isinstance(data, str):
            # error
            return res
        key = add_last_results(data, res)
        res['last_results_key'] = key
        return apply_prediction_formatting(res)


@app.route('/dgs-api/predict/clustering', methods=['POST'])
@app.route('/dgs-api/predict/all_models', methods=['POST'])
@cross_origin()
@auth.login_required
def serve_predict_all():
    if request.method == 'POST':
        logger.debug('/predict/all_models - POST')
        data, res0 = handle_prediction_request(request, lambda x: predict_DCO(DCO_model, x), 'DCO')
        if isinstance(data, str):
            # error
            return res0
        _, res1 = handle_prediction_request(request,
                                         lambda x: predict_dysfonctionnements(Dysfonctionnement_model,
                                                                             classification_encoder,
                                                                             x),
                                         'dysfonctionnement', data)
        _, res2 = handle_prediction_request(request,
                                         lambda x: predict_consequences(consequence_model, classification_encoder, x),
                                         'consequence', data)
        _, res3 = handle_prediction_request(request,
                                         lambda x: predict_effets(Effet_model, classification_encoder, x),
                                         'effet', data)
        _, res4 = handle_prediction_request(request,
                                         lambda x: predict_gravites(gravite_1234_model, gravite_01_model, x),
                                         ['gravité_ordinale', 'gravité_binaire'], data)

        for k in res0.keys():
            for r in [res1, res2, res3, res4]:
                res0[k].extend(r[k])

        for f in data:
            f_data = data[f]
            f_results = res0[os.path.basename(f)]
            topics, cluster = clusterize_doc(f_data, f_results)
            res0[os.path.basename(f)].append({'model_name': 'topics', 'predictions': topics})
            res0[os.path.basename(f)].append({'model_name': 'cluster', 'predictions': cluster})

        key = add_last_results(data, res0)
        res0['last_results_key'] = key
        formatted = apply_prediction_formatting(res0)
        if ACTIVATE_FRONT:
            for f in formatted:
                if f == 'last_results_key':
                    continue
                topics_ind = [v['model_name'] for v in formatted[f]].index('topics')
                for i in range(len(formatted[f][topics_ind]['predictions']['datasource'])):
                    topic_num = int(formatted[f][topics_ind]['predictions']['datasource'][i]['cat'].split(' ')[-1])
                    formatted[f][topics_ind]['predictions']['datasource'][i]['tooltip'] = ", ".join(clusters.get_topic_topwords(topic_num))
        return formatted


@app.route('/dgs-api/predict/last_results/<output_format>', methods=['POST'])
@cross_origin()
@auth.login_required
def serve_last_results_as_file(output_format):
    if output_format not in ['csv', 'excel']:
        return handle_bad_request(ValueError('format must be in [csv, excel]'))
    res = {}
    data = {}
    for key in json.loads(request.data)['last_results_keys']:
        key_data, key_res = get_last_results(key)
        if key_res:
            data.update(key_data)
            res.update(key_res)
    if not len(data):
        return resource_not_found(ValueError('Pas de données disponible : les données sont expirées ou déjà téléchargées'))

    saved_file = results_to_file(data, res, output_format)

    @after_this_request
    def remove_file(response):
        try:
            os.remove(saved_file)
        except Exception as error:
            logger.error("Error removing or closing downloaded file handle", error)
        return response
    try:
        return send_file(saved_file, as_attachment=True, attachment_filename=os.path.basename(saved_file))
    except Exception as e:
        return handle_bad_request(e)


@app.route('/dgs-api/predict/performances', methods=['GET'])
@cross_origin()
@auth.login_required
def get_models_perfs():
    logger.debug(f'/predic/performances - GET')
    return process_data('predict/model_perfs',
                        lambda x: antpro_formatting.format_model_performances(get_models_performances(config['models']['path'])),
                        None)


if ACTIVATE_FRONT:
    ###########################
    # Front routes on Documents
    ###########################
    @app.route('/dgs-api/documents/all_ids', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_all_doc_ids():
        logger.debug(f'/documents/all_ids')
        return process_data('document',
                            lambda x: antpro_formatting.format_doc_ids(clusters.get_all_docs_ids()),
                            None)


    @app.route('/dgs-api/documents/<docid_or_docname>/content', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_document_content(docid_or_docname):
        logger.debug(f'/documents/<docid_or_docname>/content - GET - docid_or_docname={docid_or_docname}')
        return process_data('document',
                            lambda x: antpro_formatting.format_doc_content(x, clusters.get_document_info(x)),
                            docid_or_docname)


    @app.route('/dgs-api/documents/<docid_or_docname>/topics', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_document_topics(docid_or_docname):
        logger.debug(f'/documents/<docid_or_docname>/topics - GET - docid_or_docname={docid_or_docname}')
        return process_data('document',
                            lambda x: antpro_formatting.format_document_topics(x, clusters.get_document_topics(x)),
                            docid_or_docname)


    @app.route('/dgs-api/documents/<docid_or_docname>/cluster', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_document_cluster(docid_or_docname):
        logger.debug(f'/documents/<docid_or_docname>/cluster - GET - docid_or_docname={docid_or_docname}')
        return process_data('document',
                            lambda x: antpro_formatting.format_document_cluster(x, clusters.get_document_clusterind(x)),
                            docid_or_docname)


    @app.route('/dgs-api/documents/<docid_or_docname>', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_document_complete(docid_or_docname):
        logger.debug(f'/documents/<docid_or_docname> - GET - docid_or_docname={docid_or_docname}')
        return process_data('document',
                            lambda x: antpro_formatting.format_document_complete(x, clusters.get_document_complete(x)),
                            docid_or_docname)

    ###########################
    # Front routes on Topic model
    ###########################

    @app.route('/dgs-api/topics/model/nb_topics', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topicmodel_nbtopics():
        logger.debug(f'/topics/model/nb_topics')
        return process_data('topicmodel',
                            lambda x: antpro_formatting.format_topicmodel_nbtopics(clusters.get_topicmodel_nbtopics()),
                            None)


    @app.route('/dgs-api/topics/model/coherence', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topicmodel_coherence():
        logger.debug(f'/topics/model/coherence')
        return process_data('topicmodel',
                            lambda x: antpro_formatting.format_topicmodel_score(clusters.get_topicmodel_coherence_score()),
                            None)


    @app.route('/dgs-api/topics/model/distances', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topicmodel_distances():
        logger.debug(f'/topics/model/distances')
        return process_data('topicmodel',
                            lambda x: antpro_formatting.format_distances_mat(clusters.get_topicmodel_distance_mat()),
                            None)


    @app.route('/dgs-api/topics/model/pca', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topicmodel_lda():
        logger.debug(f'/topics/model/pca')
        res = process_data('topicmodel',
                            lambda x: antpro_formatting.format_topicmodel_pca(clusters.get_topicmodel_pca()),
                            None)
        return Response(response=res, status=200) # , headers={'X-Frame-Options': 'SAMEORIGIN'})


    @app.route('/dgs-api/topics/model', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topicmodel_complete():
        logger.debug(f'/topics/model')
        return process_data('topicmodel',
                            lambda x: antpro_formatting.format_topicmodel_complete(clusters.get_topicmodel_complete()),
                            None)

    ###########################
    # Front routes on Topics
    ###########################


    @app.route('/dgs-api/topics/<topic_ind>/weight', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topic_weight(topic_ind):
        logger.debug(f'/topics/<topic_ind>/weight - GET - topic_ind={topic_ind}')
        return process_data('topic',
                            lambda x: antpro_formatting.format_topic_weight(x, clusters.get_topic_weight(int(x))),
                            topic_ind)


    @app.route('/dgs-api/topics/<topic_ind>/documents', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topic_documents(topic_ind):
        logger.debug(f'/topics/<topic_ind>/documents - GET - topic_ind={topic_ind}')
        return process_data('topic',
                            lambda x: antpro_formatting.format_topic_documents(x, clusters.get_topic_documents(int(x))),
                            topic_ind)


    @app.route('/dgs-api/topics/<topic_ind>/topwords', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topic_topwords(topic_ind):
        logger.debug(f'/topics/<topic_ind>/topwords - GET - topic_ind={topic_ind}')
        return process_data('topic',
                            lambda x: antpro_formatting.format_topic_topwords(x, clusters.get_topic_topwords(int(x))),
                            topic_ind)


    @app.route('/dgs-api/topics/<topic_ind>/wordcloud', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topic_wordcloud(topic_ind):
        logger.debug(f'/topics/<topic_ind>/wordcloud - GET - topic_ind={topic_ind}')
        return process_data('topic',
                            lambda x: antpro_formatting.format_topic_wordcloud(x, clusters.get_topic_wordcloud(int(x))),
                            topic_ind)


    @app.route('/dgs-api/topics/<topic_ind>', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_topic_complete(topic_ind):
        logger.debug(f'/topics/<topic_ind> - GET - topic_ind={topic_ind}')
        return process_data('topic',
                            lambda x: antpro_formatting.format_topic_complete(x, clusters.get_topic_complete(int(x))),
                            topic_ind)

    ###########################
    # Front routes on Cluster model
    ###########################

    @app.route('/dgs-api/clusters/model/nb_clusters', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_clustermodel_nbclusters():
        logger.debug(f'/clusters/model/nb_clusters')
        return process_data('clustermodel',
                            lambda x: antpro_formatting.format_clustermodel_nbclusters(clusters.get_clustermodel_nbclusters()),
                            None)


    @app.route('/dgs-api/clusters/model/scores', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_clustermodel_coherence():
        logger.debug(f'/clusters/model/scores')
        return process_data('clustermodel', lambda x: antpro_formatting.format_clustermodel_scores(clusters.get_clustermodel_scores()), None)


    @app.route('/dgs-api/clusters/model/distances', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_clustermodel_distances():
        logger.debug(f'/clusters/model/distances')
        return process_data('clustermodel',
                            lambda x: antpro_formatting.format_distances_mat(clusters.get_clustermodel_distance_mat()),
                            None)


    @app.route('/dgs-api/clusters/model/pca', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_clustermodel_lda():
        logger.debug(f'/clusters/model/pca')
        return process_data('clustermodel',
                            lambda x: antpro_formatting.format_clustermodel_pca(clusters.get_clustermodel_pca()),
                            None)


    @app.route('/dgs-api/clusters/model', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_clustermodel_complete():
        logger.debug(f'/clusters/model')
        return process_data('clustermodel',
                            lambda x: antpro_formatting.format_clustermodel_complete(clusters.get_clustermodel_complete()),
                            None)

    ###########################
    # Front routes on Clusters
    ###########################


    @app.route('/dgs-api/clusters/<cluster_ind>/weight', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_cluster_weight(cluster_ind):
        logger.debug(f'/clusters/<cluster_ind>/weight - GET - cluster_ind={cluster_ind}')
        return process_data('cluster',
                            lambda x: antpro_formatting.format_cluster_weight(x, clusters.get_cluster_weight(int(x))),
                            cluster_ind)


    @app.route('/dgs-api/clusters/<cluster_ind>/documents', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_cluster_documents(cluster_ind):
        logger.debug(f'/clusters/<cluster_ind>/documents - GET - cluster_ind={cluster_ind}')
        return process_data('cluster',
                            lambda x: antpro_formatting.format_cluster_documents(x, clusters.get_cluster_documents(int(x))),
                            cluster_ind)


    @app.route('/dgs-api/clusters/<cluster_ind>/topics', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_cluster_topics(cluster_ind):
        logger.debug(f'/clusters/<cluster_ind>/topics - GET - cluster_ind={cluster_ind}')
        return process_data('cluster',
                            lambda x: antpro_formatting.format_cluster_topics(x, clusters.get_cluster_topics(int(x))),
                            cluster_ind)


    @app.route('/dgs-api/clusters/<cluster_ind>/dcos', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_cluster_dcos(cluster_ind):
        logger.debug(f'/clusters/<cluster_ind>/dcos - GET - cluster_ind={cluster_ind}')
        return process_data('cluster',
                            lambda x: antpro_formatting.format_cluster_dcos(x, clusters.get_cluster_dcos(int(x))),
                            cluster_ind)


    @app.route('/dgs-api/clusters/<cluster_ind>/wordcloud', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_cluster_wordcloud(cluster_ind):
        logger.debug(f'/clusters/<cluster_ind>/wordcloud - GET - cluster_ind={cluster_ind}')
        return process_data('cluster',
                            lambda x: antpro_formatting.format_cluster_wordcloud(x, clusters.get_cluster_wordcloud(int(x))),
                            cluster_ind)


    @app.route('/dgs-api/clusters/<cluster_ind>', methods=['GET'])
    @cross_origin()
    @auth.login_required
    def get_cluster_complete(cluster_ind):
        logger.debug(f'/clusters/<cluster_ind> - GET - cluster_ind={cluster_ind}')
        return process_data('cluster',
                            lambda x: antpro_formatting.format_cluster_complete(x, clusters.get_cluster_complete(int(x))),
                            cluster_ind)


if __name__ == "__main__":
    # lancement du nettoyage périodique des données
    scheduler = BackgroundScheduler()
    job = scheduler.add_job(clean_old_results, 'interval', minutes=last_results_cleaning_period_in_min)
    scheduler.start()

    # démarrage de l'api
    logger.info('Starting API')
    cors = CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.run(host='0.0.0.0')

