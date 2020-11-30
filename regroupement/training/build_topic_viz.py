"""
Auteur :
    - Robin Quillivic, Data scientist chez Starclay, rquillivic@starclay.fr
Description :
    fochier permettant de construire la visualisation d'un topic model
"""

import pandas as pd
import numpy as np
import os
import yaml
import train_topic
import sys

import gensim
import pyLDAvis
import pyLDAvis.gensim

from gensim.corpora import Dictionary, MmCorpus

try_name = '09_10_2020_hdp'#'LDA_trigrams_medterms_150_40'
path_to_regroupement =  os.path.dirname(os.path.dirname(__file__))

sys.path.insert(1, path_to_regroupement)
from utils import loading_function


with open(os.path.join(path_to_regroupement,'training','training_config.yaml'), 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        
folder = config['path_to_save']

def load_topic_with_no_viz(path,try_name):
    """Charge un topic model sans visualisation

    Args:
        path (str): chemin vers le topic model
        try_name (str): nom du modèle associé
    """
    
    filename = try_name 
    save_dir = os.path.join(path,'LDA')   
    # Dict
    path_dict = os.path.join(save_dir, filename+'.dict')
    dictionary = Dictionary.load(path_dict)
    
    #corpus                                    
    path_corpus = os.path.join(save_dir, filename+'.mm')
    corpus =  MmCorpus(path_corpus)
    
    # Model
    path_model = os.path.join(save_dir, filename+'.model')
    model = gensim.models.LdaModel.load(path_model)
    
    return( dictionary,corpus,model)


def build_viz(dictionary,corpus,model,save=True):
    """fonction permettant de créer la visualisation d'un topic à partir d'un dictionnaire, d'un corpus et d'un modèle

    Args:
        dictionary (gensim.corpora.Dictionnary): dictionnaire du corpus
        corpus (gensim.corpora.Corpus): Corpus du modèle
        model (gensim.model): modèle de thème
        save (bool, optional): faut-il sauver des la visualisation ?. Defaults to True.

    Returns:
        data (dict) : données pour la construction de la visualisation
    """
    
    data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    if save :
        save_dir = os.path.join(folder,try_name,'LDA')
        file_name = os.path.join(save_dir, try_name+'.json')
        pyLDAvis.save_json(data, file_name)
    return data

def build_doc_topic(corpus,model,data):
    """Permet de construire la matrice thème document associé aux modèles

    Args:
        corpus (gensim.corpora.Corpus): Corpus du modèle
        model (gensim.model): modèle de thème
        data (dict): base de donnée Mrveille
    """
    doc_lda = model.get_document_topics(corpus, minimum_probability=-1)
    topic_names = ['Topic'+str(i) for i in range(model.num_topics)]
    mat = np.array([np.array([tup[1] for tup in lst]) for lst in doc_lda])

    # Construction de la matrice
    df_doc_topic = pd.DataFrame(
    mat, columns=topic_names, index=data.index)
    df_doc_topic['DCO_ID'] = data['DCO_ID']
    df_doc_topic['text'] = data['text']
    df_doc_topic['text_lem'] = data['text_lem']
    df_doc_topic['NUMERO_DECLARATION'] = data['NUMERO_DECLARATION']

    # sauvegarde
    save_dir = os.path.join(folder,try_name,'LDA')
    file_name = os.path.join(save_dir, try_name+'.pkl')  
    df_doc_topic.to_pickle(file_name)

if __name__== "__main__":
    path  = os.path.join(folder,try_name)
    dic,corpus,model = load_topic_with_no_viz(path,try_name)
    model = model.suggested_lda_model()
    filename = os.path.join(config['data']['path'], config['data']['filename'])
    data = pd.read_pickle(filename)
    #mat = build_doc_topic(corpus,model,data)
    viz = build_viz(dic,corpus,model,save=True)
        