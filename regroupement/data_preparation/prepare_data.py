"""
Auteurs:
    - Robin Quillivic, rquillivic@starclay.fr
    - Cyril Poulet, cpoulet@starclay.fr
Description:
    Permet nettoyer et de post-traiter les données textuelles
    en se basant sur les bibliothèques : Spacy, nltk, multi-rake et gensim
    
"""
import sys
import os
import yaml

import pandas as pd


import spacy
import string
try:
    import nltk
except:
    pass
from spacy.lang.fr.stop_words import STOP_WORDS

# gensim
import gensim
import re
import multiprocessing
import sys
import logging

sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))
import clean_text
from gensim.utils import deaccent
import ast
from pandas.core.common import flatten

#'fr_core_news_md', 'fr_news' diponible également

COMPLETE_STOP_WORDS = set(STOP_WORDS).union(set(clean_text.STOP_WORDS))
COMPLETE_STOP_WORDS = set([deaccent(k.lower()) for k in COMPLETE_STOP_WORDS])


class SpacyLoader:
    nlp = None

    def __init__(self):
        if SpacyLoader.nlp is None:
            logging.getLogger('data_preparation').info('Loading spacy french model')
            SpacyLoader.nlp = spacy.load('fr_core_news_md', disable=['parser', 'ner'])


def post_process_spacy_doc(doc):
    if len(doc) >= 1:

        # remove words too short
        tokens = [token for token in doc if len(token.text.lower()) > 1]
        # remove stop words
        tokens = [token for token in tokens if deaccent(token.text.lower()) not in COMPLETE_STOP_WORDS]
        tokens = [token for token in tokens if deaccent(token.lemma_.lower()) not in COMPLETE_STOP_WORDS]
        # remove numbers
        tokens = [token for token in tokens if not token.text.lower().isnumeric()]
        # remove punctuation
        tokens = [token for token in tokens if not all(c in set(string.punctuation) for c in token.text.lower())]
        # get lemmas and deaccentuate
        lemma = [deaccent(token.lemma_.lower()) for token in tokens]
        # split on special caractère
        lemma = [re.split('[^A-Za-z]+', str(lem)) for lem in lemma]
        # flatten the list
        lemma= list(flatten(lemma))
        # remove all specific caractere
        lemma = [re.sub('[^A-Za-z]+', '', str(lem)) for lem in lemma]
        #remove all token with less than 2 letters
        lemma = [lem for lem in lemma if len(lem)>2]
        return lemma
    else:
        return []


def nltk_tokenisation(text, with_stop_words=True):
    """
    Transforme le texte en liste de tokens, en minuscule, en ayant supprimé la ponctuation et les mots frequents.
    Exemple : Entrée = "je suis heureux aujourd'hui"; Sortie : ['je', 'suis', 'heureux', "aujourd'hui"]
    Args: 
        x (str): le texte
    Returns:
    - tokens (list(str)): liste de tokens


    """
    txt = text.lower()
    if not with_stop_words:
        tokens = nltk.word_tokenize(
            txt, language='french', preserve_line=False)
    else:
        words = nltk.word_tokenize(txt, language='french', preserve_line=False)
        tokens = [word for word in words if word not in set(
            STOP_WORDS) | set(string.punctuation)]
    return tokens


def spacy_extract_pos(text, f='nltk'):
    """
    Fonction qui prend en entrée du texte et qui renvoie au format l'étiquetage morpho-synthaxique du document sous forme de list(tuple) ou bien d'une dataframe selon la valeur de f.
    Args :
        text (str) :  le texte 
        f (str): le format peut prendre 'nltk' ou 'pd'
    Returns :
        df (Dataframe) : dataframe  avec les tags
        nltk-format :list(tuple)
    """
    doc = SpacyLoader().nlp(text)
    parsed_text = {'word': [], 'upos': []}
    nltk_format = []
    for wrd in doc:
        parsed_text['word'].append(wrd.text)
        parsed_text['upos'].append(wrd.pos_)
        nltk_format.append((wrd.text, wrd.pos_))
    # return a dataframe of pos and text
    df = pd.DataFrame(parsed_text)
    if f == 'nltk':
        return(nltk_format)
    else:
        return(df)


def nltk_keyWord(text):
    """
    Fonction qui prend le texte en entrée et renvoie en sortie une liste de mot identifés comme important.
    Les mots importants sont identifés utilisant l'étiquetage morpho-syntaxique de spacy et une expression régulière sur ces étiquettes.
    Args: 
        text (str):  le texte 

    Returns:
        keywords (list(str()) : Les mots clefs
    """
    tagged = spacy_extract_pos(text, f='nltk')
    keywords = set()
    chunkGram = r"""NE:  {<PROPN>+<PROPN>?|<PROPN|NOUN>+<CC.*|NOUN.*>+<PROPN>}
                {<PROPN>}"""
    chunkParser = nltk.RegexpParser(chunkGram)
    chunked = chunkParser.parse(tagged)
    NE = [" ".join(w for w, t in ele) for ele in chunked if
          isinstance(ele, nltk.Tree)]
    for i in NE:
        keywords.add(i)
    return(keywords)


def rake_key_phrases_extract(text):
    """
    Utilisant le moteur Rake, cette fonction retourne une liste des phrases
    considérées comme importante.
    Args: 
        text (str): le texte 
    Returns :
        k (list): liste de phrase.
    """
    from multi_rake import Rake
    r = Rake(language_code='fr', stopwords=STOP_WORDS | set(string.punctuation))
    k = r.apply(text)
    K = [l[0] for l in k]
    return K


def post_process_rake_extract(spacy_doc, extracted_phrases):
    def _ppt(str_val):
        return deaccent(str_val.lower())

    prep_sp_text = [_ppt(t.text) for t in spacy_doc]
    res = []
    for ep in extracted_phrases:
        prep_ep = _ppt(ep)
        try:
            new_start_token = 0
            new_end_token_ind = 0
            for i in range(len(spacy_doc)):
                if " ".join(prep_sp_text[i:]).index(prep_ep) == 0:
                    new_start_token = i
                    break
            for j in range(new_start_token+1, len(spacy_doc)):
                if prep_ep in " ".join(prep_sp_text[new_start_token:j]):
                    new_end_token_ind = j
                    break
            if new_end_token_ind <= new_start_token:
                continue
            tokens = spacy_doc[new_start_token:new_end_token_ind]
            lemmas = post_process_spacy_doc(tokens)
            if lemmas is not None and len(lemmas):
                res.append(" ".join(post_process_spacy_doc(tokens)))
        except ValueError:
            # not found
            continue   # res.append(_ppt(ep))
    return res


def prepare_data(mrv, med_term=False,medical_terms=None, save_dir=None, use_multiprocessing=True):
    """Fonction pour préparer les données avant l'entrainement

    Args:
        mrv (pd.DataFrame): Base de données MRveille
        medical_terms (pd.DataFrame): Termes médicaux extraits
        save (bool, optional): Sauvegarde ou non le fichier. Defaults to False.

    Returns:
        df (pd.DataFrame): Données nettoyées
    """
    def _save():
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            df.to_pickle(os.path.join(save_dir, 'cleaned_data_5.pkl'))

    # Sélection des colonnes
    print('clean docs')
    df = mrv[['NUMERO_DECLARATION', 'DESCRIPTION_INCIDENT',
              'ETAT_PATIENT', 'FABRICANT', 'DCO_ID']]
    # gestion des Nan
    df[['DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'FABRICANT']] = \
        df[['DESCRIPTION_INCIDENT', 'ETAT_PATIENT', 'FABRICANT']].fillna(' ')\
            .replace(r'\r', ' ', regex=True).replace(r'\n', ' ', regex=True)\
            .replace('([0-9]+)([^\d])', '\\1 \\2', regex=True) \
            .replace('([\\\/])(\w)', '\\1 \\2', regex=True) \
            .replace(' [ ]+', ' ', regex=True)
    # Construction de la variable textes
    df['text'] = (df['DESCRIPTION_INCIDENT'] + '. ' + df['ETAT_PATIENT']).str.strip()
    # # Nettoyage des données textuelles
    # df['text'] = [clean_text.preprocess_text(str(deaccent(x))) for x in df['text']]
    df['text'] = df['text'].map(lambda x: clean_text.replace_typical_misspell(x))

    # Construction des lemmes et supressions des stop word
    # df['text_tok'] = df['text'].map(lambda x:  nltk_tokenisation(str(x)))   -> not used

    print('lemmatize docs')
    spacy_docs = [SpacyLoader().nlp(doc) for doc in df['text'].values]
    if use_multiprocessing:
        p1 = multiprocessing.Pool()
        df['text_lem'] = p1.map(post_process_spacy_doc, spacy_docs)   #  -> lemmatisé, désaccentué, lower
    else:
        df['text_lem'] = [post_process_spacy_doc(d) for d in spacy_docs]   #  -> lemmatisé, désaccentué, lower
    _save()

    #print('extract rake keywords')
    #rake_extracts = p1.map(rake_key_phrases_extract, df['text'].values)
    #df['rake_kw'] = p1.starmap(post_process_rake_extract, zip(spacy_docs, rake_extracts)) #  -> lemmatisé, désaccentué, lower
    _save()

    # 
    print('extract bigrams and trigrams')
    Docs = df['text_lem'].tolist()
    bigram = gensim.models.Phrases(Docs, min_count=1, threshold=2)
    trigram = gensim.models.Phrases(bigram[Docs], threshold=2)

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    bigrams = [bigram_mod[line] for line in Docs]
    df['bigram'] = bigrams

    trigrams = [trigram_mod[bigram_mod[line]] for line in Docs]
    df['trigram'] = trigrams
    _save()

    # récupération des termes médicaux + post-process de ces termes + on ne garde qu'un exemplaire de chaque terme par doc
    if med_term :
        print('extract medical terms')
        med_term = medical_terms.result.map(
            lambda x: [elt[0] for elt in ast.literal_eval(x)])

        if use_multiprocessing:
            df['med_term'] = p1.starmap(post_process_rake_extract, zip(spacy_docs, med_term.values))
        else:
            df['med_term'] = [post_process_rake_extract(spdoc, mdterms) for spdoc, mdterms in zip(spacy_docs, med_term.values)]
        df['med_term_uniq'] = [list(set(v)) for v in df['med_term'].values]
        _save()

    return df

def clean_fabricant(text):
    STOP_WORDS = ["france","sarl",'sas','ltd','inc','sro','sa', 'ab']
    #lower
    text = text.lower()
    # Stop words
    tokens = text.split(" ") 
    tokens = [token for token in tokens if deaccent(token.lower()) not in STOP_WORDS]
    # ponctuation
    tokens = [token for token in tokens if not all(c in set(string.punctuation) for c in token.lower())]

    # Acccent
    tokens = [deaccent(token.lower()) for token in tokens]
    # Caratère spéciaux
    tokens = [re.split('[^A-Za-z]+', str(token)) for token in tokens]
    # flatten the list
    tokens = list(flatten(tokens))
    #
    tokens = [re.sub('[^A-Za-z]+', '', str(token)) for token in tokens]

    # Mots de 2 caratères
    tokens = [token for token in tokens if len(token.lower()) > 2]

    return " ".join(tokens)





if __name__ == "__main__":
    import time

    path_to_conf_file = os.path.dirname((os.path.dirname(__file__)))
    config_file = 'config.yaml'

    if len(sys.argv) == 2:
        config_file = sys.argv[1]

    config_file = os.path.join(path_to_conf_file, config_file)
        
    if not os.path.exists(config_file):
        print(f'File {config_file} does not exists')
        exit(-1)
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    save_dir = config['save_folder']['path']
    if not os.path.isabs(save_dir):
        save_dir = os.path.abspath(os.path.join(path_to_conf_file, save_dir))
    data_file = config['data']['mrv']['path']
    if not os.path.isabs(data_file):
        data_file = os.path.abspath(os.path.join(path_to_conf_file, data_file))

    # Quelques fonction classiques et utiles pour le traitement des données textuelles
    mrv = pd.read_csv(data_file)

    medical_terms = pd.read_csv(os.path.join(save_dir, 'umls.csv'))
    med_term = medical_terms.result.map(lambda x: [elt[0] for elt in ast.literal_eval(x)])
    # df = pd.read_pickle(os.path.join(save_dir, "cleaned_data.pkl"))
    # post_process_rake_extract(SpacyLoader().nlp(df['text'].values[-3]), med_term.values[-3])

    start = time.time()
    prepare_data(mrv, medical_terms, save_dir=save_dir)
    print(f'temps passé : {time.time() - start}')
