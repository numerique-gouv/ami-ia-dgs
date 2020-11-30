import os
import pandas as pd
import re


def results_to_file(loaded_files, results, output_format='csv'):
    """
    Fonction qui permet de reformater les résultats de prédiction sous forme csv ou excel,
    et de le sauvegarder comme fichier

    :param loaded_files: {key: df_data} ou key est le nom du fichier, et df_data ses données au format MRVeil
    :param results: {key_f: [{modele_name: key_m, predictions: df_pred}]}
                        où key_f est le nom du fichier, key_m le nom du modèle et df_pred les résultats de prédictions
    :param output_format: format de sortie (csv ou excel)
    :return: path du fichier sauvegardé
    """

    # 1
    file_names, df_data = zip(*loaded_files.items())
    file_names = [os.path.basename(f) for f in file_names]
    df_data = pd.concat(df_data)
    df_data = df_data.fillna('')

    # 2
    df_result = pd.DataFrame()
    df_result['fichier'] = file_names
    N = []
    for text in file_names:
        try:
            N.append(re.findall(r'\d+', text)[-1])
        except:
            N.append('INCONNU')
    # logger.info(N)
    df_result['NUMERO_DECLARATION'] = N
    df_result['NUMERO_DECLARATION'] = df_data['NUMERO_DECLARATION'].tolist()
    df_result['TYPE_VIGILANCE'] = df_data['TYPE_VIGILANCE'].tolist()

    # 3
    def get_model_result(results, filename, modelname):
        r_file = results[filename]
        r_model = [v for v in r_file if v['model_name'] == modelname][0]
        return r_model['predictions']

    DCO, DYSFONCTIONNEMENT, CONSEQUENCES, EFFETS = [], [], [], []
    DCO_ID, TDY_ID, CDY_ID, TEF_ID = [], [], [], []

    proba_dco = [get_model_result(results, f, 'DCO') for f in file_names]  # Liste des proba pour les n pdf
    for prob in proba_dco:
        # On sélectionne seulement la réponse la plus problable
        char_data = prob.iloc[0]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + \
            '  ('+str(round(char_data['proba'], 3))+')'
        DCO.append(text)
        DCO_ID.append(int(char_data["class"]))
    df_result['DCO'] = DCO
    df_result['DCO_ID'] = DCO_ID

    proba_dysfonc = [get_model_result(results, f, 'dysfonctionnement') for f in file_names]
    for prob in proba_dysfonc:
        char_data = prob.iloc[0]
        if char_data["class"] == 'D0':
            char_data = prob.iloc[1]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + '  (' + str(round(char_data['proba'], 3)) + ')'
        DYSFONCTIONNEMENT.append(text)
        TDY_ID.append(char_data["class"])
    df_result['Dysfonctionnements'] = DYSFONCTIONNEMENT
    df_result['TDY_ID'] = TDY_ID

    proba_conseq = [get_model_result(results, f, 'consequence') for f in file_names]
    for i, prob in enumerate(proba_conseq):
        char_data = prob.iloc[0]
        if (char_data["class"] == 'C0') \
                and (df_data['TYPE_VIGILANCE'].iloc[i].lower() in ['réactovigilance', 'materiovigilance']):
            char_data = prob.iloc[1]
        # On concatène le texte et la proba
        text = char_data['class_name'] + '  (' + str(round(char_data['proba'], 3)) + ')'
        CONSEQUENCES.append(text)
        CDY_ID.append(char_data["class"])
    df_result['Consequences'] = CONSEQUENCES
    df_result['CDY_ID'] = CDY_ID

    proba_effets = [get_model_result(results, f, 'effet') for f in file_names]
    for i, prob in enumerate(proba_effets):
        char_data = prob.iloc[0]
        if (char_data["class"] == 'E1213') \
                and (df_data['TYPE_VIGILANCE'].iloc[i].lower() in ['réactovigilance', 'reactovigilance']):
            char_data = prob.iloc[1]
        # On concatène le texte et la proba
        text = char_data['class_name'] + '  (' + str(round(char_data['proba'], 3)) + ')'
        EFFETS.append(text)
        TEF_ID.append(char_data["class"])
    df_result['Effets'] = EFFETS
    df_result['TEF_ID'] = TEF_ID

    GRAVITE_2, GRAVITE_5 = [], []

    proba_grav_1234 = [get_model_result(results, f, 'gravité_ordinale') for f in file_names]
    for prob in proba_grav_1234:
        char_data = prob.iloc[0]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + '  (' + str(round(char_data['proba'], 3)) + ')'
        GRAVITE_5.append(text)

    proba_grav_bin = [get_model_result(results, f, 'gravité_binaire') for f in file_names]
    for prob in proba_grav_bin:
        char_data = prob.iloc[0]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + '  (' + str(round(char_data['proba'], 3)) + ')'
        GRAVITE_2.append(text)

    df_result['GRAVITE_5'], df_result['GRAVITE_2'] = GRAVITE_5, GRAVITE_2


    TOPICS, CLUSTER = [], []

    proba_topics = [get_model_result(results, f, 'topics') for f in file_names]
    for prob in proba_topics:
        char_data = prob.iloc[0]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + '  (' + str(round(char_data['proba'], 3)) + ')'
        TOPICS.append(text)

    proba_clusters = [get_model_result(results, f, 'cluster') for f in file_names]
    for prob in proba_clusters:
        char_data = prob.iloc[0]
        # On concaténe le texte et la proba
        text = char_data['class_name'] + '  (' + str(round(char_data['proba'], 3)) + ')'
        CLUSTER.append(text)

    df_result['TOPIC'], df_result['CLUSTER'] = TOPICS, CLUSTER

    output_file = ''
    if output_format == 'csv':
        output_file = '/tmp/output.csv'
        df_result.to_csv(output_file, index=False)
    elif output_format == 'excel':
        output_file = '/tmp/output.xlsx'
        df_result.to_excel(output_file, index=False)
    return output_file
