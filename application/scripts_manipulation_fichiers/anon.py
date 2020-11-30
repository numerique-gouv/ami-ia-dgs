"""
Script d'anonymisation de fichiers inputs en csv
"""
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import fnmatch
import logging


def findfiles(which, where='.'):
    """
    Returns list of filenames from `where` path matched by 'which'
       shell pattern. Matching is case-insensitive.

    :param which: regex to match filenames
    :param where: path to search
    :return: list of files
    """
    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    path = Path(where)
    list_of_files = []
    for file in path.iterdir():
        list_of_files.append(file)
    return [name for name in list_of_files if rule.match(name.name)]


def process_files(list_of_csv: list, check_list: list, output_dir: str):
    """
    Function to anonymize a list of csv files.

    :param list_of_csv: list of files to process
    :param check_list: list of files NOT to process
    :param output_dir: output directory
    :return:
    """
    for file in tqdm(list_of_csv):
        filename = Path(file).stem
        filename_check = Path(file).name
        if filename not in check_list:
            print(f"Processing {filename}")
            try:
                df = pd.read_csv(file, sep="|", index_col=0)
                df['Column0'] = df['Column0'].replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=[" ", " "],
                                                      regex=True)
                df = df[~df['Column0'].str.lower().isin([x.lower() for x in ano_champs])]
                df.to_csv(f'{output_dir}{filename_check}')
            except:
                logging.warning(f"Failed to read Pdf file {filename}")
                pass

        else:
            print(f"File {filename} was already processed")


def main(concat_dir: str, concat_anon_dir: str, processed_anon_dir: str):
    # Logging
    logging.basicConfig(
        filename='anonlog.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.WARNING,
        datefmt='%Y-%m-%d %H:%M:%S')

    # Process concat_dir folder
    print(f'Processing {concat_dir}')
    print()
    list_of_csv_concat = findfiles('*.csv', concat_dir)
    concat_anon_files = findfiles('*.csv', concat_anon_dir)
    print('concat', concat_anon_files)
    if len(concat_anon_files) > 0:
        concat_anon_files = [name.stem for name in concat_anon_files]
    process_files(list_of_csv_concat, concat_anon_files, concat_anon_dir)

    # Process processed_dir folder
    print(f'Processing {processed_dir}')
    print()
    list_of_csv_processed = findfiles('*.csv', processed_dir)
    processed_anon_files = findfiles('*.csv', processed_anon_dir)
    if len(processed_anon_files) > 0:
        processed_anon_files = [name.stem for name in processed_anon_files]
    process_files(list_of_csv_processed, processed_anon_files, processed_anon_dir)


if __name__ == "__main__":
    ano_champs = ano_champs = ["Nom", "Prénom", "Téléphone", "Adresse électronique", "Adresse postale",
                               "Code postal / Commune",
                               "Nom (3 premières lettres)", "Sexe", "Date de naissance", "ou âge (réel ou estimé)",
                               "Poids",
                               "Date de survenue", "Période de survenue", "Type de lieu de survenue",
                               "Date de survenue de l'incident",
                               "Date de détection de l'incident", "Profession", "Complément profession",
                               "Téléphone ou pour les correspondants locaux : Télécopie",
                               "Nom de l'établissement ou de l'organisme", "Age (réel ou estimé)",
                               "Nom de l'etablissement", "Adresse", "Service",
                               "Personne à contacter dans l'établissement", "Téléphone du contact",
                               "Êtes vous la personne ayant présenté les symptômes  ?",
                               "Veuillez préciser votre lien avec la personne",
                               "Nom de l'établissement ou de l'organisme",
                               "Êtes-vous le correspondant de matériovigilance ou son suppléant ?",
                               "Êtes-vous le correspondant de réactovigilance ou son suppléant ?",
                               "Si vous êtes dans un établissement de santé, il est préférable de prendre contact avec votre correspondant local de matériovigilance qui a pour rôle de recevoir et transmettre ensuite les déclarations à l'ansm. Sinon, vous pouvez les transmettre à l'ansm vous-même via ce portail."
                               ]
    concat_dir = 'concat_dir/'
    processed_dir = 'processed_dir'
    concat_anon_dir = 'concat_anon_dir/'
    processed_anon_dir = 'processed_anon_dir/'
    main(concat_dir, concat_anon_dir, processed_anon_dir)