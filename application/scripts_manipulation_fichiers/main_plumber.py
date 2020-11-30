"""
Code to Process PDF with pdfplumber
"""

import pdfplumber
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import fnmatch
import logging
import time


def findfiles(which, where='.'):
    """
    Returns list of filenames from `where` path matched by 'which'
       shell pattern. Matching is case-insensitive.

    :param which: regexp to find files
    :param where: directory to search
    :return: list of files
    """
    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    path = Path(where)
    list_of_files = []
    for file in path.iterdir():
        list_of_files.append(file)
    return [name for name in list_of_files if rule.match(name.name)]


def plumber_df(my_file: str):
    """
    Extract tables list and df with plumber

    :param my_file: file to process
    :return: list of dataframes (one / input page), concatenated dataframe
    """
    pdf_object = pdfplumber.open(my_file)
    char_df = pd.DataFrame()
    tmp_lst = []
    for i, page in enumerate(pdf_object.pages):
        temp_df = None
        try:
            temp_df = page.extract_table()
        except:
            logging.warning(f"No Table found in page {i} of file {my_file}")
            pass
        if temp_df is not None:
            # Save to list
            tmp_lst.append(temp_df)
            try:
                char_df = char_df.append(temp_df)
            except:
                logging.warning(f"Failed to concatenate table of page {i}")
                pass

    char_df.columns = ["Column0", "Column1"]

    return tmp_lst, char_df


def plumber_(filepath: str, output_path: str, concat_path: str):
    """
    Save plumber extractions

    :param filepath: file to process
    :param output_path: path to save individual pages to
    :param concat_path:  path to save concatenated dataframe to
    :return: str
    """
    filename = Path(filepath).name

    # Plumber extract table from each page, save them and concatenate in a Df
    tables, table_df = plumber_df(filepath)
    # Plumber extract table from each page and save them
    for i, table in enumerate(tables):
        df_to_save = pd.DataFrame(table[1:], columns=["Column0", "Column1"])
        df_to_save.to_csv(f"{output_path}{filename}_{i}.csv", sep="|")

    if table_df.shape[0] > 0:
        table_df.to_csv(f"{concat_path}{filename}.csv", sep='|')

    return f"{filename} was correctly processed"


def main(working_dir: str, processed_dir: str, concat_dir: str):
    # Logging
    logging.basicConfig(
        filename='parsrlog.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.WARNING,
        datefmt='%Y-%m-%d %H:%M:%S')

    # List all PDF files
    list_of_files = findfiles('*.pdf', working_dir)
    concatenate_files = findfiles('*.csv', concat_dir)
    concatenate_files = [name.stem for name in concatenate_files]

    for file in tqdm(list_of_files):
        filename = Path(file).stem
        filename_check = Path(file).name
        if filename_check not in concatenate_files:
            print(f"Processing {filename}")
            try:
                plumber_(str(file), processed_dir, concat_dir)
            except:
                logging.warning(f"Failed to read Pdf file {filename}")
                pass

        else:
            print(f"File {filename} was already processed")


if __name__ == "__main__":
    working_dir = 'working_dir/'
    processed_dir = 'processed_dir/'
    concat_dir = 'concat_dir/'
    start_time = time.time()
    main(working_dir, processed_dir, concat_dir)
    print("--- %s seconds ---" % (time.time() - start_time))