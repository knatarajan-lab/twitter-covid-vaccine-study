# coding=utf-8

# Python imports
import re

# Third-party imports
import pandas as pd

# Project imports
from resources.constants import data_path, DATA_COLUMNS, PUNCTS


def load_emojis():
    emoji_path = data_path / 'emoji_map.jsonl'
    e_df = pd.read_json(emoji_path, lines=True)
    e_dict = e_df.to_dict('series')
    emoji_dict = dict(zip(e_dict["emoji"].to_list(), e_dict["value"].to_list()))
    return emoji_dict


def clean_text(text, emojis, re_puncts, re_spaces):
    # emoji_text = ''.join([emojis.get(c, c) for c in text])
    ascii_text = text.encode('ascii', 'ignore').decode()
    non_and = ascii_text.replace("&", " and ")
    non_punctuated = re.sub(re_puncts, ' ', non_and)
    single_spaced = re.sub(re_spaces, ' ', non_punctuated)
    non_caps = single_spaced.lower()
    non_digit = re.sub(r'\d+', '', non_caps)
    monospaced = re.sub(r'\s+', ' ', non_digit)
    return monospaced


def load_covid_df(data_folder):
    """
    Loads the dataset from csv
    :return:
    """
    re_puncts = re.compile(PUNCTS)
    re_spaces = re.compile(r'\s+')
    emojis = load_emojis()
    train_path = data_path / data_folder
    df = pd.DataFrame(columns=DATA_COLUMNS,
                      dtype=str)
    for file in train_path.glob('*.jsonl'):
        file_df = pd.read_json(file,
                               lines=True,
                               dtype=str)
        file_df['id'] = file_df['id'].astype('int64')
        df = pd.concat([df, file_df], ignore_index=True)
    for file in train_path.glob('*.csv'):
        file_df = pd.read_csv(file,
                              delimiter=',',
                              header=0,
                              names=DATA_COLUMNS,
                              dtype=str)
        df = pd.concat([df, file_df], ignore_index=True)
    df['text'] = df['text'].apply(clean_text, args=(emojis, re_puncts, re_spaces))
    return df
