import csv
import json
import logging
from datetime import date, timedelta

import pandas as pd

from resources.constants import data_path

logger = logging.getLogger(__name__)

COLS = [
    "keyword", "start_hour", "end_hour", "place_id", "retweet_count",
    "reply_count", "like_count", "quote_count", "created_at", "text", "id",
    "author_id", "newest_id", "oldest_id", "result_count"
]


def process_tweets_file(date_str):
    all_records_df = pd.DataFrame()
    input_file_path = data_path / 'v5' / f'store_tweets_{date_str}.jsonl'
    output_dir = data_path / 'GCS_v5' / f'dated={date_str}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / 'tweets.csv'
    with input_file_path.open('r') as f, output_file_path.open('w+') as g:
        records = []
        for line in f.readlines():
            line_dict = {}
            js_obj = json.loads(line)
            if 'newest_id' not in js_obj['tweet']:
                line_dict['keyword'] = js_obj['keyword']
                line_dict['start_hour'] = js_obj['start_hour']
                line_dict['end_hour'] = js_obj['end_hour']
                if 'geo' in js_obj['tweet']:
                    line_dict['place_id'] = js_obj['tweet']['geo']['place_id']
                else:
                    line_dict['place_id'] = None
                js_obj['tweet'].pop('geo', None)
                js_obj['tweet']['text'] = js_obj['tweet']['text'].replace(
                    '\n', ' ').replace('\r', ' ')
                line_dict.update(**js_obj['tweet'].pop('public_metrics'))
                line_dict.update(**js_obj['tweet'])
                records.append(line_dict)
            else:
                for record in records:
                    record.update(**js_obj['tweet'])
                records_df = pd.DataFrame.from_records(records)
                if all_records_df.size == 0:
                    all_records_df = records_df
                else:
                    all_records_df = all_records_df.append(records_df)
                records = []
        all_records_df.index.name = 'index'
        all_records_df.to_csv(path_or_buf=g,
                              columns=COLS,
                              quoting=csv.QUOTE_NONNUMERIC,
                              index=False)


def process_all_dates():
    start_date = date.fromisoformat('2021-01-10')
    end_date = date.fromisoformat('2022-01-10')
    delta = end_date - start_date
    for i in range(delta.days + 1):
        dt = start_date + timedelta(days=i)
        dt_str = dt.isoformat()
        process_tweets_file(dt_str)


def main():
    process_all_dates()


if __name__ == '__main__':
    main()
