import os

import matplotlib.pyplot as plt
import pandas as pd
from google.cloud.bigquery import Client
from google.auth import default

from resources.constants import JINJA_ENV, data_path

QUERY_TPL = JINJA_ENV.from_string("""
SELECT DATE(created_at) date, COUNT(DATE(created_at)) n
FROM `{{project_id}}.{{dataset_id}}.{{table_id}}`
WHERE keyword LIKE '%({{keyword}}%'
GROUP BY date
ORDER BY date
""")

keywords = ['janssen', 'moderna', 'pfizer']

if __name__ == '__main__':
    credentials, project_id = default()
    client = Client(project=project_id, credentials=credentials)
    params = {
        'project_id': os.environ.get('GOOGLE_CLOUD_PROJECT'),
        'dataset_id': 'covax_tweets',
        'table_id': 'tweets'
    }
    df_maps = {}
    for keyword in keywords:
        query_string = QUERY_TPL.render(params, keyword=keyword)
        job = client.query(query_string)
        res = job.result().to_dataframe()
        df_maps[keyword] = res.set_index(["date"]).rename({'n': keyword},
                                                          axis=1)
    all_df = pd.concat(df_maps.values(), axis=1).reset_index()

    all_df.plot(x='date', y=df_maps.keys(), kind='line', figsize=(12, 8))
    plt.tight_layout(pad=0)
    plt.savefig(data_path / 'view_trends.png')
