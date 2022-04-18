import os

import pandas as pd
import sqlalchemy

import joblib

import app_env
import helpers

from utils.tokenizer import tokenize

import concurrent.futures


class DataLoader:
    def __init__(self, limit=None, tokenize_thread_count=app_env.ml_params_vcpu_limit()):
        self.__idea_with_tokens = None
        self.__limit = limit
        self.__tokenize_thread_count = tokenize_thread_count

    def ideas(self):
        sql = """
SELECT id AS idea_id, ({index_column}) AS content, coalesce(parent_projects.parent_project_id, parent_id) AS project_id
FROM pages
  LEFT OUTER JOIN LATERAL (
    SELECT projects.parent_id AS parent_project_id, projects.id AS project_id
    FROM pages AS projects
      INNER JOIN pages AS parent_projects ON parent_projects.type = 'Pim::ParentProject' AND parent_projects.id = projects.parent_id
    WHERE pages.parent_id = projects.id
  ) parent_projects ON pages.parent_id = parent_projects.project_id
WHERE type IN ('Pim::Idea', 'Pim::VndIdea', 'Pim::BestPracticeIdea')
      AND (content IS NOT NULL OR trim(content) <> '')
ORDER BY idea_id
        """

        index_column = map(lambda x: "COALESCE({x})".format(x=x), app_env.ml_params_index_content_columns())

        sql = sql.format(index_column=" ||  ' ' || ".join(index_column))

        params = {}
        if self.__limit is not None:
            sql += "\nLIMIT :limit"
            params['limit'] = self.__limit

        sql = sqlalchemy.text(sql)

        ideas = pd.read_sql_query(sql, app_env.db_engine(), params=params, index_col='idea_id')

        return ideas

    def ideas_with_tokens(self):
        if self.__idea_with_tokens is not None:
            return self.__idea_with_tokens

        if app_env.ml_params_learn_data_use_cache():
            if self.cache_file_path_exists('ideas_with_tokens'):
                print("[DataLoader] Reader data from cache file {file_path}".
                      format(file_path=self.cache_file_path('ideas_with_tokens')))
                self.__idea_with_tokens = joblib.load(self.cache_file_path('ideas_with_tokens'))
                return self.__idea_with_tokens

        print("[DataLoader] Fetch ideas from BD. Limit {limit}".format(limit=self.__limit))
        df = self.ideas()

        print("[DataLoader] Idea count is {count}".format(count=len(df)))

        print("[DataLoader] Filling tokenized column in dataframe")
        df['tokenized'] = ''
        self.dataframe_tokenized(df)

        print("[DataLoader] Filter dataframe with emtpy tokenized")
        df = df[df.apply(lambda x: len(x['tokenized']) != 0, axis=1)]
        print("[DataLoader] Count row after filtering is {count}".format(count=len(df)))

        self.__idea_with_tokens = df

        if app_env.ml_params_learn_data_use_cache():
            print("[DataLoader] Save cache data to file: {file_path}".
                  format(file_path=self.cache_file_path('ideas_with_tokens', for_save=True)))
            joblib.dump(self.__idea_with_tokens, self.cache_file_path('ideas_with_tokens', for_save=True))

        return self.__idea_with_tokens

    def dataframe_tokenized(self, df):
        def toks(data):
            result = {}
            for idea_id in data:
                result[idea_id] = ' '.join(tokenize(data[idea_id]['content']))
            return result

        def future_callback(_fut):
            result = _fut.result()
            df.loc[result.keys(), 'tokenized'] = [result[k] for k in result]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Start the load operations and mark each future with its URL
            for idea_ids in helpers.chunks(df.index, 1000):
                future = executor.submit(toks, df.loc[idea_ids][['content']].to_dict('index'))
                future.add_done_callback(future_callback)

        print("[DataLoader] Done tokenize")

    def cache_file_path(self, name, for_save=False):
        limit_str = 'all'
        if self.__limit is not None:
            limit_str = str(self.__limit)
        return app_env.data_model_runtime_path("data_loader__{name}__{limit}".format(name=name, limit=limit_str),
                                               not_builtin=for_save)

    def cache_file_path_exists(self, name):
        return os.path.exists(self.cache_file_path(name))
