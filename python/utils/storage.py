import os
import datetime

# import nmslib
import hnswlib

import numpy as np
import pandas as pd
import joblib

import app_env
import helpers

from utils.data_loader import DataLoader
from utils.sent2vec import Sent2Vec
from utils.tokenizer import tokenize

import concurrent.futures
import nltk
import operator


class Storage:
    def __init__(self):
        self.backend = None
        self.data = None
        self.sv_model = None

    def init(self):
        # создадим и загрузим объект-индексатор, который вычисляет релевантные ответы

        # self.backend = nmslib.init(method='hnsw', space='cosinesimil')
        self.backend = hnswlib.Index(space='cosine',
                                     dim=helpers.embedding_size)

        if self.storage_file_path_exists():
            # self.backend.loadIndex(self.storage_file_path())
            self.backend.load_index(self.storage_file_path(),
                                    max_elements=helpers.hnswlib_max_elements)


        if self.storage_data_file_path_exists():
            self.data, self.sv_model = joblib.load(self.storage_data_file_path())


    def is_inited(self):
        return self.backend is not None

    def is_need_build(self):
        if self.backend is None or self.data is None or self.sv_model is None:
            return True
        else:
            return False


    def storage_file_path(self, for_save = False):
        return app_env.data_model_runtime_path('storage.bin', not_builtin=for_save)

    def storage_file_path_exists(self):
        return os.path.exists(self.storage_file_path())

    def storage_data_file_path(self, for_save = False):
        return app_env.data_model_runtime_path('storage_data', not_builtin=for_save)

    def storage_data_file_path_exists(self):
        return os.path.exists(self.storage_data_file_path())


    def query_to_embedding(self, query):
        # токенизируем запрос
        query = tokenize(query)
        # если запрос нулевой, то возвращаем пустой список вместо выдачи
        if len(query) == 0:
            # raise Exception('empty tokenized query')
            return []
        # возьмем эмбеддинг запроса
        try:
             return self.sv_model.get_embedding(query)
        except:
            return None


    def get_by_query(self, query, top, project_ids=None):

        if self.backend is None or self.data is None:
            return {"neighbors": np.array([])}
            
        query_tok = tokenize(query)
        query_embed = self.query_to_embedding(query)
        
        # indices, _ = self.backend.knnQuery(query_embed, k=top)
        indices, _ = self.backend.knn_query(query_embed, k=top)

        data = self.data

        # indices = list(indices)
        # indices = [a for a in indices if a in data['project_id'].tolist()]

        # top_data = data.set_index('project_id').loc[indices].reset_index(inplace=False)
        # topn_tok_list = top_data['tokenized'].apply(lambda x: x.split()[:200]).tolist()

        topn_tok_list = data.loc[indices]['tokenized'].apply(lambda x: x.split()[:200]).tolist()

        common_w_len = [len(set(query_tok[:200])&set(topn_tok_list[ix])) for ix, _ in enumerate(indices)]
        common_ngram_len = [len(set(nltk.ngrams(query_tok[:200],2))&set(nltk.ngrams(topn_tok_list[ix],2))) for ix, _ in enumerate(indices)] 
        neighbors = [x for (_,_,x) in sorted(zip(common_ngram_len,common_w_len,indices), key=operator.itemgetter(0,1), reverse=True)]
        
        if project_ids is not None:
            data = data[data['project_id'].isin(project_ids)]
            neighbors = [x for x in neighbors if x in data.index]
            # neighbors = data.index & neighbors

        if len(neighbors) == 0:
            return {"neighbors": np.array([])}

        return {"neighbors": np.array(neighbors)}

    def build(self, limit=None, data_loader=None, worker_count=app_env.ml_params_vcpu_limit()):
        self.log("Start building index with new algorithm")
        self.log("Checking if old storage exists")
        if self.storage_data_file_path_exists() and self.storage_file_path_exists():
            self.log("Storage data exists! Reading storage data...")
            main_data, old_sv_model = joblib.load(self.storage_data_file_path())
            self.log("Storage successfully read")
            self.log("Getting ideas in storage")
            old_ideas_ids = set(main_data.index)
            self.log(f"{len(old_ideas_ids)} ideas found")
            hoursCount = 24
            self.log(f"Getting ideas from DB for last {hoursCount} hours ")
            data_loader = DataLoader(limit, tokenize_thread_count=worker_count)
            new_ideas_ids = data_loader.ideas_ids_for_last_hours(hoursCount)
            self.log(f"Got ideas form DB. Count: {len(new_ideas_ids)}")
            self.log("Checking ideas which are in db but not in storage yet")
            new_ideas_to_index =  new_ideas_ids - old_ideas_ids
            if len(new_ideas_to_index) == 0:
                self.log("No new ideas to add to index")
                return
            else:
                self.log(f"Ideas to add to storage found. Count: {len(new_ideas_to_index)}")
                self.log("Loading ideas from db")
                ideas_to_add_to_index = data_loader.ideas_with_tokens(ids=new_ideas_to_index)
                self.log(f"Building vectors to new ideas. Count: {len(ideas_to_add_to_index)}")
                ideas_to_add_to_index = self.dataframe_text_embedding(ideas_to_add_to_index, old_sv_model)
                self.log("Vectors successfully built")

                self.log("Adding new ideas to old data storage")
                self.log(f"Shape before: {main_data.shape[0]}")
                self.log(f"Tail before: {main_data.index[-1]}")
                new_data = pd.concat([main_data, ideas_to_add_to_index[['project_id', 'text_embedding', 'tokenized']]])
                self.log(f"Shape after: {new_data.shape[0]}")
                self.log(f"Tail after: {new_data.index[-1]}")
                self.log(f"Saving old data storage: {self.storage_data_file_path(for_save=True)}")
                joblib.dump([new_data, old_sv_model], self.storage_data_file_path(for_save=True))
                self.log("Storage data saved successfully")

                self.log("Loading storage from file")
                # old_storage = nmslib.init(method='hnsw', space='cosinesimil')
                # old_storage.loadIndex(self.storage_file_path())
                old_storage = hnswlib.Index(space='cosine',
                                             dim=helpers.embedding_size)
                old_storage.load_index(self.storage_file_path(),
                                        max_elements=helpers.hnswlib_max_elements)

                self.log("Adding ideas to old_storage")
                updated_storage = self.build_storage(ideas_to_add_to_index, old_storage=old_storage)
                self.log("Ideas added successfully")
                self.log(f"Save index to file '{self.storage_file_path(for_save=True)}'")
                # updated_storage.saveIndex(self.storage_file_path(for_save=True))
                updated_storage.save_index(self.storage_file_path(for_save=True))

                # self.log("Recreating storage")
                # updated_storage = self.build_storage(new_data)
                # self.log("Storage recreated successfully")
                # self.log(f"Save index to file '{self.storage_file_path(for_save=True)}'")
                # # updated_storage.saveIndex(self.storage_file_path(for_save=True))
                # updated_storage.save_index(self.storage_file_path(for_save=True))

                self.backend = updated_storage
                self.data = main_data
                self.sv_model = old_sv_model

        else:
            self.log("Storage doesn't exist. Falling back to old algorithm")
            self.build_old(limit=limit, data_loader=data_loader, worker_count=worker_count)

    def build_old(self, limit=None, data_loader=None, worker_count=app_env.ml_params_vcpu_limit()):
        print("[Storage] Start build new index and data")

        if data_loader is None:
            data_loader = DataLoader(limit, tokenize_thread_count=worker_count)

        print("[Storage] Fetch ideas from BD. Limit {limit}".format(limit=limit))
        df = data_loader.ideas_with_tokens()
        print("[Storage] Get {count} ideas".format(count=len(df)))

        print("[Storage] Start build vectors")

        sv_model = Sent2Vec(helpers.ft_model, helpers.embedding_size)

        df = self.dataframe_text_embedding(df, sv_model)

        storage = self.build_storage(df)

        print("[Storage] Save index to file '{file_path}'".format(file_path=self.storage_file_path(for_save=True)))
        # storage.saveIndex(self.storage_file_path(for_save=True))
        storage.save_index(self.storage_file_path(for_save=True))

        print("[Storage] Save data to file '{file_path}'".format(file_path=self.storage_data_file_path(for_save=True)))
        data = df[['project_id', 'text_embedding', 'tokenized']]
        joblib.dump([data, sv_model], self.storage_data_file_path(for_save=True))

        print("[Storage] Update index and data")
        self.backend = storage
        self.data = data
        self.sv_model = sv_model

        print("[Storage] Done")

    def dataframe_text_embedding(self, df, sv_model):
        df['text_embedding'] = None

        def toks(data, sv_model):
            result = {}
            for idea_id in data:
                try:
                    result[idea_id] = sv_model.get_embedding(data[idea_id]['tokenized'].split())
                except:
                    result[idea_id] = None
                    print("[Storage] WARNING: Skip tokenize idea {idea_id}. Cannot build embeddings for tokenize: '{"
                          "tokenize}'".format(idea_id=idea_id, tokenize=data[idea_id]['tokenized']))
            return result

        def future_callback(_fut):
            result = _fut.result()
            for idea_id in result:
                df.at[idea_id, 'text_embedding'] = result[idea_id]

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Start the load operations and mark each future with its URL
            for idea_ids in helpers.chunks(df.index, 1000):
                future = executor.submit(toks,
                                         df.loc[idea_ids][['tokenized']].to_dict('index'),
                                         sv_model)
                future.add_done_callback(future_callback)

        print("[Storage] Drop row with emtpy text_embedding")
        new_df = df[df.apply(lambda x: x['text_embedding'] is not None, axis=1)]
        print("[Storage] Done tokenize")
        return new_df

    def build_storage(self, df, old_storage = None):
        print("[Storage] Start build index")
        if old_storage is None:
            # storage = nmslib.init(method='hnsw', space='cosinesimil')

            storage = hnswlib.Index(space='cosine',
                                         dim=helpers.embedding_size)

            storage.init_index(max_elements=helpers.hnswlib_max_elements,
                                    ef_construction=helpers.hnswlib_ef_construction,
                                    M=helpers.hnswlib_M,
                                    random_seed=42)

            storage.set_ef(helpers.hnswlib_ef)

        else:
            storage = old_storage
        
        # for _, row in df.iterrows():
        #     storage.addDataPoint(row.name, row.text_embedding)

        data = np.stack(df.text_embedding)
        data_labels = np.array(df.index)
        storage.add_items(data, data_labels)

        # print("[Storage] Done fill index")
        # if old_storage is None:
        #     storage.createIndex({'post': 2})

        print("[Storage] Index builded")
        return storage

    def log(self, message):
        print(f"[Storage][{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

