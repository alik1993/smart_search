from utils.storage import Storage
from utils.data_loader import DataLoader

import helpers

import datetime

import warnings
warnings.filterwarnings('ignore')


# объявим класс для поиска релевантных документов по запросу
class SearchEngine:
    def __init__(self, data_limit=None):
        self.lastInit = datetime.datetime.now()
        
        self.data_loader = DataLoader(limit=data_limit)
        self.storage = Storage()

        if self.storage.is_inited() is False:
            self.storage.init()

        if self.storage.is_need_build() is True:
            self.log("Building storage info")
            self.storage.build(data_loader=self.data_loader)
            # self.__force_load_storage = False

        # self.storage.init()

    def search(self, query, n_top, project_ids=None):
        if (self.lastInit is not None and (datetime.datetime.now() - self.lastInit).total_seconds() > 240):
            self.log("Time to reinit storage info")
            try:
                self.storage.init()
            except:
                self.log("Can't init storage")
            self.lastInit = datetime.datetime.now()
            self.log("Initialization complete")

        storage_result = self.storage.get_by_query(query, helpers.search_index_top, project_ids=project_ids)
        neighbors = storage_result['neighbors']
        neighbors = neighbors[:n_top]
        return neighbors

    def log(self, message):
        print(f"[SearchEngine][{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)
