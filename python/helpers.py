from gensim.models.fasttext import FastText
from gensim.models.fasttext import FastTextKeyedVectors

import app_env

# ft_model = FastText.load(app_env.data_model_runtime_path('embeddings/ft_model.model'))
ft_model = FastTextKeyedVectors.load(app_env.data_model_runtime_path('embeddings/ft_model.model'))

embedding_size = 300

search_index_top = 300

hnswlib_max_elements = 1000000
hnswlib_ef_construction = 350
hnswlib_ef = 350
hnswlib_M = 50


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
