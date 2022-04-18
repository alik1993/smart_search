import os
import sqlalchemy


def data_model():
    return os.getenv('MODEL_DATA', os.path.join('..', 'data'))


def data_model_builtin():
    return os.path.join(data_model(), 'builtin')


def data_model_builtin_path(path):
    return os.path.join(data_model_builtin(), path)


def data_model_runtime():
    return os.path.join(data_model(), 'runtime')


def data_model_runtime_path(path, not_builtin=False):
    rt_path = os.path.join(data_model_runtime(), path)
    if not_builtin is True:
        return rt_path

    if os.path.exists(rt_path):
        return rt_path
    else:
        return data_model_builtin_path(path)


# def database_url():
#     return os.environ['DATABASE_URL']

def database_url():
     f = open("/var/run/secrets/dbpass", "r")
     dbpass = f.readline()
     f.close()
     db_url=os.environ['DATABASE_URL']
     return db_url+dbpass


def database_search_path():
    return os.getenv('DATABASE_SEARCH_PATH', 'tenant1,public,extensions')



def db_engine():
    return sqlalchemy.create_engine(database_url(),
                                    connect_args={'options': '-csearch_path={}'.format(database_search_path())})


def ml_params_use_gpu():
    return os.getenv('ML_PARAMS_USE_GPU', 'false') == 'true'


def ml_params_cuda_devise_id():
    return os.getenv('ML_PARAMS_CUDA_DEVISE_ID', '')


def ml_params_cuda_devise():
    res = ['cuda']
    if ml_params_cuda_devise_id() != ():
        res.append(ml_params_cuda_devise_id())

    return ':'.join(res)


def ml_params_vcpu_limit():
    return int(os.getenv('ML_PARAMS_VCPU_LIMIT', '1'))


def ml_params_memory_limit():
    return int(os.getenv('ML_PARAMS_MEMORY_LIMIT', '1024'))


def ml_params_data_loader_num_workers():
    return int(os.getenv('ML_PARAMS_DATA_LOADER_NUM_WORKERS', '4'))


def ml_params_dataset_learn_limit():
    v = os.getenv('ML_PARAMS_DATASET_LEARN_LIMIT', None)
    if v is None:
        return 0
    else:
        return int(v)


def idea_indexes_path():
    return data_model_runtime_path('idea_indexes', not_builtin=True)


BOOLEAN_STRING = ['1', 'true', 'on']


def ml_learn_data_use_cache():
    value = os.getenv('ML_PARAMS_DATA_USE_CACHE', 'false')

    return any(v in value for v in BOOLEAN_STRING)


def ml_params_learn_data_use_cache():
    value = os.getenv('ML_PARAMS_LEARN_DATA_USE_CACHE', 'false')

    return any(v in value for v in BOOLEAN_STRING)


def ml_params_index_content_columns():
    return os.getenv('ML_PARAMS_INDEX_CONTENT_COLUMNS', 'content').split(',')
