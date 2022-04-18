import numpy as np


# создадим класс, который отвечает за перевод текста в вектор
class Sent2Vec:
    # обозначим инициализацию
    def __init__(self, embed_model, default_size=300):
        self.embed_model = embed_model  # эмбеддинги
        self.default_size = default_size  # размер итоговых эмбеддингов

    # объявим метод для получения эмбеддингов предложений
    def get_embedding(self, sentence):
        # sentence = [x for x in sentence if x in self.embed_model.wv]
        if len(sentence) == 0:
            # return np.zeros(300)
            raise Exception("embedding model doesn't contain words from the sentence")
        else:
            embedding = [self.embed_model.wv[x] for x in sentence]
            embedding = np.mean(embedding, axis=0)
            return embedding