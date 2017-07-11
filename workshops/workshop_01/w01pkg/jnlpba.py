import os

import gensim

from .tools import load_dataset

jnlpba_mapping = ["B-cell_line", "B-cell_type", "B-DNA", "B-protein", "B-RNA", "I-cell_line", "I-cell_type", "I-DNA",
                  "I-protein", "I-RNA", "O"]


def load_jnlpba(train_file, test_file, gensim_model_path):

    gensim_model = gensim.models.Word2Vec.load(os.path.abspath(gensim_model_path))

    word_set = set(gensim_model.wv.index2word)
    word_dict = {}
    for i, key in enumerate(gensim_model.wv.index2word):
        word_dict[key] = i

    x_train, y_train = load_dataset(train_file, word_dict, word_set, jnlpba_mapping)
    x_test, y_test = load_dataset(test_file, word_dict, word_set, jnlpba_mapping)

    return (x_train, y_train), (x_test, y_test)
