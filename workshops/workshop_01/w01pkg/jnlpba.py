import os
import re

import gensim

jnlpba_mapping = ["B-cell_line", "B-cell_type", "B-DNA", "B-protein", "B-RNA", "I-cell_line", "I-cell_type", "I-DNA",
                  "I-protein", "I-RNA", "O"]


def load_jnlpba(train_file, test_file, gensim_model_path):

    gensim_model = gensim.models.Word2Vec.load(os.path.abspath(gensim_model_path))

    word_set = set(gensim_model.wv.index2word)
    word_dict = {}
    for i, key in enumerate(gensim_model.wv.index2word):
        word_dict[key] = i

    x_train, y_train = _load_dataset(train_file, word_dict, word_set)
    x_test, y_test = _load_dataset(test_file, word_dict, word_set)

    return (x_train, y_train), (x_test, y_test)


def _load_dataset(data_file, word_dict, word_set):

    x_data = list()
    y_data = list()

    current_sequence = list()
    current_labels = list()

    with open(os.path.abspath(data_file), "r", encoding="UTF-8") as input_file:

        for i, line in enumerate(input_file):

            if re.match("^$", line):
                if len(current_sequence) > 0:
                    x_data.append(current_sequence)
                    y_data.append(current_labels)

                    current_sequence = list()
                    current_labels = list()
                continue

            parts = line.rstrip("\n").split("\t")

            token_str = re.sub("\d", "0", parts[0].lower())

            if token_str in word_set:
                current_sequence.append(word_dict[token_str])
            else:
                current_sequence.append(word_dict["#unk#"])

            current_labels.append(jnlpba_mapping.index(parts[-1]))

    if len(current_sequence) > 0:
        x_data.append(current_sequence)
        y_data.append(current_labels)

    return x_data, y_data
