import os
import re

import gensim

from .tools import load_dataset

ncbi_mapping = ["B-Disease", "I-Disease", "O"]


def load_ncbi(train_file, dev_file, test_file, gensim_model_path):

    gensim_model = gensim.models.Word2Vec.load(os.path.abspath(gensim_model_path))

    word_set = set(gensim_model.wv.index2word)
    word_dict = {}
    for i, key in enumerate(gensim_model.wv.index2word):
        word_dict[key] = i

    x_train, y_train = load_dataset(train_file, word_dict, word_set, ncbi_mapping)
    x_dev, y_dev = load_dataset(dev_file, word_dict, word_set, ncbi_mapping)
    x_test, y_test = load_dataset(test_file, word_dict, word_set, ncbi_mapping)

    return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)


def generate_output(original_test_file, output_test_file, y_pred):

    with open(os.path.abspath(original_test_file), "r", encoding="UTF-8") as input_file:
        with open(os.path.abspath(output_test_file), "w", encoding="UTF-8") as output_file:

            index_line = 0
            index_tok = 0

            current_tokens = list()
            current_labels = list()

            for line in input_file:

                if re.match("^$", line):
                    index_line += 1
                    index_tok = 0
                    for tok, lab in zip(current_tokens, current_labels):
                        output_file.write("{}\t{}\n".format(
                            tok,
                            lab
                        ))
                    output_file.write(line)
                    current_labels = list()
                    current_tokens = list()
                    continue

                parts = line.rstrip('\n').split("\t")
                current_tokens.append(parts[0])
                current_labels.insert(0, ncbi_mapping[y_pred[index_line][len(y_pred[index_line])-index_tok-1]])

                index_tok += 1
