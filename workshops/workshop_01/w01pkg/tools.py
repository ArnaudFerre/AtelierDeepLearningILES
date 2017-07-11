import re
import os


def _load_dataset(data_file, word_dict, word_set, mapping):

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

            current_labels.append(mapping.index(parts[-1]))

    if len(current_sequence) > 0:
        x_data.append(current_sequence)
        y_data.append(current_labels)

    return x_data, y_data
