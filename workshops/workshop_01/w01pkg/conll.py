import os
import re
from collections import defaultdict

import requests

PARAMS = {"annotators": "tokenize,ssplit,pos", "outputFormat": "json"}


def convert_to_conll(input_corpus_file, output_conll_file, corenlp_url):

    regex_title = re.compile("^(\d+)\|t\|(.*)$")
    regex_abstract = re.compile("^(\d+)\|a\|(.*)$")
    regex_annotation = re.compile("^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t([^\t]+)\t([^\t]+)$")

    instances = dict()

    with open(os.path.abspath(input_corpus_file), "r", encoding="UTF-8") as input_file:
        for line in input_file:

            if re.match("^$", line):
                continue

            match_title = regex_title.match(line)
            if match_title:
                instances[int(match_title.group(1))] = {
                    "title": match_title.group(2),
                    "annotations": list()
                }

            match_abstract = regex_abstract.match(line)
            if match_abstract:
                instances[int(match_abstract.group(1))]["abstract"] = match_abstract.group(2)

            match_annotation = regex_annotation.match(line)
            if match_annotation:
                instances[int(match_annotation.group(1))]["annotations"].append({
                    "begin": int(match_annotation.group(2)),
                    "end": int(match_annotation.group(3)),
                    "text": match_annotation.group(4),
                    "entity": match_annotation.group(5),
                    "tokens": list()
                })

    with open(os.path.abspath(output_conll_file), "w", encoding="UTF-8") as output_file:
        for pmid, content in instances.items():

            r = requests.post(corenlp_url, params=PARAMS, data="{} {}".format(
                content["title"],
                content["abstract"]
            ))
            r_content = r.json()

            annotations_sorted = sorted(content["annotations"], key=lambda tup: tup["begin"])

            for sentence in r_content["sentences"]:
                for token in sentence["tokens"]:
                    for annotation in annotations_sorted:

                        if annotation["begin"] == token["characterOffsetBegin"] and \
                                        token["characterOffsetEnd"] <= annotation["end"]:

                            annotation["tokens"].append((sentence["index"], token["index"]))

                        elif annotation["end"] == token["characterOffsetEnd"] and \
                                        annotation["begin"] <= token["characterOffsetBegin"]:

                            annotation["tokens"].append((sentence["index"], token["index"]))

                        elif annotation["begin"] <= token["characterOffsetBegin"] < token["characterOffsetEnd"] \
                                <= annotation["end"]:

                            annotation["tokens"].append((sentence["index"], token["index"]))

                for token in sentence["tokens"]:

                    label = "O"

                    for annotation in annotations_sorted:

                        if (sentence["index"], token["index"]) in annotation["tokens"]:
                            if annotation["tokens"].index((sentence["index"], token["index"])) == 0:
                                label = "B-Disease"
                            else:
                                label = "I-Disease"

                    line_str = "{}\t{}\t{}\n".format(
                        token["originalText"],
                        token["pos"],
                        label
                    )
                    output_file.write(line_str)

                output_file.write("\n")


def remove_label(input_tab_file, output_tab_file):

    with open(os.path.abspath(input_tab_file), "r", encoding="UTF-8") as input_file:
        with open(os.path.abspath(output_tab_file), "w", encoding="utf-8") as output_file:
            for line in input_file:
                if re.match("^$", line):
                    output_file.write(line)
                else:
                    parts = line.rstrip("\n").split("\t")
                    output_file.write("{}\t{}\n".format(parts[0], parts[1]))


def eval_output(system_tab_file, gs_tab_file):

    system_annotations = list()
    gs_annotations = list()

    with open(os.path.abspath(system_tab_file), "r", encoding="UTF-8") as sys_file:
        for line in sys_file:

            if re.match("^$", line):
                continue

            parts = line.rstrip("\n").split('\t')
            system_annotations.append((parts[0], parts[-1]))

    with open(os.path.abspath(gs_tab_file), "r", encoding="UTF-8") as gs_file:
        for line in gs_file:

            if re.match("^$", line):
                continue

            parts = line.rstrip("\n").split('\t')
            gs_annotations.append((parts[0], parts[-1]))

    gs_types = dict()
    for _, label in gs_annotations:
        if label not in gs_types:
            gs_types[label] = defaultdict(int)

    for _, label in system_annotations:
        if label not in gs_types:
            raise Exception("A predicted label seems incorrect: {}".format(label))

    gs_types["all"] = defaultdict(int)

    for (_, label_sys), (_, label_gs) in zip(system_annotations, gs_annotations):

        gs_types[label_sys]["pred"] += 1
        gs_types[label_gs]["ref"] += 1

        gs_types["all"]["pred"] += 1
        gs_types["all"]["ref"] += 1

        if label_gs == label_sys:
            gs_types[label_gs]["corr"] += 1
            gs_types["all"]["corr"] += 1

    final_scores = dict()

    for label, scores in gs_types.items():

        precision = scores["corr"]/scores["pred"]
        recall = scores["corr"]/scores["ref"]
        f1_measure = (2 * precision * recall) / (precision + recall)

        final_scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1_measure": f1_measure
        }

        for k, v in scores.items():
            final_scores[label][k] = v

    return final_scores
