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

    gs_types = {
        "entities": dict(),
        "labels": dict()
    }

    for _, label in gs_annotations:
        if label not in gs_types["labels"]:
            gs_types["labels"][label] = defaultdict(int)

    for _, label in system_annotations:
        if label not in gs_types["labels"]:
            raise Exception("A predicted label seems incorrect: {}".format(label))

    gs_types["labels"]["all"] = defaultdict(int)

    gs_types["entities"] = dict()
    gs_types["entities"]["all"] = defaultdict(int)

    for (_, label_sys), (_, label_gs) in zip(system_annotations, gs_annotations):

        gs_types["labels"][label_sys]["pred"] += 1
        gs_types["labels"][label_gs]["ref"] += 1

        gs_types["labels"]["all"]["pred"] += 1
        gs_types["labels"]["all"]["ref"] += 1

        if label_gs == label_sys:
            gs_types["labels"][label_gs]["corr"] += 1
            gs_types["labels"]["all"]["corr"] += 1

    sys_entities = get_entities(system_annotations)
    gs_entities = get_entities(gs_annotations)

    for ent_idx, ent_str in sys_entities.items():

        ent_type = ent_idx[-1]
        if ent_type not in gs_types["entities"]:
            gs_types["entities"][ent_type] = defaultdict(int)

        gs_types["entities"][ent_type]["pred"] += 1
        gs_types["entities"]["all"]["pred"] += 1

        if ent_idx in gs_entities:
            gs_types["entities"][ent_type]["corr"] += 1
            gs_types["entities"]["all"]["corr"] += 1

    for ent_idx, ent_str in gs_entities.items():
        ent_type = ent_idx[-1]
        gs_types["entities"][ent_type]["ref"] += 1
        gs_types["entities"]["all"]["ref"] += 1

    final_scores = {
        "labels": dict(),
        "entities": dict()
    }

    for label, scores in gs_types["labels"].items():

        precision = scores["corr"]/scores["pred"]
        recall = scores["corr"]/scores["ref"]
        f1_measure = (2 * precision * recall) / (precision + recall)

        final_scores["labels"][label] = {
            "precision": precision,
            "recall": recall,
            "f1_measure": f1_measure
        }

        for k, v in scores.items():
            final_scores["labels"][label][k] = v

    for label, scores in gs_types["entities"].items():

        precision = scores["corr"]/scores["pred"]
        recall = scores["corr"]/scores["ref"]
        f1_measure = (2 * precision * recall) / (precision + recall)

        final_scores["entities"][label] = {
            "precision": precision,
            "recall": recall,
            "f1_measure": f1_measure
        }

        for k, v in scores.items():
            final_scores["entities"][label][k] = v

    return final_scores


def get_entities(annotations):

    entities = dict()

    previous_tag = "O"
    previous_cat = ""

    current_entity_id = list()
    current_entity_str = list()
    current_entity_cat = ""

    for i, (tok, label) in enumerate(annotations):

        if label.startswith("I"):

            current_cat = label.split("-")[1]

            if previous_tag == "O":

                current_entity_id.append(i)
                current_entity_str.append(tok)
                current_entity_cat = current_cat

                previous_tag = "I"
                previous_cat = current_cat

            else:
                if previous_cat != current_cat:

                    entities[tuple(current_entity_id)+tuple([current_entity_cat])] = " ".join(current_entity_str)

                    current_entity_id.clear()
                    current_entity_str.clear()
                    current_entity_cat = ""

                current_entity_id.append(i)
                current_entity_str.append(tok)
                current_entity_cat = current_cat

        elif label.startswith("B"):

            current_cat = label.split("-")[1]

            if previous_tag in ["B", "I"]:

                entities[tuple(current_entity_id) + tuple([current_entity_cat])] = " ".join(current_entity_str)

                current_entity_id.clear()
                current_entity_str.clear()
                current_entity_cat = ""

                current_entity_id.append(i)
                current_entity_str.append(tok)
                current_entity_cat = current_cat

                previous_tag = "B"
                previous_cat = current_cat

            else:

                current_entity_id.append(i)
                current_entity_str.append(tok)
                current_entity_cat = current_cat

                previous_tag = "B"
                previous_cat = current_cat

        elif label.startswith("O"):

            if previous_tag in ["B", "I"]:

                entities[tuple(current_entity_id) + tuple([current_entity_cat])] = " ".join(current_entity_str)

                current_entity_id.clear()
                current_entity_str.clear()
                current_entity_cat = ""

                previous_tag = "O"
                previous_cat = ""

    if len(current_entity_id) > 0:
        entities[tuple(current_entity_id) + tuple([current_entity_cat])] = " ".join(current_entity_str)

        current_entity_id.clear()
        current_entity_str.clear()
        current_entity_cat = ""

    return entities
