__author__ = 'mcquil02'

import csv
from pprint import pprint


def import_gold_standard_data(gold_standard_dict):
    with open('/Users/mcquil02/Documents/RTO/MRE11/core_gold_standard.csv','rU') as f:
        reader=csv.reader(f)
        for row in reader:
            core_id = row[0]
            gold_standard_dict[core_id] = {
                "proportion": row[1],
                "intensity": row[2]
            }
    return gold_standard_dict


def calculate_pseudo_score(gold_standard_dict):
    gold_standard_dict = import_gold_standard_data(gold_standard_dict)
    for core_id, values in gold_standard_dict.items():
        if int(values["proportion"]) == 0:
            dis_prop = 0
        elif int(values["proportion"]) >= 1 and int(values["proportion"]) < 25:
            dis_prop = 1
        elif int(values["proportion"]) >= 25 and int(values["proportion"]) < 50:
            dis_prop = 2
        elif int(values["proportion"]) >= 50 and int(values["proportion"]) < 75:
            dis_prop = 3
        elif int(values["proportion"]) >= 75 and int(values["proportion"]) < 95:
            dis_prop = 4
        elif int(values["proportion"]) >= 95 and int(values["proportion"]) <= 100:
            dis_prop = 5
        score =  dis_prop + int(values["intensity"])
        if score < 7:
            high = "no"
        elif score >= 7:
            high = "yes"
        gold_standard_dict[core_id] = {
            "proportion": values["proportion"],
            "intensity": values["intensity"],
            "discrete_prop": dis_prop,
            "pseudo_score": score,
            "high": high
        }
    return gold_standard_dict



gold_standard_dict = dict()

import_gold_standard_data(gold_standard_dict)
calculate_pseudo_score(gold_standard_dict)


