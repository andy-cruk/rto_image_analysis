__author__ = 'mcquil02'

from pymongo import MongoClient
from format_gold_standard import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_aggregated_citizen_data(db_connection):
    aggregations = dict()
    cursor = db_connection.RTO_15092015.core_aggregations.find().limit(1000000)
    for item in cursor:
        full_core_id = item["core_id"]
        # You will need to change the split criteria depending on the format of id_no
        split_core_id = full_core_id.split("_")
        core_id = split_core_id[1]
        aggregations[core_id] = {
            "proportion_mean": item["proportion_mean"],
            "intensity_mean": item["intensity_mean"],
            "pseudo_score": item["pseudo_score"],
            "high": item["high"]
        }

    return aggregations

def create_expert_citizen_consensus_agreement_pre_processing_dict():
    expert_dict = gold_standard_dict
    citizen_scientist_dict = load_aggregated_citizen_data(db_connection)
    comparisons_dict = dict()
    for expert_core_id, expert_values in expert_dict.items():
        for citizen_core_id, citizen_values in citizen_scientist_dict.items():
            if expert_core_id == citizen_core_id:
                if expert_core_id not in comparisons_dict.keys():
                    comparisons_dict[expert_core_id] = {
                        "citizen_core_id": citizen_core_id,
                        "citizen_answer": citizen_values['high'],
                        "expert_answer": expert_values['high'],
                        "citizen_pseudo_score": citizen_values['pseudo_score'],
                        "expert_pseudo_score": expert_values['pseudo_score'],
                        "false_positive": bool(citizen_values['high'] == "yes" and expert_values['high'] == "no"),
                        "true_positive": bool(citizen_values['high'] == "yes" and expert_values['high'] == "yes"),
                        "false_negative": bool(citizen_values['high'] == "no" and expert_values['high'] == "yes"),
                        "true_negative": bool(citizen_values['high'] == "no" and expert_values['high'] == "no")
                    }
    return comparisons_dict

def calculate_statistical_measures():
    expert_citizen_dict = create_expert_citizen_consensus_agreement_pre_processing_dict()
    no_classifications = len(expert_citizen_dict)
    accuracy_numerator = 0
    no_false_positives = 0
    no_true_positives = 0
    no_false_negatives = 0
    no_true_negatives = 0
    expert_high_value = 0
    citizen_high_value = 0

    for core_id, dict_of_values in expert_citizen_dict.items():
        if dict_of_values['citizen_answer'] == dict_of_values['expert_answer']:
            accuracy_numerator += 1
        if dict_of_values['false_positive']:
            no_false_positives += 1
        if dict_of_values['true_positive']:
            no_true_positives += 1
        if dict_of_values['false_negative']:
            no_false_negatives += 1
        if dict_of_values['true_negative']:
            no_true_negatives += 1
        if dict_of_values['expert_answer'] == "yes":
            expert_high_value += 1
        if dict_of_values['citizen_answer'] == "yes":
            citizen_high_value += 1

    proportion_expert_high_value = float(expert_high_value)/no_classifications
    proportion_citizen_high_value = float(citizen_high_value)/no_classifications
    accuracy = float(no_true_positives + no_true_negatives)/no_classifications
    probability_both_say_high_randomly = proportion_expert_high_value * proportion_citizen_high_value
    probability_both_say_low_randomly = (1 - proportion_expert_high_value) * (1 - proportion_citizen_high_value)
    overall_probability_random_agreement_excluding_tied = probability_both_say_high_randomly + probability_both_say_low_randomly

    accuracy = round(float(accuracy_numerator)/no_classifications,2)
    sensitivity = round(float(no_true_positives)/(no_true_positives + no_false_negatives),2)
    specificity = round(float(no_true_negatives)/(no_true_negatives + no_false_positives),2)
    precision = round(float(no_true_positives)/(no_true_positives + no_false_positives),2)
    f_measure = round(float(2 * no_true_positives)/((2 * no_true_positives) + no_false_negatives + no_false_negatives),2)
    kappa = round(float(accuracy - overall_probability_random_agreement_excluding_tied)/(1 - overall_probability_random_agreement_excluding_tied),2)

    print(accuracy_numerator, no_false_positives, no_true_positives, no_false_negatives, no_true_negatives, expert_high_value, citizen_high_value)
    print("Accuracy:", accuracy, "Sensitivity:", sensitivity, "Specificity:", specificity, "Precision:", precision, "F-measure:", f_measure, "Kappa:", kappa)

def plot_citizen_expert_scores():
    expert_citizen_dict = create_expert_citizen_consensus_agreement_pre_processing_dict()
    N = len(expert_citizen_dict)
    list_of_expert_scores = list()
    list_of_citizen_scores = list()

    for core_id, dict_of_values in expert_citizen_dict.items():
        expert_score = dict_of_values['expert_pseudo_score']
        list_of_expert_scores.append(expert_score)
        citizen_score = dict_of_values['citizen_pseudo_score']
        list_of_citizen_scores.append(citizen_score)

    expert_scores = np.array(list_of_expert_scores)
    citizen_scores = np.array(list_of_citizen_scores)

    spearman = stats.spearmanr(expert_scores, citizen_scores)
    print(spearman)

    colors = np.random.rand(N)
    plt.scatter(expert_scores,citizen_scores, c = colors)
    plt.title('Scatter plot of pseudo scores')
    plt.xlabel('Expert score')
    plt.ylabel('Citizen Scientist score')
    plt.show()


# main program

# connect to mongo subjects data
db_connection = MongoClient("localhost", 27017)

#### import and format the gold standard data
gold_standard_dict = dict()
import_gold_standard_data(gold_standard_dict)
calculate_pseudo_score(gold_standard_dict)

#### compare expert result with citizen scientist consensus
# calculate_statistical_measures()

#### Produce a scatter plot between the expert pseudo score and the citizen scientist consensus pseudo score
plot_citizen_expert_scores()

#### close the connection to mongodb
db_connection.close()