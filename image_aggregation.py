__author__ = 'paters01'

from pymongo import MongoClient
import collections


def create_subject_aggregations(subject_aggregations, db_connection):
    subjects_collection = db_connection.RTO_15092015.subjects
    subject_count = subjects_collection.count()
    print("Processing %d subjects" % (subject_count))

    # iterate through subjects and create an aggregation object for each and initialise it
    processed_count = 0
    # for subject in subjects_collection.find({"classification_count": {"$gt" : 0}}).limit(1000000):
    for subject in subjects_collection.find({"metadata.stain_type": "TEST MRE11", "classification_count": {"$gt" : 0}}).limit(1000000):
        subject_id =  subject["_id"]
        # print subject_id
        new_subject_aggregation = collections.OrderedDict()
        # add to dictionary
        subject_aggregations[subject_id] = new_subject_aggregation
        # initialise aggregation with subject data from database
        new_subject_aggregation["subject_id"] = subject_id
        new_subject_aggregation["state"] = subject["state"]
        if new_subject_aggregation["state"] != "inactive":
            new_subject_aggregation["activated_at"] = subject["activated_at"]
        new_subject_aggregation["classification_count"] = subject["classification_count"]
        new_subject_aggregation["created_at"] = subject["created_at"]
        new_subject_aggregation["group_id"] = subject["group_id"]
        new_subject_aggregation["group_zooniverse_id"] = subject["group"]["zooniverse_id"]
        new_subject_aggregation["group_name"] = subject["group"]["name"]
        new_subject_aggregation["location_standard"] = subject["location"]["standard"]
        new_subject_aggregation["location_thumbnail"] = subject["location"]["thumbnail"]
        if "metadata" in subject:
            metadata = subject["metadata"]
            new_subject_aggregation["collection"] = metadata["collection"]
            new_subject_aggregation["id_no"] = metadata["id_no"]
            new_subject_aggregation["index"] = metadata["index"]
            new_subject_aggregation["orig_directory"] = metadata["orig_directory"]
            new_subject_aggregation["orig_file_name"] = metadata["orig_file_name"]
            new_subject_aggregation["stain_type"] = metadata["stain_type"]
        else:
            pass
        new_subject_aggregation["project_id"] = subject["project_id"]
        new_subject_aggregation["random"] = subject["random"]
        new_subject_aggregation["updated_at"] = subject["updated_at"]
        new_subject_aggregation["workflow_id"] = subject["workflow_ids"][0]
        new_subject_aggregation["zooniverse_id"] = subject["zooniverse_id"]

        new_subject_aggregation["cancer_yes"] = 0
        new_subject_aggregation["cancer_no"] = 0
        new_subject_aggregation["stained_na"] = 0
        new_subject_aggregation["stained_none"] = 0
        new_subject_aggregation["stained_1_25"] = 0
        new_subject_aggregation["stained_25_50"] = 0
        new_subject_aggregation["stained_50_75"] = 0
        new_subject_aggregation["stained_75_95"] = 0
        new_subject_aggregation["stained_95_100"] = 0
        new_subject_aggregation["bright_na"] = 0
        new_subject_aggregation["bright_weak"] = 0
        new_subject_aggregation["bright_medium"] = 0
        new_subject_aggregation["bright_strong"] = 0

        processed_count += 1
        if processed_count % 1000 == 0:
            print("processed %d" % processed_count)
    print("processed %d" % processed_count)


def import_classification_data(subject_aggregations, db_connection):
    classifications_collection = db_connection.RTO_15092015.classifications
    classifications_count = classifications_collection.count()
    print("Processing %d classifications" % (classifications_count))

    processed_count = 0

    # for each row in raw image data
    for classification in classifications_collection.find().limit(10000000):
        # get subject id
        subject_id =  classification["subject_ids"][0]
        # print(subject_id)

        if subject_id not in subject_aggregations:
            print("subject %s not found for classification %s" % (subject_id, classification["_id"]))
        else:
            # get subject from dictionary
            subject_aggregation = subject_aggregations[subject_id]

            # process row to increment counts in ImageAggregation object
            if "annotations" in classification:  # annotations holds answers to questions
                annotations = classification["annotations"]
                # cancer yes/no questions
                cancer_answer = annotations[0]["a-1"]
                if cancer_answer == "1":
                    subject_aggregation["cancer_yes"] += 1
                elif cancer_answer == "2":
                    subject_aggregation["cancer_no"] += 1
                else:
                    print("unexpected cancer value %s" % cancer_answer)
                # percentage stained question
                stained_answer = annotations[1]["a-2"]
                if stained_answer == "0":
                    subject_aggregation["stained_na"] += 1
                elif stained_answer == "1":
                    subject_aggregation["stained_none"] += 1
                elif stained_answer == "2":
                    subject_aggregation["stained_1_25"] += 1
                elif stained_answer == "3":
                    subject_aggregation["stained_25_50"] += 1
                elif stained_answer == "4":
                    subject_aggregation["stained_50_75"] += 1
                elif stained_answer == "5":
                    subject_aggregation["stained_75_95"] += 1
                elif stained_answer == "6":
                    subject_aggregation["stained_95_100"] += 1
                else:
                    print("unexpected stain value %s" % stained_answer)
                # brightness question
                bright_answer = annotations[2]["a-3"]
                if bright_answer == "0":
                    subject_aggregation["bright_na"] += 1
                elif bright_answer == "1":
                    subject_aggregation["bright_weak"] += 1
                elif bright_answer == "2":
                    subject_aggregation["bright_medium"] += 1
                elif bright_answer == "3":
                    subject_aggregation["bright_strong"] += 1
                else:
                    print("unexpected bright value %s" % bright_answer)

        processed_count += 1
        if processed_count % 10000 == 0:
            print("processed %d" % processed_count)
    print("processed %d" % processed_count)


def save_aggregated_data(subject_aggregations, db_connection):
    subject_aggregations_db_collection = db_connection.RTO_15092015.subject_aggregations
    # clear existing data
    subject_aggregations_db_collection.remove()
    for subject_aggregation in subject_aggregations.values():
        subject_aggregations_db_collection.insert(subject_aggregation)


def save_aggregated_data_m(subject_aggregations, db_connection):
    subject_aggregations_db_collection = db_connection.RTO_15092015.subject_aggregations_m
    # clear existing data
    subject_aggregations_db_collection.remove()
    for subject_aggregation in subject_aggregations.values():
        subject_aggregations_db_collection.insert(subject_aggregation)


def load_aggregated_data(db_connection):
    aggregations = dict()
    cursor = db_connection.RTO_15092015.subject_aggregations.find().limit(1000000)
    for item in cursor:
        aggregations[item["subject_id"]] = collections.OrderedDict(item)
    return aggregations


def calculate_median(subject_or_core_aggregation):
    if "cancer_yes" in subject_or_core_aggregation:  # this is only present for subjects, NOT cores
        cancer_yes = subject_or_core_aggregation["cancer_yes"]
        cancer_no = subject_or_core_aggregation["cancer_no"]
        cancer_present = (cancer_yes >= cancer_no)
        subject_or_core_aggregation["cancer_present"] = cancer_present

        subject_or_core_aggregation["stained_median"] = ""
        subject_or_core_aggregation["bright_mode"] = ""

        if not cancer_present:
            return

    stained_none = subject_or_core_aggregation["stained_none"]
    stained_1_25 = subject_or_core_aggregation["stained_1_25"]
    stained_25_50 = subject_or_core_aggregation["stained_25_50"]
    stained_50_75 = subject_or_core_aggregation["stained_50_75"]
    stained_75_95 = subject_or_core_aggregation["stained_75_95"]
    stained_95_100 = subject_or_core_aggregation["stained_95_100"]

    total_count = stained_none + stained_1_25 + stained_25_50 + stained_50_75 + stained_75_95 + stained_95_100
    target_count = (total_count + 1) / 2

    stained_median = ""
    if target_count > 0:
        running_total = stained_none
        if running_total >= target_count:
            stained_median = "stained_none"
        else:
            running_total += stained_1_25
            if running_total >= target_count:
                stained_median = "stained_1_25"
            else:
                running_total += stained_25_50
                if running_total >= target_count:
                    stained_median = "stained_25_50"
                else:
                    running_total += stained_50_75
                    if running_total >= target_count:
                        stained_median = "stained_50_75"
                    else:
                        running_total += stained_75_95
                        if running_total >= target_count:
                            stained_median = "stained_75_95"
                        else:
                            stained_median = "stained_95_100"
    subject_or_core_aggregation["stained_median"] = stained_median

    bright_weak = subject_or_core_aggregation["bright_weak"]
    bright_medium = subject_or_core_aggregation["bright_medium"]
    bright_strong = subject_or_core_aggregation["bright_strong"]

    #### Code for median intensity

    # total_count = bright_weak + bright_medium + bright_strong
    # target_count = (total_count + 1) / 2

    # bright_median = ""
    # if target_count > 0:
    #     running_total = bright_weak
    #     if running_total >= target_count:
    #         bright_median = "bright_weak"
    #     else:
    #         running_total += bright_medium
    #         if running_total >= target_count:
    #             bright_median = "bright_medium"
    #         else:
    #             bright_median = "bright_strong"
    # subject_or_core_aggregation["bright_median"] = bright_median

    #### Code for mode intensity - rules need adjusted depending on the biomarker

    bright_mode = ""
    if bright_weak >= bright_medium and bright_weak > bright_strong:
        bright_mode = "bright_weak"
    if bright_medium > bright_weak and bright_medium >= bright_strong:
        bright_mode = "bright_medium"
    if bright_strong > bright_weak and bright_strong > bright_medium:
        bright_mode = "bright_strong"
    subject_or_core_aggregation["bright_mode"] = bright_mode

def calculate_mean(core_aggregation):
    mean_proportion = ""
    stained_none = core_aggregation["stained_none"]
    stained_1_25 = core_aggregation["stained_1_25"]
    stained_25_50 = core_aggregation["stained_25_50"]
    stained_50_75 = core_aggregation["stained_50_75"]
    stained_75_95 = core_aggregation["stained_75_95"]
    stained_95_100 = core_aggregation["stained_95_100"]

    mean_numerator_proportion = stained_none * 0 + stained_1_25 * 1 + stained_25_50 * 2 + stained_50_75 * 3 + stained_75_95 * 4 + stained_95_100 * 5
    mean_denominator_proportion = stained_none + stained_1_25 + stained_25_50 + stained_50_75 + stained_75_95 + stained_95_100

    if mean_denominator_proportion:
        mean_proportion = float(mean_numerator_proportion) / mean_denominator_proportion
    else:
        mean_proportion = 0

    core_aggregation["proportion_mean"] = mean_proportion

    mean_intensity = ""
    bright_weak = core_aggregation["bright_weak"]
    bright_medium = core_aggregation["bright_medium"]
    bright_strong = core_aggregation["bright_strong"]

    mean_numerator_intensity = bright_weak * 1 + bright_medium * 2 + bright_strong * 3
    mean_denominator_intensity = bright_weak + bright_medium + bright_strong

    if mean_denominator_intensity:
        mean_intensity = float(mean_numerator_intensity)/mean_denominator_intensity
    else:
        mean_intensity = 0

    core_aggregation["intensity_mean"] = mean_intensity

    score = mean_proportion + mean_intensity
    core_aggregation["pseudo_score"] = round(score,0)

    high =""
    if score < 7:
            high = "no"
    elif score >= 7:
        high = "yes"
    core_aggregation["high"] = high

def calculate_aggregation_medians(subject_aggregations):
    for subject_aggregation in subject_aggregations.values():
        calculate_median(subject_aggregation)


def save_aggregated_core_data(core_aggregations, db_connection):
    core_aggregations_db_collection = db_connection.RTO_15092015.core_aggregations
    # clear existing data
    core_aggregations_db_collection.remove()
    for core_aggregation in core_aggregations.values():
        core_aggregations_db_collection.insert(core_aggregation)


def create_core_aggregations(core_aggregations, subject_aggregations):
    count = 0
    for subject_aggregation in subject_aggregations.values():
        core_id = subject_aggregation["id_no"]  # note, this may not exist if subjects without classifications are included
        count += 1
        print(count, ":", core_id)
        if core_id in core_aggregations.keys():
            core_aggregation = core_aggregations[core_id]
        else:
            # doesn't exist so create and initialise it
            core_aggregation = collections.OrderedDict()
            core_aggregation["core_id"] = core_id
            core_aggregation["stain_type"] = subject_aggregation["stain_type"]
            core_aggregation["no_cancer_images"] = 0
            core_aggregation["stained_none"] = 0
            core_aggregation["stained_1_25"] = 0
            core_aggregation["stained_25_50"] = 0
            core_aggregation["stained_50_75"] = 0
            core_aggregation["stained_75_95"] = 0
            core_aggregation["stained_95_100"] = 0
            core_aggregation["bright_weak"] = 0
            core_aggregation["bright_medium"] = 0
            core_aggregation["bright_strong"] = 0
            core_aggregations[core_id] = core_aggregation

        # add in counts from subject aggregation
        subject_stained_median = subject_aggregation["stained_median"]
        if subject_stained_median != "":
            core_aggregation[subject_stained_median] += 1

        subject_bright_mode = subject_aggregation["bright_mode"]
        if subject_bright_mode != "":
            core_aggregation[subject_bright_mode] += 1

        if subject_aggregation["cancer_present"]:
             core_aggregation["no_cancer_images"]+=1


def calculate_core_medians(core_aggregations):
    for core_aggregation in core_aggregations.values():
        calculate_median(core_aggregation)


def calculate_core_means(core_aggregations):
    for core_aggregation in core_aggregations.values():
        calculate_mean(core_aggregation)

# main program

# connect to mongo subjects data
db_connection = MongoClient("localhost", 27017)
# create empty dictionary
subject_aggregations = dict()

create_subject_aggregations(subject_aggregations, db_connection)
import_classification_data(subject_aggregations, db_connection)
save_aggregated_data(subject_aggregations, db_connection)

subject_aggregations = load_aggregated_data(db_connection)
calculate_aggregation_medians(subject_aggregations)
save_aggregated_data_m(subject_aggregations, db_connection)

core_aggregations = dict()
create_core_aggregations(core_aggregations, subject_aggregations)
calculate_core_means(core_aggregations)
save_aggregated_core_data(core_aggregations, db_connection)


db_connection.close()

