__author__ = 'paters01'

from bson.objectid import ObjectId
from pymongo import MongoClient
import collections


def create_subject_aggregations(subject_aggregations, db_connection):
    subjects_collection = db_connection.RTO_20150107.subjects
    subject_count = subjects_collection.count()
    print("Processing %d subjects" % (subject_count))

    # iterate through subjects and create an aggregation object for each and initialise it
    #{"_id": ObjectId("5425947c69736d3813000000")}
    processed_count = 0
    for subject in subjects_collection.find().limit(1000000):
        subject_id =  subject["_id"]
        # print subject_id
        new_subject_aggregation = collections.OrderedDict()
        # add to dictionary
        subject_aggregations[subject_id] = new_subject_aggregation
        # initialise aggregation
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
    classifications_collection = db_connection.RTO_20150107.classifications
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
    subject_aggregations_db_collection = db_connection.RTO_20150107.subject_aggregations
    # clear existing data
    subject_aggregations_db_collection.remove()
    for subject_aggregation in subject_aggregations.values():
        subject_aggregations_db_collection.insert(subject_aggregation)


def load_aggregated_data(db_connection):
    aggregations = dict()
    cursor = db_connection.RTO_20150107.subject_aggregations.find().limit(10)
    for item in cursor:
        aggregations[item["subject_id"]] = collections.OrderedDict(item)
    return aggregations


def calculate_aggregation_medians(subject_aggregations):
    for subject_aggregation in subject_aggregations.values():
        stained_na = subject_aggregation["stained_na"]
        stained_none = subject_aggregation["stained_none"]
        stained_1_25 = subject_aggregation["stained_1_25"]
        stained_25_50 = subject_aggregation["stained_25_50"]
        stained_50_75 = subject_aggregation["stained_50_75"]
        stained_75_95 = subject_aggregation["stained_75_95"]
        stained_95_100 = subject_aggregation["stained_95_100"]

        total_count = stained_none + stained_1_25 + stained_25_50 + stained_50_75 + stained_75_95 + stained_95_100
        target_count = total_count / 2

        stained_median = ""
        running_total = stained_none
        if running_total > target_count:
            stained_median = "stained_none"

        running_total += stained_1_25
        if running_total > target_count:
            stained_median = "stained_1_25"

        running_total += stained_25_50
        if running_total > target_count:
            stained_median = "stained_25_50"

        running_total += stained_50_75
        if running_total > target_count:
            stained_median = "stained_50_75"

        running_total += stained_75_95
        if running_total > target_count:
            stained_median = "stained_75_95"
        else:
            stained_median = "stained_95_100"

        subject_aggregation["stained_median"] = stained_median



        bright_na = subject_aggregation["bright_na"]
        bright_weak = subject_aggregation["bright_weak"]
        bright_medium = subject_aggregation["bright_medium"]
        bright_strong = subject_aggregation["bright_strong"]

# main program

# connect to mongo subjects data
db_connection = MongoClient("localhost", 27017)
# create empty dictionary
# subject_aggregations = dict()
#
# create_subject_aggregations(subject_aggregations, db_connection)
# import_classification_data(subject_aggregations, db_connection)
# save_aggregated_data(subject_aggregations, db_connection)

subject_aggregations = load_aggregated_data(db_connection)

print(type(subject_aggregations))

db_connection.close()

