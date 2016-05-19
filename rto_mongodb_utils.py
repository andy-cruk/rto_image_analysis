''' This set of functions lets you manipulate the mongoDB database
In case it's ever useful, this is how you can join the subjects and classification database. Only works
in this direction because I cannot work out how to index into an array for subjects.id
subjectsCollection.aggregate([{"$lookup":{"from":"classifications","localField":"_id","foreignField":"subjects.id","as":'classifications'}}]).next()
'''

import user_aggregation
import os
import pandas as pd
import pymongo
import re


# set the name of the database on your local host to connect to
currentDB = 'RTO_20160425'

def add_indices(classifCollection,subjectsCollection):
    """ adds indices that will speed up various functions in user_aggregation
    :param classifCollection:
    :param subjectsCollection:
    :return:
    """
    classifCollection.create_index("_id")
    classifCollection.create_index("subject_ids")
    classifCollection.create_index("created_at")
    classifCollection.create_index("stain_type_lower")
    classifCollection.create_index("id_no")
    classifCollection.create_index("has_expert")
    classifCollection.create_index("cancer")
    classifCollection.create_index("proportion")
    classifCollection.create_index("intensity")

    subjectsCollection.create_index("_id")
    subjectsCollection.create_index("metadata.stain_type_lower")
    subjectsCollection.create_index("classification_count")
    subjectsCollection.create_index("hasExpert")
    subjectsCollection.create_index("metadata.id_no")
    subjectsCollection.create_index([("metadata.stain_type_lower",pymongo.ASCENDING),("classification_count",pymongo.ASCENDING)])
    subjectsCollection.create_index([("metadata.stain_type_lower",pymongo.ASCENDING),("classification_count",pymongo.ASCENDING),("hasExpert",pymongo.ASCENDING)])


def add_lowercase_metadata_staintype(db):
    """ Works on subject database. Not all
    :return: returns nothing
    """
    print "adding lowercase metadata stain type"
    # Ki67 has the wrong metadata.stain_type
    db.update_many({"group.name":"TMAs_Ki67"},{"$set":{"metadata.stain_type":"Ki67"}})
    # NBS1 has the wrong metadata.stain_type
    db.update_many({"group.name":"TMAs_NBS1"},{"$set":{"metadata.stain_type":"NBS1"}})
    # 53BP1 has the wrong metadata.stain_type and group.name (should be a 1, not an i at the end)
    db.update_many({"group.name":"53BPI"},{"$set":{"group.name":"53BP1","metadata.stain_type":"53BP1"}})
    # the different MRE11 variants need to update their stain_type too
    db.update_many({"group.name": "MRE11_new_TMAs"}, {"$set": {"metadata.stain_type": "MRE11new"}})
    db.update_many({"group.name": "MRE11c"}, {"$set": {"metadata.stain_type": "MRE11c"}})
    # check all entries have metadata.stain_type
    assert(db.find({"metadata.stain_type":{"$exists": False}}).count() == 0)
    # add lowercase version
    for d in db.find({}):
        db.update({"_id":d["_id"]},{"$set":{"metadata.stain_type_lower":d["metadata"]["stain_type"].lower()}})


def add_whether_subject_is_part_of_core_with_expert_data(db):
    """
    Uses sj_in_expert_core to add a boolean to each entry regarding whether or not its parent core has an expert score
    Only updates stains that have GS excel sheet
    :param db: pymongo handle to a collection, in this case the subjects collection
    :return: nothing
    """
    print 'adding for each segment whether it is part of an expert core'
    # loop over each stain
    for stain in user_aggregation.stains:
        print "adding expert tags for stain "+stain
        # get all entries in mongodb for this stain, returning only _id and id_no
        subjectCursor = db.find(filter={'metadata.stain_type_lower':stain},no_cursor_timeout=True)
        # loop over each and check whether it has expert data
        for sj in subjectCursor:
            hasExpert = user_aggregation.sj_in_expert_core(sj["metadata"]["id_no"], coresGS=user_aggregation.coresGS)
            db.update_one({'_id': sj['_id']}, {'$set': {'hasExpert': hasExpert}}, upsert=False)


def sanity_checks_on_db(db):
    """ Test whether format of critical data in subjects collection is correct
    Only tests those entries belonging to a dataset with GS
    Will not detect entries where both metadata.stain_type is erroneous AND group.name is erroneous
    """
    print "running sanity checks on database"
    for stain in user_aggregation.stains:
        print 'checking '+stain
        # check if any records exist for this one
        if db.find({'metadata.stain_type_lower':stain}).count() <= 0:
            continue
        # check both stain_type_lower and group.name are populated in each document
        assert(db.find({'metadata.stain_type_lower': stain}).count() == db.find({'group.name': {"$exists": True},'metadata.stain_type_lower': stain}).count())
        # check id_no exists in each document
        assert(db.find({'metadata.stain_type_lower': stain}).count() == db.find({'metadata.stain_type_lower': stain, 'metadata.id_no':{"$exists": True}}).count())
        # check id_no has consistent format (same length).
        # if not, run
        # db.getCollection('subjects').distinct("metadata.id_no",{'metadata.stain_type_lower':'ck20'})
        # and look for inconsistent naming; add correction to correct_known_mistakes(db)
        id_no = list(db.find({'metadata.stain_type_lower':stain},{'_id':False,'metadata.id_no':True}))
        lengths = [len(x['metadata']['id_no']) for x in id_no]
        if len(set(lengths)) > 1:
            for id in id_no:
                print id['metadata']['id_no'].replace(' ','.')
            raise Exception('found different lengths of id_no')


def correct_known_mistakes(db, classifCollection):
    """ This corrects a number of mistakes that have to be corrected through hardcoding

    """
    print "correcting known mistakes"
    # remove some unwanted data in subjects database
    db.delete_many({"group.name": {"$in": ["squamous_lung_cd8","squamous_lung_pdl1","er","adeno_lung_cd8","adeno_lung_pdl1","tonsil_egfr"]}})

    # wherever metadata.orig_file_name is used, correct
    db.update_many({"metadata.orig_file_name":{"$exists":True}},{"$rename":{"metadata.orig_file_name":"metadata.orig_filename"}})

    # correct a bunch of id_no that were stored incorrectly
    db.update_many({'group.name':'test_mre11'},{"$set":{"group.name":'test mre11'}})
    db.update_many({"metadata.id_no":"7913 ME11"},{"$set":{"metadata.id_no":"7913 MRE11"}})
    db.update_many({"metadata.id_no":"7838MRE11"},{"$set":{"metadata.id_no":"7838 MRE11"}})
    db.update_many({"metadata.id_no":"7972 MRE1"},{"$set":{"metadata.id_no":"7972 MRE11"}})
    db.update_many({"metadata.id_no":"8302 MRE11_"},{"$set":{"metadata.id_no":"8302 MRE11"}})
    db.update_many({"metadata.id_no":"8079 p21_"},{"$set":{"metadata.id_no":"8079 p21"}})
    db.update_many({"metadata.id_no":"7987  p21"},{"$set":{"metadata.id_no":"7987 p21"}})
    db.update_many({"metadata.id_no":"8583  p21"},{"$set":{"metadata.id_no":"8583 p21"}})
    db.update_many({"metadata.id_no":"8772 MRE11 new_"},{"$set":{"metadata.id_no":"8772 MRE11 new"}})
    db.update_many({"metadata.id_no":"7812 RAD50 "},{"$set":{"metadata.id_no":"7812 RAD50"}})
    db.update_many({"metadata.id_no": "8736 ARD50"}, {"$set": {"metadata.id_no": "8736 RAD50"}})
    db.update_many({"metadata.id_no": "7753 ARD50"}, {"$set": {"metadata.id_no": "7753 RAD50"}})
    db.update_many({"metadata.id_no": "7853 ARD50"}, {"$set": {"metadata.id_no": "7853 RAD50"}})
    db.update_many({"metadata.id_no": "8221 ARD50"}, {"$set": {"metadata.id_no": "8221 RAD50"}})
    db.update_many({"metadata.id_no": "8445 RAD59"}, {"$set": {"metadata.id_no": "8445 RAD50"}})
    db.update_many({"metadata.id_no": "7812 RAD50 "}, {"$set": {"metadata.id_no": "7812 RAD50"}})

    # remove a bunch of tutorial images
    db.delete_many({"metadata.id_no":{"$in":[
            "7635 MRE11 new 1+",
            "7676 MRE11 new 1+",
            "7838 MRE11 new 3+",
            "7845 MRE11 new 3+",
            "7887 MRE11 new 2+",
            "7909 MRE11 new 0+_",
            "7920 MRE11 new 1+",
            "7957 MRE11 new 2+",
            "7996 MRE11 new use for percentage",
            "7998 MRE11 new use for percentage",
            "8616 MRE11 new stroma",
            "8629 MRE11 new use for percentage",
            "8663 MRE11 new lymphocytes",
            "8680 MRE11 new blood vessels",
            "8719 MRE11 new stroma",
    ]}})
    # change id_no for 'test mre11'
    cursor = db.find({"metadata.stain_type_lower": 'test mre11'})
    for doc in cursor:
        # get the id_no
        id_no = doc["metadata"]["id_no"]
        if '_' in id_no: # only run this if it hasn't been adjusted yet, i.e. it has to have an underscore
            # get the TMA number, which is the first element after first underscore
            n = id_no.split('_')[1]
            # write id_no
            db.update_one({"_id": doc["_id"]}, {"$set": {"metadata.id_no": n+' test mre11'}})
    # MRE11new has id_no which has a space between MRE11 and new.
    db.update_many({"metadata.id_no":"7909 MRE11 new_"},{"$set":{"metadata.id_no":"7909 MRE11 new"}})
    cursor = db.find({"metadata.stain_type_lower": 'mre11new'})
    for doc in cursor:
        # get the id_no
        id_no = doc["metadata"]["id_no"]
        if "MRE11 new" in id_no: # only run if hasn't been adjusted yet
            # get the TMA number, which is the first element after first underscore
            n = id_no.split()[0]
            # write id_no
            db.update_one({"_id": doc["_id"]}, {"$set": {"metadata.id_no": n+' MRE11new'}})

    # a bunch of rad50 and every sample added after that doesn't have metadata.id_no. So has to be derived from the filename, which can be stored in 2 different fields
    # Also see db.getCollection('subjects').distinct('metadata.orig_filename',{"metadata.stain_type_lower":"rad50","metadata.id_no":{"$exists":false}})
    # applied in mongodb
    ###### all stains that have orig_filename with format ^\w*/\w*/dddd_stain/\w*\.jpg
    stains = ['rad50', 'ck20', 'ck5']
    docs = db.find({"metadata.stain_type_lower": {"$in": stains}, "metadata.id_no": {"$exists": False}})
    coreNames = [x["metadata"]["orig_filename"][re.search("^\w*/[ a-zA-Z0-9_]*/",x["metadata"]["orig_filename"]).end(0): re.search("/\w*\..*$",x["metadata"]["orig_filename"]).start(0)].replace('_',' ') for x in docs.rewind()]
    for ix,doc in enumerate(docs.rewind()):
        db.update_one({'_id': doc['_id']}, {'$set': {'metadata.id_no': coreNames[ix], 'metadata.id_no_corrected': True}})

    ##### all stains that have orig_directory with format ^[ a-zA-Z0-9_]*/\w*/dddd_stain/\w*\.jpg. Note the [ a-zA-Z0-9_] which accounts for spaces too, which \w does not. Some filenames have spaces in them (!)
    # note also some filenames end in Thumbs.db, so are not jpg
    stains = ('tip60','53bp1')
    docs = db.find({"metadata.stain_type_lower": {"$in": stains}})
    coreNames = [x["metadata"]["orig_directory"][re.search("^[ a-zA-Z0-9_]*/\w*/",x["metadata"]["orig_directory"]).end(0):re.search("/\w*\..*$",x["metadata"]["orig_directory"]).start(0)].replace('_',' ') for x in docs.rewind()]
    for ix,doc in enumerate(docs.rewind()):
        db.update_one({'_id': doc['_id']},{'$set': {'metadata.id_no': coreNames[ix]}})



    # these have to be changed after adding id_no
    db.update_many({"metadata.id_no":"8280 CK5 "},{"$set":{"metadata.id_no":"8280 CK5"}})
    db.update_many({"metadata.id_no":"8632 Ki67 "},{"$set":{"metadata.id_no":"8632 Ki67"}})
    db.update_many({"metadata.id_no":"8640  Ki67"},{"$set":{"metadata.id_no":"8640 Ki67"}})


    # add some data to classifCollection to then be able to delete erroneous classifications
    add_info_to_each_classification(stain_and_core=True)

    # delete classifications where incompatible responses were recorded.
    # Will only look in GS stains, because otherwise hundreds of thousands of lung/CD8 stains are chucked out
    # cancer + proportion stain, but intensity = 0. Also see
    # db.getCollection('classifications').distinct('updated_at',{"annotations.a-2":{"$in":['2','3','4','5','6']},"annotations.a-3":'0'})
    dr = classifCollection.delete_many({"stain_type_lower": {"$in": user_aggregation.stains}, "annotations.a-1": '1', "annotations.a-2": {"$in": ['2', '3', '4', '5', '6']}, "annotations.a-3": '0'})
    print "deleted", dr.deleted_count, "classifications with proportion > 0, intensity = 0"


def add_info_to_each_classification(stain_and_core=False, hasExpert=False, annotations=False):
    # can add different pieces of information to classification database. This is because the order of different pieces
    # of information is different so important to be able to select what piece goes when
    print "adding core-level info to each classification, and/or adding answers as top-level fields"
    #### add the cleaned metadata.id_no and stain_type_lower to each classification that we have GS data for
    total_entries = classifCollection.find({}).count()
    count = 0. # initialise as float to make sure you get decimal points when dividing
    for cln in classifCollection.find({}):
        if (count % 20000) == 0:
            print('%.1f' % 100*(count/total_entries))
        count += 1

        # get metadata.id_no
        sj = subjectsCollection.find({"_id": cln["subject_ids"][0]})
        if sj.count() == 1:
            dat = sj.next()
            if stain_and_core:
                classifCollection.update_one({"_id": cln["_id"]}, {"$set": {
                    "id_no": dat["metadata"]["id_no"],
                    "stain_type_lower": dat["metadata"]["stain_type_lower"]
                }})
            if hasExpert and ('hasExpert' in dat):
                classifCollection.update_one({"_id": cln["_id"]}, {"$set": {"hasExpert": dat["hasExpert"]}})
            else:
                classifCollection.update_one({"_id": cln["_id"]}, {"$set": {"hasExpert": False}})
        else:
            if stain_and_core:
                classifCollection.update_one({"_id": cln["_id"]},{"$set": {
                    "id_no": None,
                    "stain_type_lower": None
                }})
            if hasExpert:
                classifCollection.update_one({"_id": cln["_id"]}, {"$set": {"hasExpert": False}})
        if annotations:
            # now take out the annotations and put them in their own top-level field
            classifCollection.update_one({"_id": cln["_id"]}, {"$set": {
                "cancer": cln["annotations"][0]['a-1'],
                "proportion": cln["annotations"][1]['a-2'],
                "intensity": cln["annotations"][2]['a-3']
            }})


if __name__ == "__main__":
    subjectsCollection, classifCollection, dbConnection = user_aggregation.pymongo_connection_open()
    add_lowercase_metadata_staintype(subjectsCollection)
    correct_known_mistakes(subjectsCollection,classifCollection)
    add_whether_subject_is_part_of_core_with_expert_data(subjectsCollection)
    add_info_to_each_classification(hasExpert=True, annotations=True, stain_and_core=False)
    add_indices(classifCollection,subjectsCollection)
    sanity_checks_on_db(subjectsCollection)
    print "done with rto_mongodb_utils.py"
