''' This set of functions lets you manipulate the mongoDB database
'''

import user_aggregation
import os
import pandas as pd
import pymongo
import re

# set the name of the database on your local host to connect to
currentDB = 'RTO_20160204'

def add_indices(classifCollection,subjectsCollection):
    """ adds indices that will speed up various functions in user_aggregation
    :param classifCollection:
    :param subjectsCollection:
    :return:
    """
    classifCollection.create_index("_id")
    classifCollection.create_index("subject_ids")
    classifCollection.create_index("created_at")

    subjectsCollection.create_index("_id")
    subjectsCollection.create_index("stain_type_lower")
    subjectsCollection.create_index("classification_count")
    subjectsCollection.create_index("hasExpert")
    subjectsCollection.create_index("metadata.id_no")
    subjectsCollection.create_index([("stain_type_lower",pymongo.ASCENDING),("classification_count",pymongo.ASCENDING)])
    subjectsCollection.create_index([("stain_type_lower",pymongo.ASCENDING),("classification_count",pymongo.ASCENDING),("hasExpert",pymongo.ASCENDING)])


def add_lowercase_metadata_staintype(db):
    """ Works on subject database. Not all
    :return: returns nothing
    """
    print "adding lowercase metadata stain type"
    # p53 doesn't have metadata.stain_type, add that
    db.update_many({"group.name":"p53"},{"$set":{"metadata.stain_type":"p53"}})
    # check all entries have metadata.stain_type
    assert(db.find({"metadata.stain_type":{"$exists":False}}).count() == 0)
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
    # find stains with GS data; will also return folders and other files, so make sure folder is clean.
    f = os.listdir('GS')
    stains = [x.lstrip('GS_').rstrip('.xlsx') for x in f]
    # loop over each stain
    for stain in stains:
        print "adding expert tags for stain "+stain
        # load GS data
        coresGS = pd.read_excel("GS\GS_"+stain+".xlsx")
        # get all entries in mongodb for this stain, returning only _id and id_no
        subjectCursor = db.find(filter={'metadata.stain_type_lower':stain},no_cursor_timeout=True)
        # loop over each and check whether it has expert data
        for sj in subjectCursor:
            hasExpert = user_aggregation.sj_in_expert_core(sj["metadata"]["id_no"],coresGS=coresGS)
            db.update_one({'_id':sj['_id']},{'$set':{'hasExpert':hasExpert}},upsert=False)


def sanity_checks_on_db(db):
    """ Test whether format of critical data in subjects collection is correct
    Only tests those entries belonging to a dataset with GS
    Will not detect entries where both metadata.stain_type is erroneous AND group.name is erroneous
    """
    print "running sanity checks on database"
    f = os.listdir('GS')
    stains = [x.lstrip('GS_').rstrip('.xlsx') for x in f]
    for stain in stains:
        print 'checking '+stain
        # check both stain_type_lower and group.name are populated in each document
        assert(db.find({'metadata.stain_type_lower':stain}).count() == db.find({'group.name':{"$exists":True},'metadata.stain_type_lower':stain}).count())
        # check id_no exists in each document
        assert(db.find({'metadata.stain_type_lower':stain}).count() == db.find({'metadata.stain_type_lower':stain,'metadata.id_no':{"$exists":True}}).count())
        # check id_no has consistent format (same length).
        # if not, run
        # db.getCollection('subjects').distinct("metadata.id_no",{'metadata.stain_type_lower':'ck20'})
        # and look for inconsistent naming; add correction to correct_known_mistakes(db)
        id_no = list(db.find({'metadata.stain_type_lower':stain},{'_id':False,'metadata.id_no':True}))
        lengths = [len(x['metadata']['id_no']) for x in id_no]
        assert(len(set(lengths)) <= 1)


def correct_known_mistakes(db, classifCollection):
    """ This corrects a number of mistakes that have to be corrected through hardcoding

    """
    print "correcting known mistakes"
    # remove some unwanted data
    db.delete_many({"group.name": {"$in": ["squamous_lung_cd8","squamous_lung_pdl1","er","adeno_lung_cd8","adeno_lung_pdl1","tonsil_egfr"]}})

    # delete classifications where people gave non-zero proportion but then zero intensity was recorded
    # classifCollection.delete_many({" HOW DO YOU INDEX NESTED ARRAY FIELDS

    raise Exception
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

    # a bunch of rad50 and every sample added after that doesn't have metadata.id_no.
    # Also see db.getCollection('subjects').distinct('metadata.orig_filename',{"metadata.stain_type_lower":"rad50","metadata.id_no":{"$exists":false}})
    # applied in mongodb

    ###### all stains that have orig_filename with format ^\w*/\w*/dddd_stain/\w*\.jpg
    stains = ['rad50', 'ck20', 'ck5']
    # if you don't include metadata.id_no exists:True, you will include RAD50 where metadata.orig_filename is different format, which will result in an error
    docs = db.find({"metadata.stain_type_lower":{"$in":stains},"metadata.id_no":{"$exists":False}})
    coreNames = [x["metadata"]["orig_filename"][re.search("^\w*/\w*/",x["metadata"]["orig_filename"]).end(0):re.search("/\w*\..*$",x["metadata"]["orig_filename"]).start(0)].replace('_',' ') for x in docs.rewind()]
    for ix,doc in enumerate(docs.rewind()):
        db.update_one({'_id':doc['_id']},{'$set':{'metadata.id_no':coreNames[ix],'metadata.id_no_corrected':True}})

    ##### all stains that have orig_directory with format ^[ a-zA-Z0-9_]*/\w*/dddd_stain/\w*\.jpg. Note the [ a-zA-Z0-9_] which accounts for spaces too, which \w does not. Some filenames have spaces in them (!)
    # note also some filenames end in Thumbs.db, so are not jep
    stains = ('tip60','53bpi')
    docs = db.find({"metadata.stain_type_lower":{"$in":stains}})
    coreNames = [x["metadata"]["orig_directory"][re.search("^[ a-zA-Z0-9_]*/\w*/",x["metadata"]["orig_directory"]).end(0):re.search("/\w*\..*$",x["metadata"]["orig_directory"]).start(0)].replace('_',' ') for x in docs.rewind()]
    for ix,doc in enumerate(docs.rewind()):
        db.update_one({'_id':doc['_id']},{'$set':{'metadata.id_no':coreNames[ix]}})


    # P53 doesn't have any metadata.id_no OR metadata.orig_filename. In fact, has nothing to see what core it belongs to. Being checked with Adam McMaster at Zooniverse now


if __name__ == "__main__":
    subjectsCollection, classifCollection, dbConnection = user_aggregation.pymongo_connection_open()
    # add_lowercase_metadata_staintype(subjectsCollection)
    # correct_known_mistakes(subjectsCollection,classifCollection)
    # add_whether_subject_is_part_of_core_with_expert_data(subjectsCollection)
    # add_indices(classifCollection,subjectsCollection)
    # sanity_checks_on_db(subjectsCollection)
    print "done with rto_mongodb_utils.py"