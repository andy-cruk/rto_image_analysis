''' This set of functions lets you manipulate the mongoDB database
'''

import user_aggregation
import os
import pandas as pd
import pymongo
import re

currentDB = 'RTO_20151209'
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
    ''' Works on subject database
    :return: returns nothing
    '''
    for d in db.find({"metadata.stain_type":{"$exists": True}}):
        db.update({"_id":d["_id"]},{"$set":{"metadata.stain_type_lower":d["metadata"]["stain_type"].lower()}})


def complete_subject_naming_to_correct_zooniverse_inconsistencies(db):
    """ Some subject entries have group.name, others have metadata.stain_type

    :param db:
    :return:
    """


def add_whether_subject_is_part_of_core_with_expert_data(db):
    """
    Uses sj_in_expert core to add a boolean to each entry regarding whether or not its parent core has an expert score
    Only updates stains that have GS excel sheet
    :param db: pymongo handle to a collection, in this case the subjects collection
    :return: nothing
    """
    print 'adding indices to database'
    # find stains with GS data
    f = os.listdir('GS')
    stains = [x.lstrip('GS_').rstrip('.xlsx') for x in f]
    # loop over each stain
    for stain in stains:
        print "starting stain "+stain
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
        assert(db.find({'metadata.stain_type_lower':stain}).count() == db.find({'group.name':stain}).count())
        # check id_no exists in each document
        assert(db.find({'metadata.stain_type_lower':stain}).count() == db.find({'metadata.stain_type_lower':stain,'metadata.id_no':{"$exists":True}}).count())
        # check id_no has consistent format
        id_no = list(db.find({'metadata.stain_type_lower':stain},{'_id':False,'metadata.id_no':True}))
        lengths = [len(x['metadata']['id_no']) for x in id_no]
        assert(len(set(lengths)) == 1)


def correct_known_mistakes(db):
    """ This corrects a number of mistakes that have to be corrected through hardcoding

    """
    print "correcting known mistakes"
    db.update_many({'group.name':'test_mre11'},{"$set":{"group.name":'test mre11'}})
    db.update_many({"metadata.id_no":"7913 ME11"},{"$set":{"metadata.id_no":"7913 MRE11"}})
    db.update_many({"metadata.id_no":"7838MRE11"},{"$set":{"metadata.id_no":"7838 MRE11"}})
    db.update_many({"metadata.id_no":"7972 MRE1"},{"$set":{"metadata.id_no":"7972 MRE11"}})
    db.update_many({"metadata.id_no":"8302 MRE11_"},{"$set":{"metadata.id_no":"8302 MRE11"}})
    db.update_many({"metadata.id_no":"8079 p21_"},{"$set":{"metadata.id_no":"8079 p21"}})
    db.update_many({"metadata.id_no":"7987  p21"},{"$set":{"metadata.id_no":"7987 p21"}})
    db.update_many({"metadata.id_no":"8583  p21"},{"$set":{"metadata.id_no":"8583 p21"}})
    # a bunch of rad50 don't have metadata.id_no. Although zooniverse should correct this, you could add it like this.
    # Also see db.getCollection('subjects').distinct('metadata.orig_filename',{"metadata.stain_type_lower":"rad50","metadata.id_no":{"$exists":false}})
    # applied in mongodb
    docs = db.find({"metadata.stain_type_lower":"rad50","metadata.id_no":{"$exists":False}})
    coreNames = [x["metadata"]["orig_filename"].lstrip('RAD50/TMA 4A/')[:10].replace('_',' ') for x in docs.rewind()]
    for ix,doc in enumerate(docs.rewind()):
        db.update_one({'_id':doc['_id']},{'$set':{'metadata.id_no':coreNames[ix]}})
    # lovely regexp in case we want to use this for other stains. Note that orig_filename or orig_file_name is stored differently even within a stain, so will have to hardcode for each stain
    # coreNames = [x["metadata"]["orig_filename"][re.search("^\w*/\w*/",x["metadata"]["orig_filename"]).end(0):re.search("/\w*\.jpg$",x["metadata"]["orig_filename"]).start(0)].replace('_',' ') for x in docs.rewind()]




if __name__ == "__main__":
    subjectsCollection, classifCollection, dbConnection = user_aggregation.pymongo_connection_open()
    add_lowercase_metadata_staintype(subjectsCollection)
    add_whether_subject_is_part_of_core_with_expert_data(subjectsCollection)
    add_indices(classifCollection,subjectsCollection)
    correct_known_mistakes(subjectsCollection)
    sanity_checks_on_db(subjectsCollection)
    print "done with rto_mongodb_utils.py"