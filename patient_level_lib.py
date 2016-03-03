from pymongo import MongoClient
import pandas as pd
import numpy as np

def get_cores_collection():
    """ Opens a connection to the 'cores' collection in 'results' database through pymongo
    :return: (pymongo database connection,reference to 'cores' collection)
    """
    db = MongoClient("localhost", 27017)
    coresCollection = db.results.cores
    return (db, coresCollection)


def get_patients_collection():
    """ Opens a connection to the 'patients' collection in 'results' database through pymongo
    :return: (pymongo database connection,reference to 'patients' collection)
    """
    db = MongoClient("localhost", 27017)
    patientsCollection = db.results.patients
    return (db, patientsCollection)


def load_cores_into_pandas(mongoFilter=None, projection=None, limit=0):
    """ Build a pandas dataframe for all cores by first setting up connection to the collection through get_cores_collection,
    and then loading the requested cores based on mongoFilter.
    :param mongoFilter: filter directly passed to pymongo 'find'. e.g. {'stain':'p21'}
    :param projection: fields to be returned by the database query and therefore columns included in the pandas dataframe
    :param limit: limit number of documents returned; default is all (0)
    :return: a pandas dataframe with rows equal to number of documents found based on mongoFilter, and columns equal to number of fields requested based on projection
    """
    # set defaults from non-mutable default values
    if mongoFilter is None:
        mongoFilter = {}
    if projection is None:
        projection = {'_id': False}
    # get pymongo collection handle to cores database
    _, coll = get_cores_collection()
    results = coll.find(filter=mongoFilter, projection=projection).limit(limit)
    df = pd.io.json.json_normalize(results)
    return df


def combine_cores_per_patient(function=np.nanmean, aggregate='ignoring_segments'):
    """ Takes all data from the 'cores' database, loads the lookup table to see what cores belong to each patient, and combines
    all attributes of those cores using a summarising function specified in the function call.
    :param function: function that takes an array of values and returns a single number to summarise those values (to be applied across cores from same patient)
    :param aggregate: what method of aggregation to read from in the cores database
    :return: None
    """
    # get data from 'cores' database
    df = load_cores_into_pandas(projection=['coreID', 'stain', aggregate])
    # load lookup table
    lut = pd.read_excel('info/patient_core_LUT_encrypted.xlsx')
    # merge the two, yielding a dataframe the size of df but now with patient ID as extra column
    # In this df, the 'core' and 'coreID' are not unique; there's about 5 entries for each core or so (5k entries on 1k core IDs)
    df2 = df.merge(right=lut, how='left', left_on='coreID', right_on='core')

