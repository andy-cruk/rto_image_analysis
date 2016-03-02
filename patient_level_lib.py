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
    _,coll = get_cores_collection()
    results = coll.find(filter=mongoFilter, projection=projection).limit(limit)
    df = pd.io.json.json_normalize(results)
    return df


def combine_cores_per_patient(df=load_cores_into_pandas(), function=np.nanmean, stain='p21'):
    """ Takes a dataframe with core-level data, loads the lookup table to see what cores belong to each patient, and combines
    all attributes of those cores using a summarising function specified in the function call.
    :param df: pandas dataframe like the one returned by load_cores_into_pandas (cores*attributes)
    :param function: function that takes an array of values and returns a single number to summarise those values (to be applied across cores from same patient)
    :param stain: what stain to summarise for
    :return: a pandas dataframe with a column indicating patient ID (one row per patient) and all the same columns as df
    """
    pass
    # load lookup table

    # set up df to hold patient-level data (make sure one column called 'patient' is correct as it might be used to join multiple dfs when comparing between stains

    # loop over each patient and compute summary scores across cores

