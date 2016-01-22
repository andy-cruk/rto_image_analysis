from pymongo import MongoClient
import pandas as pd
import numpy as np

def get_cores_collection():
    """ Opens a connection to the 'cores' collection in 'results' database through pymongo
    :return: (pymongo database connection,reference to 'cores' collection)
    """
    db = MongoClient("localhost", 27017)
    subjectsCollection = db.results.cores
    return (db,subjectsCollection)


def load_cores_into_pandas(mongoFilter={},projection={'_id':False}):
    """ Build a pandas dataframe for all cores by first setting up connection to the collection through get_cores_collection,
    and then loading the requested cores based on mongoFilter.
    :param mongoFilter: filter directly passed to pymongo 'find'. e.g. {'stain':'p21'}
    :param projection: fields to be returned by the database query and therefore columns included in the pandas dataframe
    :return: a pandas dataframe with rows equal to number of documents found based on mongoFilter, and columns equal to number of fields requested based on projection
    """
    _,coll = get_cores_collection()
    results = coll.find(filter=mongoFilter,projection=projection)
    df = pd.DataFrame(list(results))
    return df


def combine_cores_per_patient(df=load_cores_into_pandas(),function=np.nanmean(),stain='p21'):
    """ Takes a dataframe with core-level data, loads the lookup table to see what cores belong to each patient, and combines
    all attributes of those cores using a summarising function specified in the function call.
    :param df: pandas dataframe like the one returned by load_cores_into_pandas (cores*attributes)
    :param function: function that takes an array of values and returns a single number to summarise those values (to be applied across cores from same patient)
    :param stain: what stain to summarise for
    :return: a pandas dataframe with a column indicating patient ID (one row per patient) and all the same columns as df
    """
    # load lookup table

    # set up df to hold patient-level data (make sure one column called 'patient' is correct as it might be used to join multiple dfs when comparing between stains

    # loop over each patient and compute summary scores across cores

