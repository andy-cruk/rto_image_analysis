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
    return db, patientsCollection


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
    :return: the dataframe with 1 patient per row
    """
    # get data from 'cores' database
    df = load_cores_into_pandas(projection=['coreID', 'stain', aggregate])
    # load lookup table
    lut = pd.read_excel('info/patient_core_LUT_encrypted.xlsx')
    # merge the two, yielding a dataframe the size of df but now with patient ID as extra column
    # In this df, the 'core' and 'coreID' are not unique; There are 5k cores in the df, but only 1k unique coreIDs,
    # and only about 300 unique patient_key. So every patient has 3 to 4 coreIDs, and each patient has about 15 cores in
    # total on average. (numbers subject to change as more data becomes available)
    df2 = df.merge(right=lut, how='left', left_on='coreID', right_on='core', sort=True, copy=False)
    # drop some duplicate columns; after this use the columns 'core' and 'stain'.
    df2.drop(['coreID', aggregate+'.stain', aggregate+'.core'], axis=1, inplace=True)
    # we can use a pivot table to combine the data for each patient and each stain type, applying 'function' to the data
    # from multiple cores
    df3 = pd.pivot_table(df2,
                         index=["patient_key"],
                         values=[
                             aggregate+'.aggregateSQSCorrected',
                             aggregate+'.aggregateIntensityCorrected',
                             aggregate+'.aggregatePropCorrected',
                             aggregate+'.nClassificationsTotal',
                             aggregate+'.expSQS',
                             aggregate+'.expIntensity',
                             aggregate+'.expProp',
                         ],
                         columns=['stain'],
                         aggfunc=function
                         )
    # change all column types to string
    df3.columns.names = ['measure', 'stain']
    # write to excel with multi-index
    df3.to_excel('results/patient_aggregated.xlsx')

    # WRITE TO MONGODB
    # this is one way to flatten multi-index on columns
    df3 = pd.DataFrame(df3.to_records())
    # remove the aggregate method from each column header
    df3.columns = [s.replace(aggregate+'.', '') for s in df3.columns]
    # write to mongodb. If you ever wanted to get this out again, use pd.MultiIndex.from_tuples
    _, coll = get_patients_collection()
    # coll.delete_many({})  # empty it if necessary
    coll.insert_many(df3.to_dict('records'))

    return df3



