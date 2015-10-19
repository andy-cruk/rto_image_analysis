'''Script will calculate classification accuracy as a function of how many users are included in aggregation'''
__author__ = 'Peter'

from pymongo import MongoClient
import pandas as pd
import scipy as s
import time
import numpy as np
import os.path
import matplotlib.pyplot as plt
desired_width = 320
pd.set_option('display.width', desired_width)

# user options
stain = "TEST MRE11"  # what sample to look at; must match metadata.stain_type in subjects database,e.g. "TEST MRE11" or "MRE11"
minClassifications = 1  # min number of classifications the segment needs to have, inclusive

# set dictionary with filters to feed to
filterSubjects = {"$and": [
    {"metadata.stain_type": stain},
    {"classification_count": {"$gte": minClassifications}},
]}

# save file for scores
classificationsDataframeFn = "classifications_dataframe_" + stain + ".pkl"  # will store the pandas dataframe created from all the classifications; file will load if the file exists already to prevent another 30 to 60 min of processing.


########### FUNCTION DEFINITIONS
def pymongo_connection_open():
    """ open connection to local data and return subjectsCollection, classifCollection, and dbConnection
    Change as appropriate if your database has a different name.
    """
    dbConnection = MongoClient("localhost", 27017)
    subjectsCollection = dbConnection.RTO_20150929.subjects
    classifCollection = dbConnection.RTO_20150929.classifications
    nSj = subjectsCollection.count()
    nCl = classifCollection.count()
    print 'Number of subjects:\t\t\t', nSj
    print 'Number of classifications:\t', nCl
    return subjectsCollection, classifCollection, dbConnection
def pymongo_connection_close():
    """Close pymongo connection"""
    dbConnection.close()
def classifications_dataframe_fill_and_save(fn=classificationsDataframeFn):
    """Find all classifications for requested stain and aggregate responses for each subject
    This function saves and returns a pandas dataframe that has one row per subject (=segment) and columns that indicate proportion of all classifications given that answer.
    Note: proportion for different stain answers might not add up to 1 as people that say 'no cancer' still count towards classifications
    Note that you should have set an index in your mongoDB in the db.classifications.subject_ids field, otherwise this code will be agonisingly slow.
    """
    # first grab all the subjects that satisfy the filterSubjects criteria
    subjectCursor = subjectsCollection.find(filter=filterSubjects, projection=("classification_count", "metadata.id_no"))
    assert subjectCursor.count() > 0, "no records found for filter"
    col = ("subjectID", "core", "nClassifications", "nCancer", "<25%Stain", "<50%Stain", "<75%Stain", "<95%Stain",
           "<100%Stain", "stainWeak", "stainMedium", "stainStrong")
    # set up dataframe that will store one row per segment, with classification scores. It will count occurrences in each of the buckets, which can later be divided by number of classifications
    cl = pd.DataFrame(data=None, columns=col)
    clIx = 0  # dataframe index
    ts = time.time()
    for sj in subjectCursor:  # loop over each entry in subjectCursor and aggregate ratings into dataframe
        # store the subject id in the pandas dataframe, initialising the row
        cl.loc[clIx, "subjectID"] = sj["_id"]
        # store what core it comes from
        cl.loc[clIx, "core"] = sj["metadata"]["id_no"]
        # initialise the rest of the row with zeros
        cl.iloc[clIx, 2:] = 0
        # collect all classifications from this particular segment
        clCursor = classifCollection.find(filter={"subject_ids": [sj["_id"]]}, projection=("annotations",))
        for iCl in clCursor:  # loop over each classification that was retrieved from database
            cl.loc[clIx, "nClassifications"] += 1
            # cancer yes/no store in dataframe
            cancer = int(iCl["annotations"][0]["a-1"])
            if cancer == 1:  # if yes cancer
                propStain = int(iCl["annotations"][1]["a-2"])           #retrieve proportion stained category
                intensityStain = int(iCl["annotations"][2]["a-3"])      #retrieve intensity staining category
                # check if this is an invalid entry because proportion and intensity should not be 0
                if (propStain == 0 or intensityStain == 0):
                    # reduce number of classifications by 1, effectively taking it out of the tally
                    cl.loc[clIx, "nClassifications"] -= 1
                    # continue the loop with next classification
                    continue
                cl.loc[clIx, "nCancer"] += 1
                if propStain == 2:
                    cl.loc[clIx, "<25%Stain"] += 1
                elif propStain == 3:
                    cl.loc[clIx, "<50%Stain"] += 1
                elif propStain == 4:
                    cl.loc[clIx, "<75%Stain"] += 1
                elif propStain == 5:
                    cl.loc[clIx, "<95%Stain"] += 1
                elif propStain == 6:
                    cl.loc[clIx, "<100%Stain"] += 1
                else:  # should always be one of the above
                    cl.loc[clIx, "errorProp"] += 1
                # now do intensity
                if intensityStain == 1:
                    cl.loc[clIx, "stainWeak"] += 1
                elif intensityStain == 2:
                    cl.loc[clIx, "stainMedium"] += 1
                elif intensityStain == 3:
                    cl.loc[clIx, "stainStrong"] += 1
                else:  # should always be one of the above
                    cl.loc[clIx, "errorIntensity"] += 1
            elif cancer == 2:  # if no cancer
                continue
        clIx += 1
        if clIx % 200 == 0:
            print 100 * clIx / subjectCursor.count(), "%"
    # convert numbers to int (otherwise they're stored as object)
    cl = cl.convert_objects(convert_numeric=True)
    # normalise everything but nClassifications to value between 0 and 1
    # create cln, which is normalised version of cl
    cln = cl.copy()
    # normalise each of the columns starting at nCancer and all the ones to the right, using my own function def
    cln.loc[:,"nCancer":] = normalise_dataframe_by_ix(cl,cl.columns.get_loc("nClassifications"),range(cl.columns.get_loc("nCancer"),len(cl.columns)))
    print "Saving dataframe to", fn
    cln.to_pickle(fn)
    print "Done aggregating dataframe with classifications and saving"
    return cln
def classifications_dataframe_load(fn=classificationsDataframeFn):
    """load dataframe and return pandas dataframe"""
    cln = pd.read_pickle(fn)
    return cln
def plot_classifications(cln):
    """example commands of how to work with a pandas dataframe"""
    # select only classifications that have someone saying it's cancer and return all columns
    print cln[cln['nCancer'] > 0]
    # combine selection criteria and select columns across a range, then plot histogram
    cln.loc[(cln['nCancer'] > 0.2) & (cln['noStain'] != 0), 'noStain':'<100%Stain'].hist(bins=20)
    plt.show()
def cln_add_columns_aggregating_stain(cln):
    """Add 2 columns that aggregate stain proportion and intensity answers into new columns
    Will disregard those that answered 'not cancer' and calculate aggregates based on only those people that said there is cancer.
    Aggregate is calculated by
    Proportion: transforming to numbers, taking mean, transforming back to categories
    Intensity: mean across categories {1,2,3}
    Returns cln with 2 additional columns
    """
    # define middle of each category. 2: 1 to 25% of cancer cells stained, 3: 25 to 50%, 4: 50 to 75%, 5: 75 to 95%, 6: 95 to 100%}
    meanIntensities = np.array([13,37.5,62.5,85,97.5])
    # this next one's a beast and should probably be spread out. The numerator takes the n * 5 matrix of stain proportions and dot multiplies it with the mean intensities; effectively
    # summing the products of proportion*meanIntensities. This is then divided by the proportion of people saying there was indeed cancer to correct
    # for the fact that we only want to include people that answered 'yes cancer'. For samples w/o anyone saying 'cancer', this results in a NaN
    cln["aggregateProp"] = (cln.loc[:,"<25%Stain":"<100%Stain"].dot(meanIntensities)) / cln.nCancer
    # same deal for intensity, but now we multiple simply by 1,2,3 for weak, medium, strong respectively.
    cln["aggregateIntensity"] = (cln.loc[:,"stainWeak":"stainStrong"].dot(np.array([1,2,3]))) / cln.nCancer
    return cln
def core_dataframe_fill_and_save(cln):
    """takes a dataframe cln with columns indicating proportions of responses and aggregates all subjects for a single core
    Each subject belongs to a core, so we'll loop over cores and collect each segment's data. Subjects are combined
    un-weighted, i.e. a subject with 20 classifications is weighted equally to subject with 150 classifications.
    Assumes the first column to normalise is nCancer and EVERY FOLLOWING COLUMN will be normalised too.
    """
    # initialise new DataFrame that will have one row per core
    cores = pd.DataFrame(data=None,index=range(0,len(get_core_ids(cln))),columns=cln.columns,dtype="int64")
    # put in core IDs
    cores["core"]=get_core_ids(cln)
    # get rid of unnecessary subjectID column
    cores = cores.drop("subjectID",axis=1)
    # add column indicating number of subjects for each core
    cores.insert(2,"nSubjects",np.nan)
    # loop over each core
    for ix,core in cores.iterrows(): # function retrieves unique cores from dataframe
        coreRowsInCln = cln[cln["core"]==core.core]
        # note this takes the mean across all segments, including those that probably didn't have cancer.
        cores.loc[ix,"nClassifications":] = coreRowsInCln.loc[:,"nClassifications":].mean()
        cores.loc[ix,"nSubjects"] = len(coreRowsInCln.index)
    return cores
def get_core_ids(cln):
    """retrieves a list of cores expressed as objects for a given set of classifications
    Input is a dataframe with classifications containing a column called "core"
    """
    cores = list()
    for c in cln["core"]:
        if c not in cores:
            cores.append(c)
    return cores
def normalise_dataframe_by_ix(df,divideByColumn,columnsToDivide):
    """Returns dataframe with one set of columns divided by one of the columns in the dataframe.
    provide INDICES, not COLUMN NAMES. The columnsToDivide will be divided by divideByColumn
    """
    df2 = df.copy()
    # perform the division only for those columns requested
    return df2.iloc[:,columnsToDivide].div(df2.iloc[:,divideByColumn],axis="index")


########### FUNCTION EXECUTION
# set up pymongo objects
subjectsCollection, classifCollection, dbConnection = pymongo_connection_open()
# check if dataframe with classifications exists; if not, run over each classification and store its properties in a pandas dataframe. If it does exist, load it.
if os.path.isfile(classificationsDataframeFn) == False:
    cln = classifications_dataframe_fill_and_save(fn=classificationsDataframeFn)
else:  # dataframe available, load instead of fill
    cln = classifications_dataframe_load(fn=classificationsDataframeFn)

cln = cln_add_columns_aggregating_stain(cln)
cores = core_dataframe_fill_and_save(cln)

# close the connection to local mongoDB
pymongo_connection_close()
print "All done with script"
