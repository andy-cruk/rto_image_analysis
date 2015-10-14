'''Script will calculate classification accuracy as a function of how many users are included in aggregation'''
__author__ = 'Peter'

from pymongo import MongoClient
import pandas as pd
import scipy as s
import time
import numpy as np
import os.path

# user options
stain = "TEST MRE11"                 # what sample to look at; must match metadata.stain_type in subjects database, TEST MRE11
minClassifications = 1          # min number of classifications the segment needs to have

# set dictionary with filters to feed to
filterSubjects = {"$and":[
    {"metadata.stain_type":stain},
    {"classification_count":{"$gt":minClassifications}},
    ]}

# save file for scores
classificationsDataframeFn = "classifications_dataframe_"+stain+".pkl"  # will store the pandas dataframe created from all the classifications; file will load if the file exists already to prevent another 30 to 60 min of processing.

########### FUNCTION DEFINITIONS
def pymongo_connection_open():
    # open connection to local data and return subjectsCollection, classifCollection, nSj and nCl
    dbConnection = MongoClient("localhost", 27017)
    subjectsCollection = dbConnection.RTO_20150929.subjects
    classifCollection = dbConnection.RTO_20150929.classifications
    nSj = subjectsCollection.count()
    nCl = classifCollection.count()
    print 'Number of subjects:\t\t\t', nSj
    print 'Number of classifications:\t', nCl
    return subjectsCollection,classifCollection,dbConnection
def pymongo_connection_close():
    dbConnection.close()
def classifications_dataframe_fill_and_save(fn=classificationsDataframeFn):
    # first grab all the subjects that satisfy the filterSubjects criteria and store classification_count and the core it belongs to (group_id)
    subjectCursor = subjectsCollection.find(filter=filterSubjects,projection=("classification_count","group_id"))
    assert subjectCursor.count()>0,"no records found for filter"
    col = ("subjectID","nClassifications","nCancer","noStain","<25%Stain","<50%Stain","<75%Stain","<95%Stain","<100%Stain","stainWeak","stainMedium","stainStrong","errorProp","errorIntensity")
    # set up dataframe that will store one row per segment, with classification scores. It will count occurrences in each of the buckets, which can later be divided by number of classifications
    cl = pd.DataFrame(data=None,columns=col)
    clIx = 0 # dataframe index
    ts=time.time()
    for sj in subjectCursor: # loop over each entry in subjectCursor and aggregate ratings into dataframe
        # store the subject id in the pandas dataframe, initialising the row
        cl.loc[clIx,"subjectID"] = sj["_id"]
        # initialise the rest of the row with zeros
        cl.iloc[clIx,1:] = 0
        # collect all classifications from this particular segment
        clCursor = classifCollection.find(filter={"subject_ids":[sj["_id"]]},projection=("annotations",))
        for iCl in clCursor: # loop over each classification that was retrieved from database
            cl.loc[clIx,"nClassifications"]+=1
            # cancer yes/no store in dataframe
            cancer = int(iCl["annotations"][0]["a-1"])
            if cancer == 1: # if yes cancer
                cl.loc[clIx,"nCancer"]+=int(iCl["annotations"][0]["a-1"])
                propStain = int(iCl["annotations"][1]["a-2"])
                if propStain == 1:
                    cl.loc[clIx,"noStain"]+=1
                elif propStain == 2:
                    cl.loc[clIx,"<25%Stain"]+=1
                elif propStain == 3:
                    cl.loc[clIx,"<50%Stain"]+=1
                elif propStain == 4:
                    cl.loc[clIx,"<75%Stain"]+=1
                elif propStain == 5:
                    cl.loc[clIx,"<95%Stain"]+=1
                elif propStain == 6:
                    cl.loc[clIx,"<100%Stain"]+=1
                else: # should always be one of the above
                    cl.loc[clIx,"errorProp"]+=1
                # now do intensity
                intensityStain = int(iCl["annotations"][2]["a-3"])
                if intensityStain == 1:
                    cl.loc[clIx,"stainWeak"]+=1
                elif intensityStain == 2:
                    cl.loc[clIx,"stainMedium"]+=1
                elif intensityStain == 3:
                    cl.loc[clIx,"stainStrong"]+=1
                else: #should always be one of the above
                    cl.loc[clIx,"errorIntensity"]+=1
            elif cancer == 2: #if no cancer
                continue
        clIx +=1
        if clIx%500==0:
            print 100*clIx/subjectCursor.count(),"%"
    # convert numbers to int (otherwise they're stored as object)
    cl = cl.convert_objects(convert_numeric=True)
    print "Saving dataframe to",fn
    cl.to_pickle(fn)
    print "Done aggregating dataframe with classifications and saving"
    return cl
def classifications_dataframe_load(fn=classificationsDataframeFn):
    # load dataframe and return scores
    cl = pd.read_pickle(fn)
    return cl
def plot_classifications(scores):
    ###### here are some example commands of how to work with the dataframe
    # select only classifications that said 'yes' to cancer and   return all columns
    scores[scores['cancer']==1]
    # select only classifications with cancer and return second and third column
    scores.loc[scores['cancer']==1,'proportion stained':'staining intensity']
    # plot histogram of scores across columns (combine with previous to get histogram for cancer only)
    scores.hist()
    # combine selection criteria, e.g. to check for unallowed score combinations
    scores.loc[(scores['cancer']==2) & (scores['proportion stained']!=0),'proportion stained':'staining intensity'].hist(bins=20)
    # view data type of columns
    scores.dtypes
    # select only classifications from a particular segment. First get the segment ObjectIds and then use those to index the scores dataframe.
    segments = scores.index
    scores.loc[str(segments[0])]

########### FUNCTION EXECUTION
# set up pymongo objects
subjectsCollection,classifCollection,dbConnection = pymongo_connection_open()
# check if dataframe with classifications exists; if not, run over each classification and store its properties in a pandas dataframe. If it does exist, load it.
if os.path.isfile(classificationsDataframeFn) == False:
    cl = classifications_dataframe_fill_and_save(fn=classificationsDataframeFn)
else: #dataframe available, load instead of fill
    cl = classifications_dataframe_load(fn=classificationsDataframeFn)

# close the connection to local mongoDB
pymongo_connection_close()
print "All done with script"
