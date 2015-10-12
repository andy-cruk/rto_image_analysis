'''Script will calculate classification accuracy as a function of how many users are included in aggregation'''
__author__ = 'Peter'

from pymongo import MongoClient
import pandas as pd
import scipy as s
import time
import numpy as np
import os.path

# options
stain = ""  # what stain to look at
minClassificationsPerSegment = 1  # inclusive lower bound number of classifications per segment in order to include that segment
maxClassifications = 10 ^ 10  # max classifications to retrieve; mostly for debugging when you want the script to be fast

# constants
classificationsDataframeFn = "classifications_dataframe.pkl"  # will store the pandas dataframe created from all the classifications; file will load if the file exists already to prevent another 30 to 60 min of processing.

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
    return subjectsCollection,classifCollection,nSj,nCl
def pymongo_connection_close():
    dbConnection.close()
def classifications_dataframe_fill_and_save_one_by_one(maxClassifications=maxClassifications,fn=classificationsDataframeFn):
    # Pandas dataframe and fill with each classification, including what core it came from and what cancer
    print "Putting together dataframe. Could take 30+ mins..."
    # collect a list of the segment identifier names for each classification. This will be used as index (row name) later.
    # Note that these segments are printed as an alphanumeric string, but the underlying string includes ObjectId('alphanumericstring')
    segments = list()
    for cl in classifCollection.find().limit(maxClassifications):
        segments.append(cl["subject_ids"][0])

    # initialise 'scores' dataframe with columns and rows. If you don't initialise with this number of rows, it's unbearably slow to fill the rows 1 by 1 in the loop later.
    scores = pd.DataFrame(columns=["cancer", "proportion stained", "staining intensity","core","stain"], index=segments)

    # loop over all requested classifications and fill the entire dataframe
    ix = 0  # index in 'scores'
    t = time.time()
    for cl in classifCollection.find().limit(maxClassifications):  # go through all samples
        scores.iloc[ix, 0] = cl["annotations"][0]["a-1"]  # store answer to Cancer yes/no
        scores.iloc[ix, 1] = cl["annotations"][1]["a-2"]  # store answer to proportion stained
        scores.iloc[ix, 2] = cl["annotations"][2]["a-3"]  # intensity stained
        # add what core this classification came from and what stain type by accessing the subjectsCollection
        # 1 or more entries are missing the key 'stain_type' so only finding ones that have it; otherwise it'll return None and the if statement later will catch it
        sjCursor = subjectsCollection.find_one({'$and': [{'_id':segments[ix]},{["metadata"]["stain_type"]:{"$exists":True}}]})
        # check if a matching subject was found for this segment - not always the case for some reason
        if sjCursor is None:
            print "no matching subject found for core with index",ix,", or it didn't have the 'stain_type' key"
            scores.iloc[ix,3] = np.NaN
            scores.iloc[ix,4] = np.NaN
            ix+=1
            continue
        # all should be ok to read out the segment ID
        # scores.iloc[ix,3] = sjCursor["group_id"] # this is excruciatingly slow; takes about 400 ms per entry, probably because it is an ObjectId
        scores.iloc[ix,3] = str(sjCursor["group_id"]) # this is much faster because now it has to put in a string rather than ObjectId.
        scores.iloc[ix,4] = sjCursor["metadata"]["stain_type"]
        # update counter and report
        ix += 1
        if ix%20000  == 0:
            print "Classifications processed:\t",100*ix/nCl,"%, time remaining:",(1-ix/nCl)*((time.time()-t)/60)/(ix/nCl),"minutes"

    # transform variables to integers rather than objects (compare scores.dtypes with scores.convert_objects(convert_numeric=True).dtypes). This is because setting the dtype during initialisation of "scores" does not work
    scores = scores.convert_objects(convert_numeric=True)

    # save the dataframe; load using x=pd.read_pickle(classificationsDataframeFn)
    scores.to_pickle(fn)
    print "done saving dataframe"
    return scores
def classifications_dataframe_load(fn=classificationsDataframeFn):
    # load dataframe and return scores
    scores = pd.read_pickle(fn)
    assert len(scores)==classifCollection.find().count(),"number of scores in dataframe does not match number of items in MongoDB"
    return scores
def filter_classifications(scores,stain=None,minClassificationsPerSegment=None):
    # filter the data based on several criteria
    if stain is not None: # only filter if it is not None
        scores = scores[scores["stain"]==stain]
    if minClassificationsPerSegment is not None:
        segments = scores.index #get all segments

    return scores
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
subjectsCollection,classifCollection,nSj,nCl = pymongo_connection_open()
# # check if dataframe with classifications exists; if not, run over each classification and store its properties in a pandas dataframe. If it does exist, load it.
# if os.path.isfile(classificationsDataframeFn) == False:
#     scores = classifications_dataframe_fill_and_save_one_by_one(maxClassifications=maxClassifications,fn=classificationsDataframeFn)
# else: #dataframe available, load instead of fill
#     scores = classifications_dataframe_load(fn=classificationsDataframeFn)
#
#
# # close the connection to local mongoDB
# pymongo_connection_close()
# print "All done with script"
