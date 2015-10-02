'''Script will calculate classification accuracy as a function of how many users are included in aggregation'''

# options
dataset = "TEST MRE11"          #what dataset to look at
minClassifications = 1          #inclusive lower bound number of classifications per segment in order to include that segment
maxClassifications = 10^10       #max classifications to retrieve; mostly for debugging when you want the script to be fast

# constants
classificationsDataframeFn = "classifications_dataframe.pkl"    # will store the pandas dataframe created from all the classifications; file will load if the file exists already to prevent another 30 to 60 min of processing.


__author__ = 'Peter'

from pymongo import MongoClient
import pandas as pd
import scipy as s
import time
import numpy as np
import os.path

# open connection to local data
dbConnection = MongoClient("localhost", 27017)
subjectsCollection = dbConnection.RTO_20150929.subjects
classifCollection = dbConnection.RTO_20150929.classifications
nSj = subjectsCollection.count()
nCl = classifCollection.count()
# show that subjects_collection is a pymongo type variable
print 'Number of subjects:\t\t\t',nSj
print 'Number of classifications:\t',nCl

####### check if dataframe with classifications exists; if not, run over each classification and store its properties in a pandas dataframe. If it does, load it.
if os.path.isfile(classificationsDataframeFn)==False:
    print "No saved .pkl with dataframe for classifications detected, putting it together now. Could take 30+ mins..."
    # collect a list of the segment identifier names for each classification. This will be used as index (row name) later
    segments = list()
    for cl in classifCollection.find().limit(maxClassifications):
        segments.append(cl["subject_ids"][0])

    # initialise 'scores' dataframe with columns and rows. If you don't initialise with this number of rows, it's unbearably slow to fill the rows 1 by 1 in the loop later.
    scores = pd.DataFrame(columns=["cancer","proportion stained","staining intensity"],index=segments)

    # loop over all requested classifications
    ix = 0 #index in 'scores'
    for cl in classifCollection.find().limit(maxClassifications): # go through all samples
        scores.iloc[ix,0] = cl["annotations"][0]["a-1"]      # store answer to Cancer yes/no
        scores.iloc[ix,1] = cl["annotations"][1]["a-2"]      # store answer to proportion stained
        scores.iloc[ix,2] = cl["annotations"][2]["a-3"]      # store answer to intensity of staining
        ix+=1
        if ix%10000==0:
            print "Samples processed:\t",100*ix/nCl,"%"

    # transform variables to integers rather than objects (compare scores.dtypes with scores.convert_objects(convert_numeric=True).dtypes). This is because setting the dtype during initialisation of "scores" does not work
    scores = scores.convert_objects(convert_numeric=True)

    # save the dataframe; load using x=pd.read_pickle(classificationsDataframeFn)
    scores.to_pickle(classificationsDataframeFn)
    print "done saving dataframe"
else:
    print ".pkl with classifications in dataframe found; loading now..."
    scores = pd.read_pickle(classificationsDataframeFn)
    print "... done loading dataframe"

####### here are some example commands of how to work with the dataframe
# # select only classifications that said 'yes' to cancer and return all columns
# scores[scores['cancer']==1]
# # select only classifications with cancer and return second and third column
# scores.loc[scores['cancer']==1,'proportion stained':'staining intensity']
# # plot histogram of scores across columns (combine with previous to get histogram for cancer only)
# scores.hist()
# # combine selection criteria, e.g. to check for unallowed score combinations
# scores.loc[(scores['cancer']==2) & (scores['proportion stained']!=0),'proportion stained':'staining intensity'].hist(bins=20)
# # view data type of columns
# scores.dtypes


####### close the connection to local mongoDB
dbConnection.close()
print "All done with script"