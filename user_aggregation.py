'''Script will calculate classification accuracy as a function of how many users are included in aggregation'''

# options
dataset = "TEST MRE11"          #what dataset to look at
minClassifications = 1          #inclusive lower bound number of classifications per segment in order to include that segment



__author__ = 'Peter'

from pymongo import MongoClient
import pandas as pd
import scipy as s
import time
import numpy as np

# open connection to local data
dbConnection = MongoClient("localhost", 27017)
subjectsCollection = dbConnection.RTO_20150929.subjects
classifCollection = dbConnection.RTO_20150929.classifications
# show that subjects_collection is a pymongo type variable
print 'Number of subjects:\t\t\t',subjectsCollection.count()
print 'Number of classifications:\t',classifCollection.count()

####### run over each classification and store its properties in a pandas dataframe
# .find() selects the actual entries so you can loop over them, index them, and so on
sel = classifCollection.find()
# set up dataframe with required columns
next up: understand how to set the index in DataFrame to the name of the segment through clever retrieval of subject_ids http://docs.mongodb.org/manual/core/cursors/
scores = pd.DataFrame(columns=["segment ID","cancer","proportion stained","staining intensity"])
# loop over each entry in rawdat
startTime = time.time()
for iC in range(sel.count()):
# for iC in range(100):
    scores.loc[iC,"segment ID"] = sel[iC]["subject_ids"][0]                     # store segment ID
    scores.loc[iC,"cancer"] = sel[iC]["annotations"][0]["a-1"]                  # store answer to Cancer yes/no
    scores.loc[iC,"proportion stained"] = sel[iC]["annotations"][1]["a-2"]      # store answer to proportion stained
    scores.loc[iC,"staining intensity"] = sel[iC]["annotations"][2]["a-3"]      # store answer to intensity of staining
    print iC
# transform variables to integers rather than objects (compare scores.dtypes with scores.convert_objects(convert_numeric=True).dtypes). This is because setting the dtype during initialisation of "scores" does not work
scores = scores.convert_objects(convert_numeric=True)
print "Time passed:",time.time()-startTime,"seconds"
# save the dataframe; load using x=pd.read_pickle(file_name)
scores.to_pickle("classifications_dataframe.pkl")




# close the connection to local mongoDB
dbConnection.close()