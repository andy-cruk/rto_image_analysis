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
# set up dataframe with indices (row names) equal to the identifier of the segment
segments = list()
for cl in classifCollection.find().limit():
    segments.append(cl["subject_ids"][0])

# initialise 'scores' dataframe with columns and rows. If you don't initialise with this number of rows, it's unbearably slow to fill the rows 1 by 1 in the loop later.
scores = pd.DataFrame(columns=["cancer","proportion stained","staining intensity"],index=segments,dtype=int)

# # keep track of time for entire loop
startTime = time.time()
# # keep track of time inside the loop
t=np.empty([1])
t[0]=time.time()
# loop over all requested classifications
ix = 0 #index in 'scores'
for cl in classifCollection.find().limit(): # go through all samples
    scores.iloc[ix,0] = cl["annotations"][0]["a-1"]          # store answer to Cancer yes/no
    scores.iloc[ix,1] = cl["annotations"][1]["a-2"]      # store answer to proportion stained
    scores.iloc[ix,2] = cl["annotations"][2]["a-3"]      # store answer to intensity of staining
    ix+=1
    if ix%10000==0:
        print ix,"Last 10k samples took",time.time()-t[0],"seconds"
        t[0]=time.time()

# transform variables to integers rather than objects (compare scores.dtypes with scores.convert_objects(convert_numeric=True).dtypes). This is because setting the dtype during initialisation of "scores" does not work
scores = scores.convert_objects(convert_numeric=True)
print "Time passed:",time.time()-startTime,"seconds"
# save the dataframe; load using x=pd.read_pickle(file_name)
scores.to_pickle("classifications_dataframe.pkl")




# close the connection to local mongoDB
dbConnection.close()