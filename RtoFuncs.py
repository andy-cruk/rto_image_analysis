'''Functions to analyse RtO data
Assumes the following number of correlations:
The correlations between expert and citizens that are returned are, in order of columns:
        0  Proportion as %                     Pearson
        1  Proportion as %                     Spearman
        2  Proportion as category              Spearman
        3  Intensity                           Spearman
        4  Proportion as % * intensity         Pearson
        5  Proportion as % * intensity         Spearman
        6  Proportion as category * intensity  Spearman
        7  Proportion as category + intensity  Spearman
'''
# Set the number of correlations calculated throughout these functions. Adjust list of correlations alongside this number if anything changes.
nCorrelations = 8


from pymongo import MongoClient
import pandas as pd
from scipy import stats as s
import time
import numpy as np
import os.path
import matplotlib.pyplot as plt

def pymongo_connection_open(sjDB="RTO_20150929",clDB="RTO_20150929"):
    """ open connection to local data and return subjectsCollection, classifCollection, and dbConnection
    Change as appropriate if your database has a different name.
    Returns subjectsCollection,classifCollection,dbConnection
    """
    dbConnection = MongoClient("localhost", 27017)
    subjectsCollection = dbConnection[sjDB].subjects
    classifCollection = dbConnection[clDB].classifications
    return subjectsCollection, classifCollection, dbConnection
def pymongo_connection_close(dbConnection):
    """Close pymongo connection"""
    dbConnection.close()
def load_expert_scores(stain):
    '''
    Loads the excel sheet and populates a pandas dataframe, which is returned. Assumes a file called GS_[stain].xlsx
    :param stain: this is a string that matches metadata.stain_type as well as the name of the stain in the excel sheet
    :return: pandas dataframe with one row per core, and columns proportion (in %), intensity
    '''
def load_subject_data(subjectsCollection,filter):
    '''
    This one jumps straight into the subjects database, bypassing having to aggregate individual classifications. It will
    extract metadata.answer_counts to get total number responses of each type
    :param subjectsCollection: a PyMongo collection; probably subjectsCollection from pymongo_connection_open
    :param filter: a filter as used by mongodb to filter entries. Probably contains metadata.stain_type, classification_count, and metadata.id_no needs to be $in an array of cores that have expert scores. For $in see https://docs.mongodb.org/manual/reference/operator/query/in/
    :return subjectDat: a pandas dataframe with one segment per row; columns are as below. All columns will be normalised to number of ratings (i.e. proportions rather than absolute number of answers
    Columns in subjectDat are taken from the subjects Mongodb:
    classification_count
    cancer
    prop0
    prop1
    prop2

    '''
    # find all the subjects requested
    subjectCursor = subjectsCollection.find(filter)
    # set up pandas dataframe with index equal to the ObjectIds from the subjects
    subjectDat = pd.DataFrame(index=[a["_id"] for a in subjectCursor.rewind()])
    try playing with this
    pd.DataFrame(list(subjectsCollection.find({"metadata.stain_type":"TEST MRE11"},projection={"metadata.answer_counts.a-4_0":1})))
def subject_to_core_sample(subjects,coreIDs,nUsersPerSample):
    '''
    Aggregates the segment data into core data by sampling, such that you can simulate answers if you had n users rather than all N.
    This code returns a single such sample.
    This needs to be fast code as it will be done many times when sampling user numbers.
    What is sampled at core level is average proportion and intensity of staining; each subject's contribution to whole core is
    weighted by the proportion of people that said there was cancer in that subject.
    :param subjects: a pandas dataframe from load_subject_data that has colums for frequencies of answers, and code IDs that can be used
    to selectively sample from the dataframe per code.
    :param coreIDs: an array or Pandas Series of core ID objects that you want a sample for
    :param nUsersPerSample: how many users to sample for each segment, and therefore how many users to average over
    :return: a len(coreIDs) x 2 numpy array; first column is sampled Proportion (in %) for each core; second is sampled intensity
    '''
    # initialise the numpy array that will be returned

    # loop over each coreID
    for
        # loop over each segment
        for
            # can you independently sample intensity and proportion or should you sample individual subjects? Check whether intensity and proportion are correlated?
def sample_correlation(subjects,coreIDs,nUsersPerSample,expertScores):
    '''
    This will create a numpy array that contains sampled estimates of correlations between users and experts.
    :param subjects: pandas dataframe that contains all the subject frequencies data
    :param coreIDs: all core ID objects that you want to use in calculating the correlation (probably all the expert-scored cores)
    :param nUsersPerSample: how many users to sample per segment and therefore how many users to average over
    :param expertScores: Pandas dataframe, see load_expert_scores(), but should now be re-ordered and filter to match coreIDs
    :return: 1 x n numpy array, where n is the number of different correlations
    '''
    # transform coreIDs into a set so it can be easily compared
    assert(list(expertScores.index)==coreIDs,"expertScores needs to match coreIDs")
    # initialise numpy array with nans
    correlationSample = np.zeros([1,nCorrelations])*np.nan
    # get sampled data, which will return a len(coreID)*2 numpy array; first column is %, second intensity
    sample = subject_to_core_sample(subjects,coreIDs,nUsersPerSample)

    # calculate the correlations
    correlationSample[0] = s.pearsonr(sample[:,0],expertScores.proportion)
    correlationSample[1] = s.spearmanr(sample[:,0],expertScores.proportion)
    correlationSample[2] = s.spearmanr(percentage_to_category(sample[:,0]),percentage_to_category(expertScores.intensity))
    correlationSample[3] = s.spearmanr(sample[:,1],expertScores.intensity)
    correlationSample[4] = s.pearsonr(sample[:,1]*sample[:,0],expertScores.proportion*expertScores.intensity)
    correlationSample[5] = s.spearmanr(sample[:,1]*sample[:,0],expertScores.proportion*expertScores.intensity)
    correlationSample[6] = s.spearmanr(sample[:,1]*percentage_to_category(sample[:,0]),percentage_to_category(expertScores.proportion)*expertScores.intensity)
    correlationSample[7] = s.spearmanr(sample[:,1]+percentage_to_category(sample[:,0]),percentage_to_category(expertScores.proportion)+expertScores.intensity)
def generate_correlation_distributions(subjects,expertScores,coreIDs=None,nUsersPerSample=0,nSamples=1000):
    '''
    Runs nSamples times to calculate a set of correlations between a sampled set of core scores and the expert scores.
    Also loads the expert scores.
    :param subjects: pandas dataframe that contains all the subject frequencies data, one segment per row
    :param expertScores: from load_expert_scores(). Will be re-organised so it's in the same order as coreIDs
    :param coreIDs: all core ID objects that you want to use in calculating the correlation (probably all the expert-scored cores). Set to None to include all expert scores. Format is ObjectId but will be transformed to number in this code
    :param nUsersPerSample: how many users to sample per segment and therefore how many users to average over. Set to 0 to include all subjects
    :param nSamples: how many correlations to calculate and therefore how many samples to average over. If nUserPerSample=0 then nSamples will be set to 1.
    :return correlationSamples: a nSamples x nCorrelationMeasures numpy array that can be used to calculate mean correlation, confidence intervals, and so on.
    '''
    # if coreIDs is empty, fill it with ObjectIds matching expertScores
    if not coreIDs:

    # if coreIDs is NOT empty, expertScores needs to be wittled down and re-organised to match coreIDs. It is key that expertScores is a number, coreIDs is a string inside a ObjectId.


    yet to implement, some kind of sort and/or join is in order
    # initialise numpy array with nans
    correlationSamples = np.zeros([nSamples,nCorrelations])*np.nan
    for i in range(nSamples): # for each sample to calculate correlations for
        correlationSamples[i,:] = sample_correlation(subjects,coreIDs,nUsersPerSample,expertScores)
    return correlationSamples
def percentage_to_category(percentages,stain):
    """Takes percentage stained and transforms to categories, useful for calculating pseudo-allred
    :param percentages: numpy array or Pandas series with percentage cancer
    :param stain: string that identifies what scoring buckets were used
    :return: categ: same dimension as 'percentages', transformed to groups, float
    """
    # set up dataframe with thresholds and categories. Percentages are inclusive. So if you're a 38, you'll be assigned the category which has the first value above 38 (e.g. 50)
    if stain in ['TEST MRE11','MRE11','p21','rad50']:
        transform = pd.DataFrame(data=np.array([[0,1,2,3,4,5],[0,25,50,75,95,100]]).T,columns=("category","percentage")) # THIS IS RtO
    else:
        print "Add your stain to the list and check the buckets used for the stain are correct"
        raise
    categ = np.zeros([len(percentages)])
    for iP in range(len(percentages)):
        # find lowest category that includes the percentage (e.g. 20 is smaller than 30, 60, and 100, so should fit in the 30 category)
        try: # can fail sometimes apparently
            categ[iP] = min([x["category"] for _,x in transform.iterrows() if percentages[iP] <= x["percentage"]])
        except:
            categ[iP] = np.nan
    return categ