'''Script will calculate classification accuracy as a function of how many users are included in aggregation
Note that the method used here, which aggregates all classifications for each subject (=segment), is inefficient. Instead,
all required data is stored in the 'subjects' database, from which # of responses can be read and then stochastically sampled
to do any bootstrapping. That would be orders of magnitude faster.

So don't use this code as inspiration for your own.

Nonetheless, this script generates a number of dataframes in order
cln         classifications: one row per subject (aggregated over users)
cores       one row per core (aggregated over subjects)
rhoBoot     one row per number of users included in bootstrap; compares expert to users
'''
__author__ = 'Peter'
from pymongo import MongoClient
import pandas as pd
from scipy import stats as s
import time
import numpy as np
import os.path
import matplotlib.pyplot as plt
from bson.objectid import ObjectId
desired_width = 320
pd.set_option('display.width', desired_width)



# USER OPTIONS
stain = "MRE11"  # what sample to look at; must match metadata.stain_type in subjects database,e.g. "TEST MRE11" or "MRE11", "rad50", "p21"
minClassifications = 1  # min number of classifications the segment needs to have, inclusive
# following not implemented:
numberOfUsersPerSubject = np.array(0) # will loop over each of the number of users and calculate Spearman rho. Set to 0 to not restrict number of users
samplesPerNumberOfUsers = 1       # for each value in numberOfUsersPerSubject, how many times to sample users with replacement. Set to 1 if you just want to run once, e.g. when you include all the users



# set dictionary with filters to feed to mongoDB
filterSubjects = {"$and": [
    {"metadata.stain_type": stain},
    {"classification_count": {"$gte": minClassifications}},
]}

# save file for scores
classificationsDataframeFn = "classifications_dataframe_" + stain + ".pkl"  # will store the pandas dataframe created from all the classifications; file will load if the file exists already to prevent another 30 to 60 min of processing.
# load GS data
coresGS = pd.read_excel("GS_"+stain+".xlsx")

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
def classifications_dataframe_fill(numberOfUsersPerSubject=numberOfUsersPerSubject):
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
    # initialise with strings and zeros; much slower than empty matrix
    # cl = pd.concat([pd.DataFrame(data=[["",""] for i in range(subjectCursor.count())],columns=col[0:2]),pd.DataFrame(np.zeros([subjectCursor.count(),len(col)-2]),columns=col[2:])],axis=1)
    clIx = 0  # dataframe index
    for sj in subjectCursor:  # loop over each entry in subjectCursor and aggregate ratings into dataframe
        # check if this subject's core is in the expert list. If not, go to next one.
        if not sj_in_expert_core(sj["metadata"]["id_no"]):
            continue
        # store the subject id in the pandas dataframe, initialising the row
        cl.loc[clIx, "subjectID"] = sj["_id"]
        # store what core it comes from
        cl.loc[clIx, "core"] = sj["metadata"]["id_no"]
        # initialise the rest of the row with zeros
        cl.iloc[clIx, 2:] = 0
        # collect all classifications from this particular segment and put in a numpy array for easy indexing later
        clCursor = np.array(list(classifCollection.find(filter={"subject_ids": [sj["_id"]]},projection=("annotations",))))
        # select a random subset of items to to selected from the cursor, or select all if all subjects were requested or not sufficient users available
        if numberOfUsersPerSubject == 0: # all requested
            selectedSubjects = np.arange(len(clCursor))
        elif (numberOfUsersPerSubject > 0) and (numberOfUsersPerSubject <= len(clCursor)): # if number of users requested between 0 and max number of users for this subject
            selectedSubjects = np.random.choice(len(clCursor),numberOfUsersPerSubject,replace=False)
        elif (numberOfUsersPerSubject > 0) and (numberOfUsersPerSubject > len(clCursor)): # if for this segment not enough users are available, select all
            selectedSubjects = np.arange(len(clCursor))
        for iCl in clCursor[selectedSubjects]:  # loop over each classification in selectedSubjects, which is either all subjects or a randomly selected subset
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
        # if clIx % 200 == 0:
        #     print 100 * clIx / subjectCursor.count(), "%"
    # convert numbers to int (otherwise they're stored as object)
    cl = cl.convert_objects(convert_numeric=True)
    # normalise everything but nClassifications to value between 0 and 1
    # create cln, which is normalised version of cl
    cln = cl.copy()
    # normalise each of the columns starting at nCancer and all the ones to the right, using my own function def
    cln.loc[:,"nCancer":] = normalise_dataframe_by_ix(cl,cl.columns.get_loc("nClassifications"),range(cl.columns.get_loc("nCancer"),len(cl.columns)))
    # print "Done aggregating dataframe with classifications"
    return cln
def sj_in_expert_core(coreID):
    """
    Checks for a given core object ID whether it can be found in the expert scores. Used by classifications_dataframe_fill
    to save having to fill the cln dataframe with segments that do not belong to a core we have GS for.
    :param id: a core id that will be checked against coresGS
    :return: True or False
    """
    for ixGS,rowGS in coresGS.iterrows():
        # if the current row in coresGS is the one that matches the coreID, return True
        if str(int(rowGS.loc["Core ID"])) in coreID:
            return True
    # if you get to this part of the code, no match has been found in rowGS
    return False
def classifications_dataframe_save(cln,fn=classificationsDataframeFn):
    print "Saving dataframe to", fn
    cln.to_pickle(fn)
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
def plot_weighted_vs_unweighted_stain(cores):
    """Plots scores for aggregate staining intensity and proportion for weighted vs unweighted instance; weighting happens across subjects based on their proportion of people who said it contained cancer
    """
    plt.figure(1)
    plt.scatter(cores.aggregateIntensity,cores.aggregateIntensityWeighted)
    plt.xlabel("unweighted")
    plt.ylabel("weighted")
    plt.title("Aggregate Intensity")
    plt.figure(2)
    plt.scatter(cores.aggregateProp,cores.aggregatePropWeighted)
    plt.xlabel("unweighted")
    plt.ylabel("weighted")
    plt.title("Aggregate Proportion")
    plt.show()
def plot_user_vs_expert(cores):
    """Plot intensity, proportion and combined scores for experts against users, and print spearman
    :param cores: pandas dataframe
    :return: None
    """
    rhoProp,rhoIntensity,rhoSQS,rhoSQSadditive = user_vs_expert_rho(cores)
    plt.subplot(3,1,1)
    plt.scatter(cores.expProp,cores.aggregatePropWeighted)
    plt.xlabel("expert")
    plt.ylabel("user")
    plt.title("proportion stained, Spearman r = "+'{0:.2f}'.format(rhoProp))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(2)

    plt.subplot(3,1,2)
    plt.scatter(cores.expIntensity,cores.aggregateIntensityWeighted)
    plt.xlabel("expert")
    plt.ylabel("user")
    plt.title("intensity stain, Spearman r = "+'{0:.2f}'.format(rhoIntensity))
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(3,1,3)
    plt.scatter(cores.expSQS,cores.aggregateSQS)
    plt.xlabel("expert")
    plt.ylabel("user")
    plt.title("combined scores, Spearman r = "+'{0:.2f}'.format(rhoSQS))
    plt.gca().set_aspect('equal', adjustable='box')

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
def core_dataframe_fill(cln):
    """takes a dataframe cln (row=subject,column=properties of subject such as responses); aggregates all subjects for a single core into a single row in dataframe "cores"
    A subject with 20 classifications is weighted equally to subject with 150 classifications.
    For combining aggregrateProp and aggregateIntensity, the contributions from different subjects are weighted by the probability of users saying
    the subject was cancer. For example, a subject with nCancer = 0.3 will be weighted at half that of a subject with nCancer = 0.6

    Some cores may not have segments in cln because cln only contains segments that belong to cores that we have GS for. These cores
    will have nans as data.
    """
    # initialise new DataFrame that will have one row per core AND THE SAME COLUMNS AS
    cores = pd.DataFrame(data=None,index=range(0,len(get_core_ids(cln))),columns=cln.columns,dtype="float64")
    # put in core IDs
    cores["core"]=get_core_ids(cln)
    # get rid of unnecessary subjectID column
    cores = cores.drop("subjectID",axis=1)
    # add column indicating number of subjects for each core
    cores.insert(1,"nSubjects",np.nan)
    # add columns for weighted aggregates
    cores.insert(len(cores.columns),"aggregatePropWeighted",np.nan)
    cores.insert(len(cores.columns),"aggregateIntensityWeighted",np.nan)
    cores.insert(len(cores.columns),"aggregateSQS",np.nan)
    cores.insert(len(cores.columns),"aggregateSQSadditive",np.nan)
    # loop over each core
    for ix,core in cores.iterrows():
        coreRowsInCln = cln[cln["core"]==core.core]
        # note this takes the mean across all segments, including those that probably didn't have cancer.
        cores.loc[ix,"nClassifications":"aggregateIntensity"] = coreRowsInCln.loc[:,"nClassifications":"aggregateIntensity"].mean()
        # add weighted aggregate scores; multiply each subject score by nCancer, then normalise by nCancer.sum()
        cores.loc[ix,"aggregatePropWeighted"]       = (coreRowsInCln.aggregateProp      *coreRowsInCln.nCancer).sum() / coreRowsInCln.nCancer.sum()
        cores.loc[ix,"aggregateIntensityWeighted"]  = (coreRowsInCln.aggregateIntensity *coreRowsInCln.nCancer).sum() / coreRowsInCln.nCancer.sum()
        # store how many subjects were included for the core
        cores.loc[ix,"nSubjects"] = len(coreRowsInCln.index)
        cores.loc[ix,"aggregateSQS"] = cores.loc[ix,"aggregatePropWeighted"]*cores.loc[ix,"aggregateIntensityWeighted"]
        cores.loc[ix,"aggregateSQSadditive"] = percentage_to_category([cores.loc[ix,"aggregatePropWeighted"]]) + cores.loc[ix,"aggregateIntensityWeighted"]
    return cores
def core_dataframe_add_expert_scores(cores):
    """add expert scores and return the updated cores dataframe
    """
    # load in dataframe with columns
    # Core ID
    # % Positive
    # Intensity Score
    # SQS
    coresGS = pd.read_excel("GS_" + stain + ".xlsx")
    # add expert columns to dataframe
    cores.insert(len(cores.columns),"expProp",np.nan)
    cores.insert(len(cores.columns),"expIntensity",np.nan)
    cores.insert(len(cores.columns),"expSQS",np.nan)
    cores.insert(len(cores.columns),"expSQSadditive",np.nan)
    # loop over each row in the cores dataframe
    for ix,row in cores.iterrows():
        # find row in coresGS that matches current core
        for ixGS,rowGS in coresGS.iterrows():
            # if the current row in coresGS is the one that matches the row in cores, then store and break
            if str(int(rowGS.loc["Core ID"])) in row.core:
                cores.loc[ix,"expProp":"expSQS"] = np.array(rowGS["% Positive":"SQS"])
                cores.loc[ix,"expSQSadditive"] = cores.loc[ix,"expIntensity"] + percentage_to_category([cores.loc[ix,"expProp"]])
                # break out of the coresGS for loop
                break
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
def user_vs_expert_rho(cores):
    """Calculate Spearman rank correlation between expert and users
    The function of scipy's spearmanr is undefined for nan so should use Panda's correlatin method instead, which only uses pairwise complete to compute r.
    :param cores: pandas dataframe
    :return: rhoProp,rhoIntensity,rhoSQS,rhoSQSadditive
    """
    rhoProp =       cores['expProp'].corr(cores.aggregatePropWeighted,method='spearman')
    rhoIntensity =  cores['expIntensity'].corr(cores.aggregateIntensityWeighted,method='spearman')
    rhoSQS =        cores['expSQS'].corr(cores.aggregateSQS,method='spearman')
    rhoSQSadditive = cores['expSQSadditive'].corr(cores.aggregateSQSadditive,method='spearman')
    return rhoProp,rhoIntensity,rhoSQS,rhoSQSadditive
def percentage_to_category(percentages):
    """Takes percentage stained and transforms to categories, useful for calculating pseudo-allred
    :param percentages: numpy array or Pandas series with percentage cancer
    :return: categ: same dimension as 'percentages', transformed to groups, float
    """
    # set up dataframe with thresholds and categories. Percentages are inclusive. So if you're a 38, you'll be assigned the category which has the first value above 38 (e.g. 50)
    # transform = pd.DataFrame(data=np.array([[0,1,2,3,4,5],[0,1,10,30,60,100]]).T,columns=("category","percentage")) # THIS IS ER
    transform = pd.DataFrame(data=np.array([[0,1,2,3,4,5],[0,25,50,75,95,100]]).T,columns=("category","percentage")) # THIS IS RtO
    categ = np.zeros([len(percentages)])
    for iP in range(len(percentages)):
        # find lowest category that includes the percentage (e.g. 20 is smaller than 30, 60, and 100, so should fit in the 30 category)
        try: # can fail sometimes apparently
            categ[iP] = min([x["category"] for _,x in transform.iterrows() if percentages[iP] <= x["percentage"]])
        except:
            pass
    return categ
def plot_rho(rhoBoot):
    # set index as separate column
    toPlot = rhoBoot.copy()
    toPlot.reset_index(inplace=True)
    # toPlot.iloc[0:len(toPlot.index),1:].plot()
    toPlot.iloc[:,1:].plot()
    plt.xticks(np.arange(len(toPlot.index)),toPlot["index"])
    plt.xlabel("number of users per subject")
    plt.ylabel("Spearman r")
    plt.title("number of users/segment vs. accuracy")
    plt.draw()


########### FUNCTION EXECUTION
# set up pymongo objects
subjectsCollection, classifCollection, dbConnection = pymongo_connection_open()

# check if dataframe with classifications exists; if not, run over each classification and store its properties in a pandas dataframe. If it does exist, load it.
if os.path.isfile(classificationsDataframeFn) == False:
    cln = classifications_dataframe_fill(numberOfUsersPerSubject=0)
    classifications_dataframe_save(cln)
else:  # dataframe available, load instead of fill
    cln = classifications_dataframe_load(fn=classificationsDataframeFn)
# add columns to cln that indicates, for each subject, what the aggregate is of the multiple columns
cln = cln_add_columns_aggregating_stain(cln)
# aggregate data from multiple subjects into a single score for each core
cores = core_dataframe_fill(cln)
# load and add expert scores, add to the cores dataframe
cores = core_dataframe_add_expert_scores(cores)

# ######### Bootstrap number of users per segment
# # loop over all requested version of numberOfUsersPerSubject for samplesPerNumberOfUsers times
# rhoBoot = pd.DataFrame(data=np.nan,columns=("rhoProp","rhoIntensity","rhoSQS","rhoSQSadditive"),index=numberOfUsersPerSubject)
# rhoFull = np.zeros((len(numberOfUsersPerSubject),samplesPerNumberOfUsers,4))
# ix = 0
# for N in numberOfUsersPerSubject:
#     rho = np.zeros([samplesPerNumberOfUsers,4])
#     for iB in range(samplesPerNumberOfUsers):
#         t=time.time()
#         cln = classifications_dataframe_fill(numberOfUsersPerSubject=N)
#         cln = cln_add_columns_aggregating_stain(cln)
#         cores = core_dataframe_fill(cln)
#         cores = core_dataframe_add_expert_scores(cores)
#         rho[iB,:] = user_vs_expert_rho(cores)
#         print "doing",N,"users; completed",iB+1,"out of",samplesPerNumberOfUsers,'. Elapsed time for this bootstrap:',(time.time()-t),'seconds'
#         if N==0: # if all users are included, no need to run this bootstrap more than once; the answer will be the same anyway.
#             print "breaking from bootstrap iteration, once is enough when including all participants"
#             # set other lines to nan, otherwise mean is taken across zeros
#             rho[1:,:] = np.nan
#             break
#     # store all rank correlations in 3D matrix to be able to calculate intervals
#     rhoFull[ix,:,:] = rho
#     rhoBoot.loc[N,:] = np.nanmean(rho,axis=0)
#     # save intermediate results
#     np.save(stain+"bootstrap_full.npy",rhoFull)
#     rhoBoot.to_pickle(stain+"bootstrap.pkl")
#     ix += 1
# rhoBoot.to_pickle(stain+"bootstrap.pkl")
# np.save(stain+"bootstrap_full.npy",rhoFull)
# print rhoBoot
# plot_rho(rhoBoot)


# VARIOUS PLOTS
# plot_weighted_vs_unweighted_stain(cores)
plot_user_vs_expert(cores)

# close the connection to local mongoDB
pymongo_connection_close()
print "All done with script"
