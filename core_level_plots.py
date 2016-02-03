""" Set of functions to plot core level stats.
Assumes user_aggregation.py has stored some data in a mongodb database.
Uses function from patient_level_lib.py to access these data
"""

from patient_level_lib import load_cores_into_pandas
from user_aggregation import pymongo_connection_open
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scikits import bootstrap
import quadratic_weighted_kappa as qwk
from pymongo import MongoClient

def plot_contribution_patterns():
    """ Irrespective of stain types, plot # of classifications over time as well as cumulative
        http://stackoverflow.com/questions/3034162/plotting-a-cumulative-graph-of-python-datetimes
        http://stackoverflow.com/questions/29672375/histogram-in-matplotlib-time-on-x-axis
    :return: figure handle,axis handles
    """
    _,classifCollection,_ = pymongo_connection_open()
    pmCursor = classifCollection.find({},projection={'_id':False,'updated_at':True})
    df = pd.DataFrame(list(pmCursor))
    counts = np.arange(0,len(df.index))
    f,ax = plt.subplots(nrows=2,sharex=True)
    delta = max(df.updated_at)-min(df.updated_at)
    ax[0].hist([dates.date2num(y) for y in df.updated_at],bins=delta.days,cumulative=False,histtype='step',log=True)
    ax[1].plot(df.updated_at,counts)
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%m/%y'))
    ax[1].xaxis.set_minor_locator(dates.MonthLocator)
    ax[1].set_xlim(left = datetime.date(2014,9,15),right=max(df.updated_at))
    ax[0].set_ylabel('daily contributions')
    ax[1].set_ylabel('cumulative contributions')
    ax[1].set_xlabel('date (month/year)')
    plt.show()
    return f,ax


def scatter_for_each_stain(xdat='expSQS',ydat='aggregateSQSCorrected',correlation='spearman'):
    """ Takes two measures from the cores database and for each stain, scatters them against one another
    :param xdat: string pointing to column in cores dataset, laid out along x-axis
    :param ydat: see xdat, but along y-axis
    :param correlation: type of correlation to compute through pd.DataFrame.corr function ('spearman','pearson','kendall')
    :return: f,ax handles
    """

    df = load_cores_into_pandas()
    # ndarray of strings
    stains = df.stain.unique()
    # assume 3 columns for subplot
    size = np.ceil(np.sqrt(len(stains))).astype(int)
    fig,axes = plt.subplots(nrows=size,ncols=size,sharex=True,sharey=True)
    axes = axes.flatten()
    for ix,stain in enumerate(stains):
        # set current axis
        ax = axes[ix]
        # get rows for this stain
        rows = (df.stain == stain) & (~np.isnan(df[xdat])) & (~np.isnan(df[ydat]))
        x = df.loc[rows,xdat]
        y = df.loc[rows,ydat]
        ax.scatter(x,y,c='black')
        ax.set_title(stain)
        # add Spearman correlation
        r = x.corr(y,method=correlation)
        ax.annotate(correlation + ' r = '+"{:.2f}".format(r),textcoords='axes fraction',xy=(0.2,0.05),xytext=(0.2,0.05))
        # add labels
        ax.set_xlabel(xdat)
        ax.set_ylabel(ydat)
        # increment counter
        ix += 1

    plt.show()
    return fig,axes


def scatter_performance_single_graph(xcorr=('expProp','aggregatePropCorrected'),ycorr=('expIntensity','aggregateIntensityCorrected'),xmethod=stats.spearmanr,ymethod=qwk.quadratic_weighted_kappa):
    """Create single scatterplot with one point per stain. Location on x is set by xmethod on the data in xcorr, ditto for y. Semi-flexible in terms of methods, though might
    need some tweaking to get it to work given different variables returned by different methods (e.g. with/without p-value).
    Plots 95% CI bootstrapped
    :param xcorr: list containing two strings, each pointing to a column in the cores dataframe. Relationship between them will be calculated using the xmethod function, and displayed along x-axis
    :param ycorr: see xcorr, but displayed along y.
    :param xmethod: function that returns two variables, first one of interest (e.g. r,p for spearman)
    :param ymethod: either quadratic_weighted_kappa that returns single value, or another function that returns like spearman (i.e. tuple with only first returned value of interest
    :return: (r (nStain*{x,y}), ci (nStain*{x,y}*{LB,UB}), f, ax)
    """

    df = load_cores_into_pandas()
    # ndarray of strings
    stains = df.stain.unique()
    # loop over each stain and calculate value and 95% CI for x and y
    r = np.array(np.zeros([len(stains),2]))*np.nan # initialise r: nStains * (x,y)
    ci = np.array(np.zeros([len(stains),2,2]))*np.nan # initialise r: nStains * (x,y) * (lower,upper)
    for ix,stain in enumerate(stains):
        # get data to calculate correspondence for the x-axis
        x = (df.stain == stain) & (~np.isnan(df[xcorr[0]])) & (~np.isnan(df[xcorr[1]]))
        xx = df.loc[x,xcorr[0]]
        xy = df.loc[x,xcorr[1]]
        # now for y
        y = (df.stain == stain) & (~np.isnan(df[ycorr[0]])) & (~np.isnan(df[ycorr[1]]))
        yx = df.loc[y, ycorr[0]]
        yy = df.loc[y, ycorr[1]]

        # calculate mean and 95% CI for x. Assumes a function like spearmanr which returns r first, p value second
        r[ix,0],_ = xmethod(xx, xy)
        ci[ix,0,:] = bootstrap.ci((xx, xy), xmethod)[:,0]


        # if weighted kappa is used then you need integers
        if ymethod is qwk.quadratic_weighted_kappa:
            yx = np.round(yx).astype(int)
            yy = np.round(yy).astype(int)
            r[ix,1] = ymethod(yx, yy)
            ci[ix,1,:] = bootstrap.ci((yx, yy), ymethod)
        else: #probably spearman, which returns a second argument (hence the ,_ and [:,0]
            r[ix,1],_ = ymethod(yx, yy)
            ci[ix,1,:] = bootstrap.ci((yx, yy), ymethod)[:,0]
    f,ax = plt.subplots(1)
    ax.errorbar(
            r[:,0],
            r[:,1],
            xerr=np.abs((np.tile(r[:,0],[2,1]).T - ci[:,0,:])).T,
            yerr=np.abs((np.tile(r[:,1],[2,1]).T - ci[:,1,:])).T,
            fmt='.k'
    )
    for i,stain in enumerate(stains):
        ax.annotate(stain,(r[i,0],r[i,1]))
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlabel(xcorr[0]+' vs '+xcorr[1]+', '+xmethod.__name__)
    ax.set_ylabel(ycorr[0]+' vs '+ycorr[1]+', '+ymethod.__name__)
    plt.show()
    return r,ci,f,ax


def plot_number_of_classifications_against_performance_for_multiple_stains_in_single_graph(mongoFilter={},measure='Hscore',nUsersPerSubject=range(1,5),CI=True,limit=100):
    """

    """
    # load bootstraps
    db = MongoClient("localhost", 27017)
    coll = db.results.bootstraps
    results = coll.find(filter=mongoFilter)
    df = pd.DataFrame(list(results))

    # ndarray of strings
    stains = df.stain.unique()
    # set up figure
    f,ax = plt.subplots(1)
    # loop over each stain and plot
    for iStain,stain in enumerate(stains):
        # calculate mean and CI for each requested nUsersPerSubject for this stain
        means = np.zeros(len(nUsersPerSubject))*np.nan
        CI = np.zeros((2,len(nUsersPerSubject)))*np.nan
        for iN,N in enumerate(nUsersPerSubject):
            # get vector with bootstrapped data for this stain, number of users and measure
            dat = df.loc[df.nUsersPerSubject==N & df.stain==stain,measure]
            # limit the datapoints used, not in random fashion so it's replicable
            dat = dat.iloc[:limit,:]
            means[iN] = dat.mean
            CI[0,iN] = np.percentile(dat,2.5)-means[iN]
            CI[1,iN] = np.percentile(dat,97.5)-means[iN]
        # plot this stain into graph with or without error bar
        if CI:
            ax.errorbar(x=nUsersPerSubject,y=means,yerr=np.abs(CI),label=stain)
        else:
            ax.plot(nUsersPerSubject,means,label=stain)
    ax.set_xlabel('number of users included per segment')
    ax.set_ylabel(measure)
    ax.legend()

    plt.show()
    return f,ax


if __name__ == "__main__":
    # f,ax = plot_contribution_patterns()
    # f,ax = scatter_for_each_stain()
    # r,ci,f,ax = scatter_performance_single_graph()
    f,ax = plot_number_of_classifications_against_performance_for_multiple_stains_in_single_graph()

    print "done with core_level_plots.py"