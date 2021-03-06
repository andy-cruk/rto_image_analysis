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

# settings
aggregate = ['ignoring_segments', 'segment_aggregation']
aggregate = aggregate[0]

def plot_contribution_patterns():
    """ Irrespective of stain types, plot # of classifications over time as well as cumulative
        http://stackoverflow.com/questions/3034162/plotting-a-cumulative-graph-of-python-datetimes
        http://stackoverflow.com/questions/29672375/histogram-in-matplotlib-time-on-x-axis
    :return: figure handle,axis handles
    """
    _,classifCollection,_ = pymongo_connection_open()
    pmCursor = classifCollection.find({}, projection={'_id': False,'updated_at': True})
    assert pmCursor.count() > 0
    # list comprehension to end up with a list of datetimes
    dat = [x['updated_at'] for x in list(pmCursor)]
    # sort the list
    dat.sort()
    counts = np.arange(0,len(dat))
    f,ax = plt.subplots(nrows=2, sharex=True)
    delta = max(dat)-min(dat)
    ax[0].hist([dates.date2num(y) for y in dat], bins=delta.days, cumulative=False, histtype='step', log=True)
    ax[1].plot(dat, counts)
    ax[1].xaxis.set_major_formatter(dates.DateFormatter('%m/%y'))
    ax[1].xaxis.set_major_locator(dates.MonthLocator(bymonth=range(1,13,1), bymonthday=1))
    # ax[1].xaxis.set_minor_locator(dates.MonthLocator(bymonth=range(1,13), bymonthday=1))
    ax[1].set_xlim(left = datetime.date(2014, 10, 1), right=datetime.date(2016, 9, 28))
    ax[0].set_ylim(bottom=200, top=350000)
    # ax[0].minorticks_on()
    ax[0].grid(b=True, axis='both', which='major', color='k', linestyle='-')
    ax[1].grid(b=True, axis='x', which='major', color='k', linestyle='-')
    # ax[0].grid(b=True, axis='y', which='minor', color='k', linestyle='--')
    ax[0].set_ylabel('daily contributions')
    ax[1].set_ylabel('cumulative contributions')
    ax[1].set_xlabel('date (month/year)')
    plt.show()
    return f, ax


def scatter_for_each_stain(xdat=aggregate+'.expSQS', ydat=aggregate+'.aggregateSQSCorrected', correlation='spearman'):
    """ Takes two measures from the cores database and for each stain, scatters them against one another
    :param xdat: string pointing to column in cores dataset, laid out along x-axis
    :param ydat: see xdat, but along y-axis
    :param correlation: type of correlation to compute through pd.DataFrame.corr function ('spearman','pearson','kendall')
    :return: f,ax handles
    """

    df = load_cores_into_pandas()
    # ndarray of strings
    stains = list(df.stain.unique())
    if 'test mre11' in stains:
        stains.remove('test mre11')
    # make square set of subplots
    ncol = 3
    nrow = int(np.ceil(len(stains)/ncol))
    fig, axes = plt.subplots(nrows=nrow,ncols=ncol,sharex=True,sharey=True)
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


def scatter_all_stains_together(xdat=aggregate+'.expSQS', ydat=aggregate+'.aggregateSQSCorrected', correlation='spearman'):
    """ Takes two measures from the cores database and for each stain, scatters them against one another
    :param xdat: string pointing to column in cores dataset, laid out along x-axis
    :param ydat: see xdat, but along y-axis
    :param correlation: type of correlation to compute through pd.DataFrame.corr function ('spearman','pearson','kendall')
    :return: f,ax handles
    """

    df = load_cores_into_pandas()
    # ndarray of strings
    stains = df.stain.unique()
    fig, ax = plt.subplots(1)
    rows = ~np.isnan(df[xdat]) & ~np.isnan(df[ydat])
    x = df.loc[rows, xdat]
    y = df.loc[rows, ydat]
    ax.scatter(x, y, c='black', alpha=0.2)
    ax.set_title('All analysed cores')
    # add Spearman correlation
    r = x.corr(y,method=correlation)
    ax.annotate(correlation + ' r = '+"{:.2f}".format(r), textcoords='axes fraction', xy=(0.2,0.05), xytext=(0.2,0.05))
    # add labels
    ax.set_xlabel(xdat)
    ax.set_ylabel(ydat)
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x))

    plt.show()
    return fig, ax


def scatter_performance_single_graph(xcorr=(aggregate+'.expProp', aggregate+'.aggregatePropCorrected'), ycorr=(aggregate+'.expIntensity', aggregate+'.aggregateIntensityCorrected'), xmethod=stats.spearmanr, ymethod=qwk.quadratic_weighted_kappa):
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


def plot_number_of_classifications_against_performance_for_multiple_stains_in_single_graph(mongoFilter={'aggregate':aggregate}, measure='Hscore', nUsersPerSubject=range(1,50), addCI=True):
    """
    """
    # load bootstraps
    db = MongoClient("localhost", 27017)
    coll = db.results.bootstraps
    results = coll.find(filter=mongoFilter)
    df = pd.DataFrame(list(results))
    # array of strings
    stains = df.stain.unique()
    # set up figure
    f, ax = plt.subplots(1)
    # set up offset for the stains. Should be proportional?
    offsetRange = np.array([0.9, 1.1])
    offsets = np.linspace(offsetRange[0], offsetRange[1], len(stains))
    # get average number of classifications per subject/core
    foo = list(db.results.cores.aggregate([
        {"$group":
            {
                "_id": "$stain",
                "nUsers": {"$avg": '$'+aggregate+'.nClassificationsTotal'}
            }
        }]))
    # transform to sensible dict {'mre11': n, 'p53': n2} etc
    avgNperStain = {}
    for item in foo:
        stain = item['_id']
        avgNperStain[stain] = item['nUsers']
    #### loop over each stain and plot
    # set color for each line
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(stains))))
    for iStain, stain in enumerate(stains):
        # calculate mean and CI for each requested nUsersPerSubject for this stain
        means = np.zeros(len(nUsersPerSubject))*np.nan
        CI = np.zeros((2,len(nUsersPerSubject)))*np.nan
        for iN, N in enumerate(nUsersPerSubject):
            # first check whether this stain has, on average, enough subjects looking at each image.
            # If average number of users per image is lower than N, it should not show the bootstrap
            # as it doesn't make sense (imagine pretending to have 1000 users when sampling from a pool of 20 actual users)
            if avgNperStain[stain] < N:
                continue

            # get vector with bootstrapped data for this stain, number of users and measure
            # return iN,N,means,CI,df,stain,measure
            dat = df.loc[(df.nUsersPerSubject==N) & (df.stain==stain),measure]
            if len(dat) > 100:  # if sufficient records found
                means[iN] = dat.mean()
                CI[0, iN] = np.nanpercentile(dat, 2.5)-means[iN]
                CI[1, iN] = np.nanpercentile(dat, 97.5)-means[iN]
            else:
                means[iN] = np.nan
                CI[:, iN] = np.nan

        # Set color for this line
        c = next(color)
        # plot this stain into graph with or without error bar
        if addCI:
            ax.errorbar(x=nUsersPerSubject*offsets[iStain], y=means, yerr=np.abs(CI), label=stain, c=c)
        else:
            ax.plot(nUsersPerSubject*offsets[iStain], means, label=stain, c=c)
        # add scatter to put little dots where means are
        ax.scatter(nUsersPerSubject*offsets[iStain],means)
    ax.set_xlabel('number of users included')
    ax.set_ylabel('accuracy on task (' + measure + ')')
    # ax.set_xlim(left = min(nUsersPerSubject)-1, right=max(nUsersPerSubject)+10)
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0.8)
    ax.grid(b=True, which='both', axis='y')
    ax.set_xscale('log')
    ax.legend(loc=4)

    plt.show()
    return f, ax


def create_table_summary_stats_each_stain(aggregate=aggregate):
    """ This will create an excel spreadsheet with text ready to be pasted into the paper as results table 2.
    Will have one row per stain, and one column for each measure that is correlated between expert and users.
    The easiest thing seems to be to write a pandas dataframe with strings indicating the calculated values

    :param aggregate: which data to retrieve from the cores collection
    """
    # set up a list of tuples that contain (userMeasure, expMeasure, function) which will calculate FUNCTION on the
    # userMeasure and expMeasure. first output of function needs to be what is used (e.g. r rather than p)

    output = [
        ('aggregateSQSCorrected', 'expSQS', stats.spearmanr),
        ('aggregatePropCorrected', 'expProp', stats.spearmanr),
        ('aggregateIntensityCorrected', 'expIntensity', qwk.quadratic_weighted_kappa)
            ]
    # load cores and remove aggregate prefix
    df = load_cores_into_pandas(projection={'_id': False, aggregate: True})
    df.columns = [s.replace(aggregate+'.','') for s in df.columns]

    # prepare dataframe that will be saved to excel
    stains = df.stain.unique()
    dfout = pd.DataFrame(data=None, index=stains, columns=[s[0] for s in output])

    # loop over each and store the function value + bootstrap CI
    for stain in stains:  # for each stain
        for iM in range(len(output)):  # for each measure
            user, exp, fun = output[iM]
            user = df.loc[df.stain == stain, user]
            exp = df.loc[df.stain == stain, exp]
            isFin = np.isfinite(user) & np.isfinite(exp)
            user = user[isFin]
            exp = exp[isFin]
            # get first return value from function
            if fun.__name__ == 'spearmanr':
                score = fun(user, exp)[0]
                ci = bootstrap.ci((user, exp), statfunction=fun, method='pi')[:, 0]
            elif fun.__name__ == 'quadratic_weighted_kappa':
                # round score to make integers
                user = np.round(user).astype(int)
                exp = np.round(exp).astype(int)
                score = fun(user, exp)
                ci = bootstrap.ci((user, exp), statfunction=fun, method='pi')
            else:
                raise Exception('no bootstrap defined for this error function')
            # http://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points
            dfout.loc[stain, output[iM][0]] = "{0:.2f}".format(score) + ' (' + "{0:.2f}".format(ci[0]) + ', ' + "{0:.2f}".format(ci[1]) + ')'
    print(dfout)
    dfout.to_excel('results/summary_stats_'+aggregate+'.xlsx')


if __name__ == "__main__":
    # f, ax = plot_contribution_patterns()
    # f, ax = scatter_for_each_stain()
    # f, ax = scatter_all_stains_together()
    # r, ci, f, ax = scatter_performance_single_graph()
    # f, ax = plot_number_of_classifications_against_performance_for_multiple_stains_in_single_graph(
    #   nUsersPerSubject=np.array([1,2,4,8,16,32,64,128,256,512,1024]))
    create_table_summary_stats_each_stain()

    # f.savefig('C:\\Users\\Peter\\Desktop\\figure.eps', format='eps')
    print("done with core_level_plots.py")