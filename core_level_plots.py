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
    ax[1].set_xlim(left = min(df.updated_at),right=max(df.updated_at))
    ax[0].set_ylabel('daily contributions')
    ax[1].set_ylabel('cumulative contributions')
    ax[1].set_xlabel('date (month/year)')
    plt.show()
    return f,ax

f,ax = plot_contribution_patterns()

print "done with core_level_plots.py"