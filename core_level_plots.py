""" Set of functions to plot core level stats.
Assumes user_aggregation.py has stored some data in a mongodb database.
Uses function from patient_level_lib.py to access these data
"""

from patient_level_lib import load_cores_into_pandas
import user_aggregation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_contribution_patterns():
    """ Irrespective of stain types, plot # of classifications over time as well as cumulative
        # http://stackoverflow.com/questions/3034162/plotting-a-cumulative-graph-of-python-datetimes
    :return: handle to figure
    """
    _,classifCollection,_ = user_aggregation.pymongo_connection_open()
    pmCursor = classifCollection.find({},projection={'_id':False,'updated_at':True}).limit(500)
    df = pd.DataFrame(list(pmCursor))
    counts = np.arange(0,len(df.index))
    plt.figure()
    plt.plot(df.updated_at,counts)
    plt.show()

