import numpy as np
import pandas as pd
import os
import json
from json import dumps
import matplotlib.pyplot as plt

#Additional imports for distance calculations
from sklearn.neighbors import DistanceMetric
from math import radians

#defining distance metric of choice
dist = DistanceMetric.get_metric('haversine')

#Reading in CT DEEP averages with fielpath for this program to be run elsewhere
assert(os.path.exists('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\CT_DEEP_means_4_5_2023.csv'))

ct_means=pd.read_csv('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\CT_DEEP_means_4_5_2023.csv', index_col=0)

#Creating col for coords in radians and setting index
ct_means["coords"]=ct_means.apply(lambda x: [radians(x["LatitudeMeasure"]), radians(x["LongitudeMeasure"])], axis=1)
ct_means.set_index("Year", inplace=True)
ct_means

#Defining function of interest
def ct_deep_idw(row, power):
    coord=row.loc["coords"]
    coord=[radians(deg) for deg in coord]
    
    year=int(row.loc["Year"])
    working=ct_means.loc[year].copy(deep=True)
    
    distances=working["coords"].apply(lambda x: (dist.pairwise([x, coord]))[0,1])
    #print(distances)
    
    #Accounting for divide by zero by taking mean of first zero value
    if min(distances)>0:
        weights=1/distances
        weights=np.power(weights, power)
        weights=weights/weights.sum()
    else:
        return("Error: Divde by Zero")
    
    weighted=np.dot(weights, working["temperature"])
    return(weighted)
