import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#Additional imports for distance calculations
from sklearn.metrics import DistanceMetric
from math import radians

#defining distance metric of choice
dist = DistanceMetric.get_metric('haversine')

#Reading in CT DEEP averages with fielpath for this program to be run elsewhere
assert(os.path.exists('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\CT_DEEP_means.csv'))
ct_means=pd.read_csv('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\CT_DEEP_means.csv', index_col=0)

#Creating col for coords in radians and setting index
ct_means["coords"]=ct_means.apply(lambda x: [radians(x["LatitudeMeasure"]), radians(x["LongitudeMeasure"])], axis=1)
ct_means.set_index("Year", inplace=True)
ct_means

#Defining function of interest (predictor assumed to be in lon/lat format with Longitude first!)
def predict(xtrain, ytrain, xtest, power):
    predictors=pd.DataFrame(data={"Longitude":xtrain[:, 0], "Latitude":xtrain[:, 1], "Temperature (C)": ytrain})
    predictors["Latitude"]=predictors["Latitude"].apply(radians)
    predictors["Longitude"]=predictors["Longitude"].apply(radians)
    
    predictees=pd.DataFrame(data={"Longitude":xtest[:, 0], "Latitude":xtest[:, 1]})
    predictees["Latitude"]=predictees["Latitude"].apply(radians)
    predictees["Longitude"]=predictees["Longitude"].apply(radians)
    
    distances=dist.pairwise(predictors[["Longitude", "Latitude"]].values, predictees[["Longitude", "Latitude"]].values)
    #print(distances)
    
    #Accounting for divide by zero by taking mean of first zero value
    if distances.min()>0:
        weights=1/distances
        weights=np.power(weights, power)
        
        print(weights.shape)
        
        normalize=np.diag(1/weights.sum(axis=0))
              
        print(normalize.shape)
        
        weights=np.matmul(weights, normalize)
        
        print(weights.shape)
        print(weights.sum(axis=0))
    else:
        return("Error: Divde by Zero")
    
    weighted=np.matmul(np.transpose(weights), predictors["Temperature (C)"].values)
    return(weighted)
