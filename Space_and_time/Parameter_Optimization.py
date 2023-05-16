# 1. This notebook uses cross validation where train and test sets are chosen based on stations, which allows the estimation of CV-error reconstructing entire time series, not just filling in existing time series. The dropping of data not given an embay_dist for the summer average below (which may be temporary) ensures that all time series are interpolatable, so the interesting question is recreating an entirely missing time series (partial reconstruction can be assessed seperately).
#
#
# 2. In theory, alpha values still differ between continuous and discrete stations, but this time less so, because the discrete variation is just the variation that could be seen within one day. On top of this, there's a better arguement that the error between discrete stations is uncorrelated. However, in practice, to simplify the use of scikit's cross-validation function, we use a uniform alpha value (double the previous continuous alpha from Space model) to begin with.
#
#
# 3. (NOTE) "df" is the overall dataset, and remains a global variable throughout the duration of the notebook. The code assumes that the variable for year of a measurement is labelled as "Year", otherwise labels should match the variable names below
#
#
# 4. The 2020 results in the first stage of param optimization were skewed by a positive longitude issue, but it is fairly clear that .25 degrees of noise is optimal, so use that param in pre-processing optimization regardless

import pandas as pd
import numpy as np
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import datetime
import time

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

#Working dir
if(os.path.basename(os.getcwd())[0:18]!="Data Visualization"):
    os.chdir("..")
assert(os.path.basename(os.getcwd())[0:18]=="Data Visualization")

# +
#Loading paths config.yml
import yaml

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Paths for loading data
import os

inputs={}

#Input of Aggregate Data with embay_dist attached (output_1 of Geostatistics_Workbook..._time)
inputs[1]="Data/Space_and_time_agg/agg_daily_morning_coastal_features_4_22_2023.csv"

for path in inputs.values():
    assert(os.path.exists(path))

# +
#Global Params

folds=5

years=[2019, 2020, 2021]

station_var=["Station ID"]

ind_var=["Longitude", "Latitude", "Day", "embay_dist"]

dep_var = ["Temperature (C)"]

predictors=len(ind_var)

#scores to include in CV scoring (as tuple):
scores = ("r2", "neg_root_mean_squared_error")

#score for refitting
priority_score="neg_root_mean_squared_error"

#Restarts in optimizer
nro=15

# +
#Parameters to test

#Length scale bounds (uniform for all kernels)
lsb1=(1e-5, 1e5)
lsb2=(1, 1e5)

#Noise
alphas=[.2]

#Alpha is loaded into first kernel for now
kernels = [1*RBF([1]*predictors, length_scale_bounds=lsb1) + 1*RationalQuadratic(length_scale_bounds=lsb2)]
# -

# # Reading in data

#This is the source of the model's predictions
df=pd.read_csv(inputs[1], index_col=0)
df.reset_index(inplace=True, drop=True)
df.head()

#Checking Dominion Stations to Make Sure they are in Vaudrey Embayment (C and NB)
df.loc[df["Organization"]=="Dominion"]

#Dropping nas (need to un-hardcord)
print(len(df))
df.dropna(subset=["Station ID", "Longitude", "Latitude", "Day", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

# # Running cross validation

# In this notebook, it has been decided to run optimization with demeaned data based on previous observations of "Geostatistics_Workbook_version_2.1.2_time". This should not affect the rmse, and r^2 values can be interpreted accordingly

# +
#Test of groupkfold
kfold=GroupKFold(n_splits=folds)
df_folds = kfold.split(df[ind_var], df[dep_var], df[station_var])

for i, (train_index, test_index) in enumerate(df_folds):
    print((len(train_index), len(test_index)))

# +
#Block for iteration through years
CV_results={}
CV_best_estimators={}

parameters = {'kernel': kernels,
              'alpha': alphas
             }
gp = GaussianProcessRegressor(n_restarts_optimizer=nro)

clf = GridSearchCV(gp, parameters, cv=GroupKFold(n_splits=folds), n_jobs=-1, scoring=scores, refit=priority_score)
    
for year in years:
    #Selecting year
    data = df.loc[df["Year"]==year, station_var+ind_var+dep_var].values
    
    #Splitting into groups, indep, and dep vars
    groups, X, y = data[:, 0], data[:, 1:predictors+1], data[:, -1]

    #De-meaning output
    y -= np.mean(y)
    
    #Fitting cross validation
    clf.fit(X, y, groups=groups)
    
    CV_results[year] = clf.cv_results_
    CV_best_estimators[year] =clf.best_estimator_
    
    #Saving to file
    pd.DataFrame(CV_results[year]).to_csv("Results/Parameter_Optimization_time_results" + str(year)+ "_II.csv")
    
    print(year)
# -

pd.DataFrame(CV_results[2020])

clf.predict([X[0,:]])


