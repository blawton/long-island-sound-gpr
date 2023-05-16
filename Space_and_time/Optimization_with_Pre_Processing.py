# This notebook uses params determined from the param optimization notebook, but this time we test if normalization actually improves results. It's clear that normalization significantly improves the chances of LFBGS convergence, but we are interested in whether or not having the RationalQuadratic Kernel lenght scale on homoskedastic vs heteroskedastic data results in better prediction error (we subtract mean in either case as it should make no difference - in theory).

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import datetime
import time

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

#Setting working dir to root
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

#In this file, kernel, lsb, and noise_alpha are set
lsb=(1e-5, 1e5)

kernels=[1*RBF([1]*predictors, length_scale_bounds=lsb) + 1*RationalQuadratic(length_scale_bounds=lsb),
         1 * Matern([1]*predictors, nu=1.5, length_scale_bounds=lsb) + 1 * RationalQuadratic(length_scale_bounds=lsb),
         1 * Matern([1]*predictors, nu=.5, length_scale_bounds=lsb) + 1 * RBF([1]*predictors, length_scale_bounds=lsb)
        ]

noise_alpha=.25
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

# # Running cross validation on pipeline

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

gp = GaussianProcessRegressor(alpha=noise_alpha, n_restarts_optimizer=nro)

pipe = Pipeline([('scaler', StandardScaler()),
                 ('model', gp)])

parameters = {
    "model__kernel": kernels
}

clf = GridSearchCV(pipe, param_grid=parameters, cv=GroupKFold(n_splits=folds), n_jobs=-1, scoring=scores, refit=priority_score)
    
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
    pd.DataFrame(CV_results[year]).to_csv("Results/Optimization_with_Pre_Processing_results_" + str(year)+ ".csv")
    
    print(year)
# -

pd.DataFrame(CV_results[2020])

clf.predict([X[0,:]])


