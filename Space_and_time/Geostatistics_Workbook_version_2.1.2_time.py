# 1. This notebook uses cross validation where train and test sets are chosen based on stations, which allows the estimation of CV-error reconstructing entire time series, not just filling in existing time series. The dropping of data not given an embay_dist for the summer average below (which may be temporary) ensures that all time series are interpolatable, so the interesting question is recreating an entirely missing time series (partial reconstruction can be assessed seperately).
#
#
# 2. In theory, alpha values still differ between continuous and discrete stations, but this time less so, because the discrete variation is just the variation that could be seen within one day. On top of this, there's a better arguement that the error between discrete stations is uncorrelated. However, in practice, to simplify the use of scikit's cross-validation function, we use a uniform alpha value (double the previous continuous alpha from Space model) to begin with.
#
#
# 3. (NOTE) "df" is the overall dataset, and remains a global variable throughout the duration of the notebook

# __Versions:__

# __v_2__

# In this variant of the notebook, we allowed the RBF kernels to be anisotropic, which resulted in a small increase in performance. Also we tried to assess the error only from the portion of the test set that is continuous, but the result is a much higher error for all predictions, implying that the model is in a sense overfit to the measurement dates of discrete data as opposed to capturing what is most likely to be the summer (July/August) mean. This is a result of the source of bias in estimating mean that was mentioned in the paper outline, namely: 
#
# > _"For bias, in addition to skewed time of day, because sampling dates at various STS stations are tied together, we must also consider the possibility that discrete STS data is systematically biased in any given year by sampling dates that are skewed towards or away from the warmest parts of the summer."_
#
# In the next variant of this workbook (v_3) we will implement a time variable (for month) in addition to the spatial variable to try to limit this type of bias, although this will be challenging as the model will still only operate within each given year.
#
# In any case, we now understand that through its incorporation of varying white noise, the RBF prediction actually functions better when it comes to accounting for the bias of discrete data, although this is to be expected because noise is less for the continuous measurement stations.

# __v_2.1__

# 1. We loop through cross validation training to get an accurate measure of error and bias to select a model for the dashboard
#
# 2. We update the monitoring location pixels when they were stuck on the border
#
# 3. We increase n_restarts optimizer from 9 to 15 to prevent lfbgs termination (for all models)
#
# 4. We increase length_scale upper bound to 1e6 (for 3 predictor RBF)

# __v_2.1.2__
#
# Uses new coastal feature file and allows URI/USGS data
#
# Also updates the continuous station selection to account for this
#
# Use nearest interpolation instead of manually adjusting bits to avoid boundary line

import pandas as pd
import numpy as np
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import cross_validate
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

paths={}
outputs={}

#TIFFs

#Uncropped Coastal Distance for merging with stations
paths[1]=config["Workbook_path1"]

#CSVs

#Aggregate Data of All Organizations for Testing
paths[2]="Data/Space_and_time_agg/agg_summer_means_daily_morning.csv"

#All stations/years with coastal features to merge with daily means
#no need to re-calc
paths[3]="Data/Space_agg/agg_summer_means_coastal_features_4_21_2021.csv"

#Ouput for Aggregate Data above but with embay_dist attached
outputs[1]="Data/Space_and_time_agg/agg_daily_morning_coastal_features_4_22_2023.csv"

for path in paths.values():
    assert(os.path.exists(path))

# +
#Global Params

folds=5

station_var=["Station ID"]

ind_var=["Longitude", "Latitude", "Day", "embay_dist"]

dep_var = ["Temperature (C)"]

predictors=len(ind_var)

years = [2019, 2020, 2021]

#Noise alpha can actual vary and may be reset below
noise_alpha=.2

lsb=(1e-5, 1e5)

#Values for different Length Scale Bounds to optimizew training time

lsb1=(1e-5, 1e5)
lsb2=(1, 1e5)

#n_restarts_optimizer
nro=15


# +
#Building Model (adapted to dashboard from "Geostatistics_Prediction_Dashboard_2.1.2.ipynb")
#If no noise is desired, set alpha=0

def build_model(predictors, year, kernel, alpha, nro):
    
    #Selecting data from proper year
    period=df.loc[df["Year"]==year, station_var+ind_var+dep_var]
    data=period.values

    #Designing an alpha matrix based on which means are from discrete data
    #orgs=df.loc[df["Year"]==year, "Organization"].values

    #applying mask
    #alpha=np.where(np.isin(orgs, cont_orgs), cont_error, discrete_error)
    
    #Partitioning X and y
    
    X_train, y_train = data[:, 1:predictors+1], data[:, -1]
    print(X_train.shape)
    
    #Ensuring proper dtypes
    X_train=np.array(X_train, dtype=np.float64)
    y_train=np.array(y_train, dtype=np.float64)

    #Demeaning y
    y_mean=np.mean(y_train)
    y_train=y_train-y_mean

    #Normalizing predictors (X) for training sets
    X_train=X_train - np.mean(X_train, axis=0)
    X_train=X_train / np.std(X_train, axis=0)

    #Constructing Process
    if (alpha>0):
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=nro, alpha=alpha)
    elif (alpha==0):
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=nro)
    else: raise ValueError("Alpha must be >=0")
    
    #Training Process
    gaussian_process.fit(X_train, y_train)

    return(gaussian_process, y_mean)


# -

# # Preparing Data (only needs to be run once)

#Reading in time data
df=pd.read_csv(paths[2], index_col=0)
df.head()

#Ensuring no reuse of station ID between orgs
assert(len(df.drop_duplicates(subset=["Station ID"]))==len(df.drop_duplicates(subset=["Station ID", "Organization"])))


#Reading in data to merge with "time" data
to_merge=pd.read_csv(paths[3], index_col=0)
to_merge.head()

df=df.merge(to_merge[["Station ID", "Year", 
                     "xPixel", "yPixel", "embay_dist"]], how="left",
           on = ["Station ID", "Year"])
df.head()

#Proportion of data not geotagged before
#should be minimal and can be dropped for now
print(len(df.loc[df["embay_dist"].isna(), "Station ID"])/len(df))
print(len(df))
df=df.loc[~df["embay_dist"].isna()].copy()
print(len(df))

#Outputting
df.to_csv(outputs[1])

# # Reading in data

#This is the source of the model's predictions
df=pd.read_csv(outputs[1], index_col=0)
df.reset_index(inplace=True, drop=True)
df.head()

#Checking Dominion Stations to Make Sure they are in Vaudrey Embayment (C and NB)
df.loc[df["Organization"]=="Dominion"]

#Dropping nas (need to un-hardcord)
print(len(df))
df.dropna(subset=["Station ID", "Longitude", "Latitude", "Day", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

# # Running cross validation

#Param
year=2019

# +
#Test of groupkfold
kfold=GroupKFold(n_splits=folds)
df_folds = kfold.split(df[ind_var], df[dep_var], df[station_var])

for i, (train_index, test_index) in enumerate(df_folds):
    print((len(train_index), len(test_index)))

# +
#With Noise CV
data=df.loc[df["Year"]==year, station_var+ind_var+dep_var].values

X, y = data[:, 1:predictors+1], data[:, -1]

kernel = 1 * RBF(length_scale=[1, 1, 1, 1], length_scale_bounds=(1e-5, 1e5))

gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=noise_alpha)

start_time=datetime.datetime.now()
CV = cross_validate(gaussian_process, X=X, y=y, groups=data[:, 0], scoring=("r2", "neg_root_mean_squared_error"), cv=GroupKFold(n_splits=folds), n_jobs=-1)
end_time=datetime.datetime.now()

print(end_time-start_time)
# -

#With noise
CV

# +
#Demeaned
data=df.loc[df["Year"]==year, station_var+ind_var+dep_var].values

X, y = data[:, 1:predictors+1], data[:, -1]

#De-meaning output
y -= np.mean(y)

kernel = 1 * RBF(length_scale=[1, 1, 1, 1], length_scale_bounds=(1e-5, 1e5))

gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=noise_alpha)

start_time=datetime.datetime.now()
CV_demeaned = cross_validate(gaussian_process, X=X, y=y, groups=data[:, 0], scoring=("r2", "neg_root_mean_squared_error"), cv=GroupKFold(n_splits=folds), n_jobs=-1)
end_time=datetime.datetime.now()

print(end_time-start_time)
# -

#CV for demeaned y
CV_demeaned

# # Building model w/ optimized hyperparameters

# This section uses the alpha parameter/kernel with the best result from Parameter_Optimization. See "root/Results" folder for reference

# ## 2019 (initial test)

# +
#Optimal params

#kernel
kernel =1*RBF([1]*predictors, length_scale_bounds=lsb) + 1*RationalQuadratic(length_scale_bounds=lsb)

#Noise alpha
noise_alpha=.2

#n_restarts_optimizer
nro=15
# -

#Building gp using build function above
opt_gp, ymean = build_model(predictors, 2019, kernel, noise_alpha, nro)

opt_gp.kernel_

#Quick test to compare to cross validated
opt_gp.predict([[-71.954278,41.277306, 182, 0.0]])

#This one has seperate length scale bounds for the RBF and Rational Quadratic
opt_gp2.kernel_

#Quick test to compare to cross validated
opt_gp2.predict([[-71.954278,41.277306, 182, 0.0]])

# ## All years

#Dictionaries for models
ymeans, gps = {}, {}

# +
#Optimal params
#years
years=[2020]

#kernel
kernel =1*RBF([1]*predictors, length_scale_bounds=lsb) + 1*RationalQuadratic(length_scale_bounds=lsb)

#Noise alpha
noise_alpha=.25
# -

#Building gp using build function above
for year in years:
    start_time=datetime.datetime.now()
    gps[year], ymeans[year] = build_model(predictors, year, kernel, noise_alpha, nro)
    end_time=datetime.datetime.now()
    print(year)
    print(end_time-start_time)

#Getting normalization factors as dictionary (to adjust length scales for comparison)
norms={}
for year in years:
    period=df.loc[df["Year"]==year, station_var+ind_var+dep_var]
    data=period.values

    X_train, y_train = data[:, 1:predictors+1], data[:, -1]

    #Ensuring proper dtypes
    X_train=np.array(X_train, dtype=np.float64)
    
    norms[year]=np.std(X_train, axis=0)
    
    print(np.std(X_train, axis=0))
    
    #Checking on input params
    check="Longitude"
    i=ind_var.index(check)
    plt.hist(X_train[:, i])
    plt.title(str(year) + " " + check + " Distribution")
    plt.show()
    print(max(X_train[:, i]))
    print(min(X_train[:, i]))

for year in years:
    params = gps[year].kernel_.get_params()
    #print(params)
    print(np.multiply(norms[year], params["k1__k2__length_scale"]))

gps[2019].kernel_.get_params()

gps[2020].kernel_.get_params()

gps[2021].kernel_.get_params()



# # Linear Regressions

plt.rcParams["figure.figsize"]=(20, 6)


def best_fit(df, indep, dep):
    #Fitting regression line
    reg_data=df[[indep, dep]]
    Y=reg_data[dep]
    X=sm.add_constant(reg_data[indep], prepend=False)
    mod = sm.OLS(Y, X)
    res=mod.fit()
    print(res.summary())
    
    #Plotting regression line on scatter
    plt.scatter(df[indep], df[dep])
    p=res.params
    plt.plot(df[indep], p.const+p[indep]*df[indep])
    plt.show()


#Scatterplot to check on relationship between temp and embayment distance without non-embayment data
plt.scatter(df["embay_dist"], df["Temperature (C)"])
plt.show()

# +
# #Dropping data not in embayments
# df=df.loc[df["embay_dist"]>0].copy(deep=True)
# plt.scatter(df["embay_dist"], df["Temperature (C)"])
# plt.show()
# -

#Listing data not in embayments to find multi-part embayments
df.loc[df["embay_dist"]==0]

#Dropping nas for temp
print(len(df))
df.dropna(subset="Temperature (C)", inplace=True)
print(len(df))

best_fit(df, "embay_dist", "Temperature (C)")

# Next we compare temp and lat

best_fit(df, "Latitude", "Temperature (C)")

# Next lon and temp

best_fit(df, "Longitude", "Temperature (C)")
