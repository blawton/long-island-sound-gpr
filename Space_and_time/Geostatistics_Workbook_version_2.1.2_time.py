import pandas as pd
import numpy as np
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# ENSURE STATION ID IS UNIQUE FOR IMPORTED SUMMER MEANS

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
paths[2]="Data/Space_and_time_agg/agg_summer_means_daily.csv"

#All stations/years with coastal features to merge with daily means
#no need to re-calc
paths[3]="Data/Space_agg/agg_summer_means_coastal_features_4_21_2021.csv"

#Ouput for Aggregate Data above but with embay_dist attached
outputs[1]="Data/Space_and_time_agg/agg_daily_coastal_features_4_22_2023.csv"

for path in paths.values():
    assert(os.path.exists(path))
# -

# # Preparing and Testing Data

#Reading in time data
df=pd.read_csv(paths[2], index_col=0)
df.head()

#Reading in data to merge with "time" data
to_merge=pd.read_csv(paths[3], index_col=0)
to_merge.head()

df=df.merge(to_merge[["Station ID", "Year", 
                     "xPixel", "yPixel", "embay_dist"]], how="left",
           on = ["Station ID", "Year"])
df.head()

#Proportion of datavnot geotagged before
#should be minimal and can be dropped for now
print(len(df.loc[df["embay_dist"].isna(), "Station ID"])/len(df))
print(len(df))
df=df.loc[~df["embay_dist"].isna()].copy()
print(len(df))

#Outputting
df.to_csv(outputs[1])

# ## Linear Regressions

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

# # GPR with Noise

# 1. In the time version of this notebook, we will begin modelling very little noise, assuming all data has the same noise used for the continuous data in the last version of this notebook (var=.1)

# 2. We can test different noise parameters for what is now cross validation on rmse prediction of time series

# 3. The "philosophy" of adding in time as a predictor is to change as little as possible in the code, while making use of the new "Day" variable from the new input.

# ## Preparing Data

# We prepare the data the same for the two parameter and three parameter regressions (which are the same code with the variable "predictors" changed from 2 to 3) so that we can compare the rmse on the same cross validation folds to try eliminating chance from the comparison as much as possible

#Dropping nas
print(len(df))
df.dropna(subset=["Station ID", "Longitude", "Latitude", "Day", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

#Getting list of continuous station IDs for testing
cont_st=df.loc[(df["Organization"]=="STS_Tier_II") | (df["Organization"]=="EPA_FISM") | (df["Organization"]=="USGS_Cont"), "Station ID"]
cont_st


#Function for iteration
def cross_validate(predictors, folds, year, kernel, noise, trials):
    
    #Making df for results
    results=pd.DataFrame(columns=["rmse", "bias"])
    
    #Selecting data from proper year
    period=df.loc[df["Year"]==year, ["Station ID", "Longitude", "Latitude", "embay_dist", "Temperature (C)"]]
    data=period.values
    k=folds
    foldsize = int(len(data)/k)+1
    
    #Designing an alpha matrix based on which means are from discrete data
    orgs=df.loc[df["Year"]==year, "Organization"].values

    #applying mask (DOESN'T WORK, BORROW FROM OTHER WORKBOOK)
    alpha=np.where(np.isin(orgs, cont_st), .1, 1.44)

    for j in range(trials):    

        #Randomly selecting training set
        np.random.shuffle(data)
        test_sets={}
        train_sets={}
        alphas={}

        for n in range(k):
            if n!=(k-1):
                test_ind=range(n*foldsize, (n+1)*foldsize)
                test_sets[n]=data[test_ind, :]
                train_ind=[i for i in range(len(data)) if i not in test_ind]
                train_sets[n]=data[train_ind, :]

                #alphas
                alphas[n]=alpha[train_ind]

            else:
                test_ind=range(n*foldsize, len(data))
                test_sets[n]=data[test_ind, :]
                train_ind=[i for i in range(len(data)) if i not in test_ind]
                train_sets[n]=data[train_ind, :]

                #alphas
                alphas[n]=alpha[train_ind]

        print(len(train_sets[2]))
        print(len(test_sets[9]))

        #Partitioning X and y (This time predictors are normalized)
        X_train={}
        y_train={}
        X_test={}
        y_means={}

        for i in range(k):
            X_train[i], y_train[i] = train_sets[i][:, 1:predictors+1], train_sets[i][:, -1]
            X_test[i] = test_sets[i][:, 1:predictors + 1]

            #Ensuring proper dtypes
            X_train[i]=np.array(X_train[i], dtype=np.float64)
            y_train[i]=np.array(y_train[i], dtype=np.float64)
            X_test[i]=np.array(X_test[i], dtype=np.float64)

            #Demeaning y
            y_means[i]=np.mean(y_train[i])
            y_train[i]=y_train[i]-y_means[i]

            #Normalizing predictors (X) for training sets
            #X_test[i]=X_test[i] - np.mean(X_test[i], axis=0)
            #X_test[i]=X_test[i] / np.std(X_test[i], axis=0)

            #Normalizing predictors (X) for testing sets
            #X_train[i]=X_train[i] - np.mean(X_train[i], axis=0)
            #X_train[i]=X_train[i] / np.std(X_train[i], axis=0)

        #Running model on each fold
        cross_v={}
        rmse={}
        bias={}

        for i in range(k):
            #print(X_train[i].shape)
            #print(y_train[i].shape)
            
            #Constructing Process
            if noise:
                gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=alphas[i])
            else:
                gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
            
            #Training Process
            gaussian_process.fit(X_train[i], y_train[i])
            print(gaussian_process.kernel_)

            #Predicting for test data
            y_pred, MSE = gaussian_process.predict(X_test[i], return_std=True)

            #Re-adding mean of training data
            y_pred += y_means[i]

            #Concatenating Predictions to Observed
            cross_v[i]=pd.DataFrame(np.append(test_sets[i], np.transpose([y_pred]), axis=1), columns=list(period.columns) + ["Predicted"])

            #Restricting to continuous data to limit error
            working=cross_v[i]
            working=working.loc[working["Station ID"].isin(cont_st)]
            cross_v[i]=working

            #Calculating RMSE and Bias
            if len(cross_v[i])>0:
                cross_v[i]["Bias"]=cross_v[i]["Predicted"]-cross_v[i]["Temperature (C)"]
                cross_v[i]["Squared Error"]=np.square(cross_v[i]["Bias"])

                rmse[i]=np.sqrt(np.sum(cross_v[i]["Squared Error"])/len(cross_v[i]))
                bias[i]=np.mean(cross_v[i]["Bias"])
            else:
                rmse[i]=np.nan
                bias[i]=np.nan
        
        current_result=pd.DataFrame(data={"rmse":[np.nanmean(list(rmse.values()))], "bias": [np.nanmean(list(bias.values()))]})
        results=pd.concat([results, current_result], ignore_index=True)

    return(results)


# +
#Testing loop
kernel = 1 * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-5, 1e2))

results=cross_validate(2, 10, 2019, kernel, True, 10)
results
# -

print(results.mean(axis=0))

# # Actual Testing

years=[2019, 2020, 2021]
folds=10
trials=10

# ## RBF 2 predictors

# +
predictors = 2
kernel = 1 * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-5, 1e2))
noise=True

agg=pd.DataFrame()

for year in years:
    results=cross_validate(predictors, folds, year, kernel, noise, trials)
    agg=pd.concat([agg, results])
    print(agg)
# -

print(agg.mean(axis=0))

# ## Rational Quadratic 2 predictors

# +
predictors = 2
kernel = 1 * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-5, 1e2))
noise = False

agg=pd.DataFrame()

for year in years:
    results=cross_validate(predictors, folds, year, kernel, noise, trials)
    agg=pd.concat([agg, results])
    print(agg)
# -

print(agg.mean(axis=0))

# ## RBF 3 predictors

# +
predictors = 3
kernel = 1 * RBF(length_scale=[1e-1, 1e-1, 1e-2], length_scale_bounds=(1e-5, 1e5))
noise = True

agg=pd.DataFrame()

for year in years:
    results=cross_validate(predictors, folds, year, kernel, noise, trials)
    agg=pd.concat([agg, results])
    print(agg)
# -

print(agg.mean(axis=0))

# ## Rational Quadratic 3 predictors

# +
predictors = 3
kernel = 1 * RationalQuadratic(length_scale=1.0, length_scale_bounds=(1e-5, 1e2))
noise = False

agg=pd.DataFrame()

for year in years:
    results=cross_validate(predictors, folds, year, kernel, noise, trials)
    agg=pd.concat([agg, results])
    print(agg)
# -

print(agg.mean(axis=0))

# ## Matern Kernel (2 predictors)

from sklearn.gaussian_process.kernels import Matern

# +
predictors = 2
kernel = 1 * Matern(length_scale=[1.0, 1.0], nu=1.5, length_scale_bounds=(1e-5, 1e5))
noise = True

agg=pd.DataFrame()

for year in years:
    results=cross_validate(predictors, folds, year, kernel, noise, trials)
    agg=pd.concat([agg, results])
    print(agg)
# -

print(agg.mean(axis=0))

# ## Matern Kernel (3 predictors)

# +
predictors = 3
kernel = 1 * Matern(length_scale=[1e-1, 1e-1, 1e-2], nu=1.5, length_scale_bounds=(1e-5, 1e5))
noise = True

agg=pd.DataFrame()

for year in years:
    results=cross_validate(predictors, folds, year, kernel, noise, trials)
    agg=pd.concat([agg, results])
    print(agg)
# -

print(agg.mean(axis=0))


