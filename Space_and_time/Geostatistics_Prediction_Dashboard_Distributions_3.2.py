#Standard imports
import pandas as pd
import numpy as np
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import NearestNDInterpolator
import matplotlib as mpl
import os
import matplotlib.ticker as mtick
import requests

#gdal imports
from osgeo import gdal, gdalconst

# +
# Display params

from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

pd.options.display.max_rows=150
pd.options.display.max_columns=150
# -

# __Readme__

# This notebook implements v2.1 of the Gaussian Process Regression to predict Long Island Sound Coastal Temperature. It does not rely on a training and test set in order to evaluate accuracy, but instead builds a model for each year and evaluates it on a grid of test points in order to create a heatmap.

# __v_2.1.2__

# In this version:
#
# 1. Window of interest adjusted to include New York Embayments as per Cayla's suggestion
# 2. Output is a geotiff for making a composite
# 3. Leverage masked input to designate embayments, and use a default of embay_dist=0 for the rest of the sound
# 4. Implement v2.1.2 of the Model (heatmaps made after this change are listed as "_v3"
# 5. Added in an option for fixed kernel implementation that averages the hyperparamaters of the desired modes from the 2019 and 2020 models
# 6. Changed "test_mean" to "mean" in model_output (HUGE bug)

# __v_3__

# 1. "manipulating Coastal Distance" now produces optional embayment only map (stored in path6)
# 2. Removed fixed kernel implementation
# 3. Removed test section
# 4. Rational Quadratic Kernel Used Instead of RBF

# __v_3.2__
#
# 1. Expanded Data to use all of June 1st-Oct 1st as opposed to just two months during the summer
# 2. Moved Train_Model to another notebook so that fixed kernels could be used for this one

#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

# +
#Global Params:

#Defining Eastern Sound Window (useful for visualizing cropped distance map)
lon_min=-72.59
lon_max=-71.81
lat_min=40.97
lat_max=41.54

#Whether or note to restrict to the Eastern Sound window
es=True

#Tags of Organizations for Continous Monitoring
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

#Years
years=[2019, 2020, 2021]

#Variables

station_var=["Station ID"]
#(order matters here because of how get_heatmap_inputs works)
indep_var=["Longitude", "Latitude", "embay_dist", "Day"]
dep_var= ["Temperature (C)"]
# -

#Working dir
if(os.path.basename(os.getcwd())[0:18]!="Data Visualization"):
    os.chdir("..")
assert(os.path.basename(os.getcwd())[0:18]=="Data Visualization")

# +
#Loading paths config.yml
import yaml

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# -

# All TIFFs loaded below are cropped to eastern sound, this cropping is done in "Cropping to Eastern Sound" script/nb

# +
import os
#Paths for Inputs
paths={}

# Coastal features

#GeoTiff for part of LIS not in a vaudrey embayment that was measured (open sound)
paths[1] = config["Prediction_Dashboard_path1"]

#Geotiff of embay distance clipped to be outside state lines (to remove land)
paths[2] = config["Prediction_Dashboard_path2"]

#Embayment distance clipped to be only in a vaudrey embayment that had measurements and an embay_dist value
paths[3] = config["Prediction_Dashboard_path3"]

#Final embayment distance (output of first paragraph of notebook from v 2.1.2)
paths[4]  = config["Prediction_Dashboard_path4"]

#Final embayment distance (output of first paragraph of notebook) BUT within measured embays
#This is the preferred dataset for dashboard 3 but doesn't open w/ gdal properly
paths[6] = config["Prediction_Dashboard_path6"]

#Like paths[6] but within entire sound
paths[8] = config["Prediction_Dashboard_path8"]

#Ouput Path for tiffs
paths[7] = config["Prediction_Dashboard_output"]

#CSV

#Training Data
paths[5] =  "Data/Space_and_time_agg/agg_daily_morning_coastal_features_6_21_2023.csv"

#Ground Truthing Data
paths[9] = config["Prediction_Dashboard_path9"]

for path in paths.values():
    assert os.path.exists(path), path
# -

# # Reading in Data and Building in Model

# ## Reading csv

# +
#This is the source of the model's predictions
df=pd.read_csv(paths[5], index_col=0)

#Checking to make sure days have right distribution
df["Day"].describe()
# -

#Checking Dominion Stations to Make Sure they are in Vaudrey Embayment (C and NB)
df.loc[df["Organization"]=="Dominion"]

#Dropping nas
print(len(df))
df.dropna(subset=["Station ID", "Longitude", "Latitude", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

# #Optional Restriction of TRAINING and later testing params to Eastern Sound
if es==True:
    whole_sound=df.copy()
    df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()
    print(len(whole_sound))
    print(len(df))

# # Data summaries

# +
#Summary of df by cont/non-cont.
working=df.drop_duplicates(["Station ID", "Year"]).copy()
working=working.loc[working["Year"].isin(range(2019, 2022))]

#Grouping by data type
working["Category"]=""
working.loc[working["Organization"].isin(cont_orgs), "Category"]="Continuous"
working.loc[~working["Organization"].isin(cont_orgs), "Category"]="Discrete"

#Fixing Names
working = working.groupby(["Year", "Category"])["Station ID"].count()
working=pd.DataFrame(working)

working.rename(columns={"Station ID":"Number of Stations"}, inplace=True)
working=working.unstack(level=1)

working.to_csv("Figures_for_Paper/tab1.csv")
working
# -

#Non-continuous data frequencies
working=df.loc[~df["Organization"].isin(cont_orgs)]
working=working.loc[working["Year"].isin(range(2019, 2022))]
working=pd.DataFrame(working.groupby(["Organization", "Station ID", "Year"])["Day"].count())
working.rename(columns={"Day":"Average Number Measurements/Station"}, inplace=True)
working.reset_index(inplace=True)
working=working.groupby(["Year", "Organization"]).mean()
working

df.loc[df["Station ID"].str[0:11]==("CFE-STS-NIR")]

df.groupby(["Year", "Organization"]).mean()

pd.unique(df["Organization"])

#Getting list of continuous organizations for alphas (move to YAML)
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

# # Making a version of Data w/ Linear Interpolation for Model Testing

#Locating outliers from below
df.loc[df["Station ID"].isin(["CFE-STS-CTR-02",
"CFE-STS-CTR-07", "URI-WW629"])]

days=range(152, 275)

# +
#Making dataframe with proper index to put interpolation into
df_int=df.copy()

by_station=df_int.drop_duplicates(["Station ID", "Year"])[["Station ID", "Year", "Latitude", "Longitude"]]
print(len(by_station))
ddf=pd.DataFrame(data={"Day":days})

#lengths should match up
cross=by_station.merge(ddf, how="cross")
print(len(cross)/len(days))

reindexed=cross.merge(df_int.drop(["Latitude", "Longitude"], axis=1), how="left", on=["Station ID", "Year", "Day"])
reindexed.loc[~reindexed["Temperature (C)"].isna()]

# +
#Interpolation

#sorting data by year
reindexed.sort_values(by=["Year", "Station ID", "Day"], inplace=True)

for year in years:
    yeardf=df_int.loc[df_int["Year"]==year]
    yearsta=pd.unique(yeardf["Station ID"])
    for sta in yearsta:
        working=reindexed.loc[(reindexed["Year"]==year) & (reindexed["Station ID"]==sta), "Temperature (C)"]
        working=working.interpolate(method="linear", limit_direction="both")
        reindexed.loc[(reindexed["Year"]==year) & (reindexed["Station ID"]==sta), "Temperature (C)"]=working
        
reindexed.dropna(subset=["Temperature (C)"], axis=0, inplace=True)
df_int=reindexed
df_int


# -

# # Building Model

#Building Model (adapted to dashboard from "Geostatistics_Workbook_version_2.1.2.ipynb")
def build_model(predictors, year, kernel, alpha):
    
    #Selecting data from proper year
    period=df.loc[df["Year"]==year, station_var+indep_var+dep_var]
    data=period.values

    #Partitioning X and y
    
    X_train, y_train = data[:, 1:predictors+1], data[:, -1]
    
    #Ensuring proper dtypes
    X_train=np.array(X_train, dtype=np.float64)
    y_train=np.array(y_train, dtype=np.float64)

    #Demeaning y
    y_mean=np.mean(y_train)
    y_train=y_train-y_mean

    #Normalizing predictors (X) for training sets
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    X_train=X_train - x_mean
    X_train=X_train / x_std
    
    #Constructing Process
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=alpha)


    #Training Process
    gaussian_process.fit(X_train, y_train)

    return(gaussian_process, y_mean, x_mean, x_std)

# # Parameters and Training Process

#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

#Day range - make sure this matches linear interpolation above
days=range(152, 275)

# +
#Params
predictors=4
alpha=.25
stride=20
lsb=(1e-5, 1e5)
kernel = 1 * RBF([1]*predictors, length_scale_bounds=lsb) + 1 * RationalQuadratic(length_scale_bounds=lsb)
years = [2019, 2020, 2021]
temp_limits = (19, 27)
day_limits = (0, 50)
#continuous monitoring "orgs"
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

#Days for showing model differentiation
display_days={"July": 196, "August": 227}

#Dicts for storage of models
kernels_liss = {2019: 1.06**2 * RBF(length_scale=[0.0334, 0.135, 2.29e+03, 0.185], length_scale_bounds="fixed") + 2.85**2 * RationalQuadratic(alpha=0.266, length_scale=1.85, alpha_bounds="fixed", length_scale_bounds="fixed"),
2020:0.939**2 * RBF(length_scale=[0.0683, 0.106, 7.87, 0.14], length_scale_bounds="fixed") + 5.53**2 * RationalQuadratic(alpha=0.126, length_scale=4.06, alpha_bounds="fixed", length_scale_bounds="fixed"),
2021:1.09**2 * RBF(length_scale=[0.213, 0.159, 4.78, 0.122], length_scale_bounds="fixed") + 1.51**2 * RationalQuadratic(alpha=0.412, length_scale=1.26, alpha_bounds="fixed", length_scale_bounds="fixed")
}
kernels_es = {2019: 0.645**2 * RBF(length_scale=[3.17, 3.04e+03, 6.03, 0.0479], length_scale_bounds="fixed") + 7.4**2 * RationalQuadratic(alpha=0.0217, alpha_bounds="fixed", length_scale=1.29, length_scale_bounds="fixed"),
2020: 1.3**2 * RBF(length_scale=[4.42, 1e+05, 22.9, 0.054], length_scale_bounds="fixed") + 3.45**2 * RationalQuadratic(alpha=0.0324, length_scale=0.451, alpha_bounds="fixed", length_scale_bounds="fixed"),
2021: 1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052], length_scale_bounds="fixed") + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538, alpha_bounds="fixed", length_scale_bounds="fixed")    
}

errors= {}
models = {}
hmaps= {}
ymeans = {}
xmeans = {}
xstds = {}
# -

#Training model with proper kernels
#Eastern Sound model Using length scales trained on overall LIS
if es:
    for year in years:
        #Getting models and ymeans outputted
        models[year], ymeans[year], xmeans[year], xstds[year] = build_model(predictors, year, kernels_es[year], alpha)
        #print(models[year].kernel_)


for year in years:
    print(models[year].kernel_)

# # Defining Functions to Use for CTDEEP idw

from Data import inverse_distance_weighter as idw


#Make sure that the inverse distance weighter is loaded, this now requires a year variable"Year" and a day variable "Day" in each row
def interpolate_means(daily_data):
    output=daily_data.copy(deep=True)
    output["coords"]=output.apply(lambda x: [x["Latitude"], x["Longitude"]], axis=1)
    output["interpolated"]=output.apply(idw.ct_deep_idw, axis=1, args=[1])
    
    #Unstacking is unneccesary because STS stations have diff coords each year
    #output=output.unstack(level=-1)
    
    #Unnecessary to convert because datatype of year should be numeric for input
    #output.columns=[int(year) for year in output.columns]
    
    return(output)


# # CTDEEP IDW Bias and RMSE

# ## Whole Sound July/August

idw_test=whole_sound.loc[whole_sound["Year"].isin(years)]

concat = interpolate_means(idw_test)

# +
#Limiting to days in July/August for comparability
rmse_range=range(182, 244)

#RMSE and bias
concat=concat.loc[concat["Day"].isin(rmse_range)].copy()
concat["Bias"]=concat["interpolated"]-concat["Temperature (C)"]
concat["Squared Error"]=np.square(concat["Bias"])
mean_error=concat.groupby("Year").mean()["Squared Error"]
mean_error=np.sqrt(mean_error)
mean_error=pd.DataFrame(mean_error)
mean_error.rename(columns={"Squared Error":"Root Mean Squared Error"}, inplace=True)
mean_error=pd.concat([mean_error, pd.DataFrame({"Root Mean Squared Error": mean_error.mean()})])
mean_error.rename(index={"Root Mean Squared Error": "Intertemporal Mean"}, inplace=True)
mean_error.index.name="Year"
mean_error.to_csv("Results/CT_DEEP_Interpolation/RMSE_LIS_JA.csv")
mean_error

# +
#Saving bias to csv
bias=concat.groupby(["Station ID", "Year"]).mean()
print(len(bias))
concat.to_csv("Results/CT_DEEP_Interpolation/Bias_LIS_JA.csv")
#print(bias.head())
#mean_error

bias.groupby(["Year"]).mean()
# -

# ## Whole Sound All Days

idw_test=whole_sound.loc[whole_sound["Year"].isin(years)]

concat = interpolate_means(idw_test)

# +
#Limiting to days in July/August for comparability
rmse_range=range(152, 274)

#RMSE and bias
concat=concat.loc[concat["Day"].isin(rmse_range)].copy()
concat["Bias"]=concat["interpolated"]-concat["Temperature (C)"]
concat["Squared Error"]=np.square(concat["Bias"])
mean_error=concat.groupby("Year").mean()["Squared Error"]
mean_error=np.sqrt(mean_error)
mean_error=pd.DataFrame(mean_error)
mean_error.rename(columns={"Squared Error":"Root Mean Squared Error"}, inplace=True)
mean_error=pd.concat([mean_error, pd.DataFrame({"Root Mean Squared Error": mean_error.mean()})])
mean_error.rename(index={"Root Mean Squared Error": "Intertemporal Mean"}, inplace=True)
mean_error.index.name="Year"
mean_error.to_csv("Results/CT_DEEP_Interpolation/RMSE_LIS_152_274.csv")
mean_error

# +
#Saving bias to csv
bias=concat.groupby(["Station ID", "Year"]).mean()
print(len(bias))
concat.to_csv("Results/CT_DEEP_Interpolation/Bias_LIS_152_274.csv")
print(bias.head())
#mean_error

bias.groupby(["Year"]).mean()
# -

# ## ES All Days

idw_test=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max) & (df["Year"].isin(years))]

concat = interpolate_means(idw_test)

# +
#Limiting to days in July/August for comparability
rmse_range=range(152, 274)

#RMSE and bias
concat=concat.loc[concat["Day"].isin(rmse_range)].copy()
concat["Bias"]=concat["interpolated"]-concat["Temperature (C)"]
concat["Squared Error"]=np.square(concat["Bias"])
mean_error=concat.groupby("Year").mean()["Squared Error"]
mean_error=np.sqrt(mean_error)
mean_error=pd.DataFrame(mean_error)
mean_error.rename(columns={"Squared Error":"Root Mean Squared Error"}, inplace=True)
mean_error=pd.concat([mean_error, pd.DataFrame({"Root Mean Squared Error": mean_error.mean()})])
mean_error.rename(index={"Root Mean Squared Error": "Intertemporal Mean"}, inplace=True)
mean_error.index.name="Year"
mean_error.to_csv("Results/CT_DEEP_Interpolation/RMSE_ES_152_274.csv")
mean_error

# +
#Saving bias to csv
bias=concat.groupby(["Station ID", "Year"]).mean()
print(len(bias))
concat.to_csv("Results/CT_DEEP_Interpolation/Bias_ES_152_274.csv")
print(bias.head())
#mean_error

bias.groupby(["Year"]).mean()
# -

# ## ES JA

idw_test=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max) & (df["Year"].isin(years))]

concat = interpolate_means(idw_test)

# +
#Limiting to days in July/August for comparability
rmse_range=range(182, 244)

#RMSE and bias
concat=concat.loc[concat["Day"].isin(rmse_range)].copy()
concat["Bias"]=concat["interpolated"]-concat["Temperature (C)"]
concat["Squared Error"]=np.square(concat["Bias"])
mean_error=concat.groupby("Year").mean()["Squared Error"]
mean_error=np.sqrt(mean_error)
mean_error=pd.DataFrame(mean_error)
mean_error.rename(columns={"Squared Error":"Root Mean Squared Error"}, inplace=True)
mean_error=pd.concat([mean_error, pd.DataFrame({"Root Mean Squared Error": mean_error.mean()})])
mean_error.rename(index={"Root Mean Squared Error": "Intertemporal Mean"}, inplace=True)
mean_error.index.name="Year"
mean_error.to_csv("Results/CT_DEEP_Interpolation/RMSE_ES_JA.csv")
mean_error
# -

#Saving bias to csv
pd.options.display.float_format = "{:.2f}".format
bias=concat.groupby(["Station ID", "Year"]).mean()
print(len(bias))
concat.to_csv("Results/CT_DEEP_Interpolation/Bias_ES_JA.csv")
#print(bias.head())
#mean_error
bias_results=bias[["Temperature (C)", "interpolated", "Bias"]].groupby(["Year"]).mean()
bias_results.rename(columns={"Temperature (C)": "Temp Mean Across Stations (Deg C)", "interpolated":"Interpolated Temp Mean Across Stations (Deg C)", "Bias":"Interpolation Bias"}, inplace=True)
bias_results

# ## By station

# +
#Generating CTDEEP interpolated time series for each station within each year

ct_deep_int={}

for year in years:
    #Re-retrieving data WITHIN AREA DEFINED ABOVE and creating reindexed dataframe with every day present to feed into model
    working = df.loc[(df["Year"]==year)].copy()
    
    #Optional restriction to see if the continuous stations have a distribution that aligns better (make sure it matches below)
    #working=working.loc[working["Organization"].isin(cont_orgs)]

    by_station=working.drop_duplicates(subset=["Station ID"]).copy()
    by_station.drop("Day", axis=1, inplace=True)
    stations=pd.DataFrame(working.drop_duplicates(subset=["Station ID"])["Station ID"])
    #print(stations)
    day_list=pd.DataFrame(days, columns=["Day"])
    #print(day_list)
    cross = stations.merge(day_list, how = "cross")
    #print(cross)
    cross = cross.merge(by_station, how="left", on="Station ID")
    #print(cross.head)
    
    #Adding year var for passing to idw function:    
    cross["Year"]=year
    ct_deep_int[year]=interpolate_means(cross)
# -

# # Plots of distributions

#Graphing param
plt.rcParams["figure.figsize"]=(30, 15)

# +
#Summer Averages histogram

fig, ax=plt.subplots(3, 3)
for i, year in enumerate(years):
    
#Plotting from underlying (linearly interpolated) data
    
    #Averaging year's data by station WITHIN AREA DEFINED ABOVE
    working = df_int.loc[(df_int["Year"]==year)].copy()
    
    #Optional restriction to see if the continuous stations have a distribution that aligns better (Make Sure it Matches Above)
    #working.loc[working["Organization"].isin(cont_orgs)]
    
    #This can be commented out to see histogram of all datapoints (Make Sure it Matches Above)
    working = working.groupby("Station ID").mean()

    #print(working.head())
    weights=np.ones_like(working["Temperature (C)"])/len(working)
    ax[0, i].hist(working["Temperature (C)"], weights=weights, edgecolor="black", color="tab:orange", bins=np.arange(temp_limits[0], temp_limits[1], .2))
    ax[0, i].set_xlim(temp_limits)
   #print(np.std(working["Temperature (C)"]))
    
    ax[0, i].set_title("Summer Average Temperature Distribution (Data), " + str(year), fontsize=18)
    ax[0, i].set_ylabel("Prob", fontsize=18)
    ax[0, i].set_xlabel("Temperature (C)", fontsize=18)
    ax[0, i].tick_params(labelsize=18)

#Plotting from model

    #Retrieving data WITHIN AREA DEFINED ABOVE and creating reindexed dataframe with every day present to feed into model
    working = df.loc[(df["Year"]==year)].copy()
    
    #Optional restriction to see if the continuous stations have a distribution that aligns better (make sure it matches below)
    #working=working.loc[working["Organization"].isin(cont_orgs)]

    by_station=working.drop_duplicates(subset=["Station ID"]).copy()
    by_station.drop("Day", axis=1, inplace=True)
    stations=pd.DataFrame(working.drop_duplicates(subset=["Station ID"])["Station ID"])
    #print(stations)
    day_list=pd.DataFrame(days, columns=["Day"])
    #print(day_list)
    cross = stations.merge(day_list, how = "cross")
    #print(cross)
    cross = cross.merge(by_station, how="left", on="Station ID")
    reindexed=cross[indep_var].copy()
    #print(reindexed.head)
    
    #Normalizing predictors
    X_pred = reindexed.values
    X_pred=X_pred - xmeans[year]
    X_pred=X_pred / xstds[year]
    
    #Using reindexed in prediction
    y_pred, MSE = models[year].predict(X_pred, return_std=True)
    y_pred+=ymeans[year]
    #print(ymeans[year])
    
    #Adding modeled data back into predictors
    reindexed["Temperature (C)"]=y_pred
    #print(reindexed)
    
    #Adding back in station ID
    reindexed["Station ID"]=cross["Station ID"]
    #print(reindexed)
    
    #This can be commented out to see histogram of all datapoints (Make Sure it Matches Below)
    reindexed=reindexed.groupby("Station ID").mean()
    #print(reindexed)
    
    weights=np.ones_like(reindexed["Temperature (C)"])/len(reindexed)
    ax[1, i].hist(reindexed["Temperature (C)"], weights=weights, edgecolor="black", color="tab:blue", bins=np.arange(temp_limits[0], temp_limits[1], .2))
    ax[1, i].set_xlim(temp_limits)
    ax[1, i].set_title("Summer Average Temperature Distribution (Approach II), " + str(year), fontsize=18)


    ax[1, i].set_ylabel("Prob", fontsize=18)
    ax[1, i].set_xlabel("Temperature (C)", fontsize=18)
    ax[1, i].tick_params(labelsize=18)

    #print(np.std(reindexed["Temperature (C)"]))

#Plotting CTDEEP interpolated

    working=ct_deep_int[year]
    working =working.groupby("Station ID").mean()
    
    weights=np.ones_like(working["interpolated"])/len(reindexed)
    ax[2, i].hist(working["interpolated"], weights=weights, edgecolor="black", color="tab:blue", bins=np.arange(temp_limits[0], temp_limits[1], .2))
    ax[2, i].set_xlim(temp_limits)
    
    ax[2, i].set_title("Summer Average Temperature Distribution (Approach I), " + str(year), fontsize=18)
    ax[2, i].set_ylabel("Prob", fontsize=18)
    ax[2, i].set_xlabel("Temperature (C)", fontsize=18)
    ax[2, i].tick_params(labelsize=18)    

fig.tight_layout()

if es:
    plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES_Summer_Average_Distributions.png")
    plt.savefig("Figures_for_paper/fig9.png")
    
plt.show()
# -

plt.rcParams["figure.figsize"]=(45, 20)

thresholds = [23, 25]

# +
#Days over Chosen Threshold(s) box and whisker

#csv for all data
all_data=dict(zip(years, [pd.DataFrame() for i in range(len(years))]))

#Plotting from Model
hfig, hax =plt.subplots(len(thresholds), len(years), squeeze=True)

for j, year in enumerate(years):
    for i, thresh in enumerate(thresholds):

#Plotting from CTDEEP data

        ct_deep_thresh=ct_deep_int[year].copy()
        ct_deep_thresh["Threshold"] = ct_deep_thresh["interpolated"]>thresh
        ct_deep_thresh["Threshold"]=ct_deep_thresh["Threshold"].astype(int)
        ct_deep_thresh = ct_deep_thresh.groupby("Station ID")["Threshold"].sum()*100/ct_deep_thresh.groupby("Station ID")["Threshold"].count()
        
    #Plotting from Model
    
        #Retrieving data WITHIN AREA DEFINED ABOVE and creating reindexed dataframe with every day present to feed into model
        working = df.loc[(df["Year"]==year)].copy()

        #Optional restriction to see if the continuous stations have a distribution that aligns better (make sure it matches below)
        #working=working.loc[working["Organization"].isin(cont_orgs)]

        by_station=working.drop_duplicates(subset=["Station ID"]).copy()
        by_station.drop("Day", axis=1, inplace=True)
        stations=pd.DataFrame(working.drop_duplicates(subset=["Station ID"])["Station ID"])
        #print(stations)
        day_list=pd.DataFrame(days, columns=["Day"])
        #print(day_list)
        cross = stations.merge(day_list, how = "cross")
        #print(cross)
        cross = cross.merge(by_station, how="left", on="Station ID")
        reindexed=cross[indep_var].copy()
        #print(reindexed.head)

        #Normalizing predictors
        X_pred = reindexed.values
        X_pred=X_pred - xmeans[year]
        X_pred=X_pred / xstds[year]

        #Using reindexed in prediction
        y_pred, MSE = models[year].predict(X_pred, return_std=True)
        y_pred+=ymeans[year]
        #print(ymeans[year])

        #Adding modeled data back into predictors
        reindexed["Temperature (C)"]=y_pred
        #print(reindexed)

        #Adding back in station ID
        reindexed["Station ID"]=cross["Station ID"]
        
        #Getting pct of days over threshold
        reindexed["Threshold"]= reindexed["Temperature (C)"]>thresh
        reindexed["Threshold"]=reindexed["Threshold"].astype(int)
        reindexed = reindexed.groupby("Station ID")["Threshold"].sum()*100/reindexed.groupby("Station ID")["Threshold"].count()

#Plotting from Underlying Lin. Interpolated Data

        #Getting year's data by station in range defined earlier in notebook
        working = df_int.loc[(df_int["Year"]==year)].copy()
        
        #Optional restriction to see if the continuous stations have a distribution that aligns better (make sure it matches below)
        #working=working.loc[working["Organization"].isin(cont_orgs)]
        
        working["Threshold"]= working["Temperature (C)"]>thresh
        working["Threshold"]=working["Threshold"].astype(int)
        working = working.groupby("Station ID")["Threshold"].sum()*100/working.groupby("Station ID")["Threshold"].count()
        
        #Printing outliers
        #print(working.loc[working>70])
        
        hax[i, j].boxplot([ct_deep_thresh, reindexed, working], vert=0)
        hax[i, j].set_yticklabels([ "Approach I", "Approach II", "Lin. Interp. Data"], fontsize=22)
        hax[i, j].set_xlabel("% of Days", fontsize=22)
        hax[i, j].set_title(str(thresh) + " Deg (C) Thresh, " + str(year), fontsize=28)
        hax[i, j].set_xlim(-5, 105)
        hax[i, j].xaxis.set_major_formatter(mtick.PercentFormatter())
        hax[i, j].tick_params(axis="both", which="major", labelsize=22)
        plt.tight_layout(pad=10)
        
#Exporting csv of Results for analysis:
        all_data[year][str(thresh) + " (C)"]=working
        all_data[year]["Year"]=year
        
agg_data=pd.concat(all_data)

if es:
    plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES_Pct_Days_over_Thresholds_BandW.png")
    plt.savefig("Figures_for_paper/fig10.png")
plt.show()

# -

# # Sample Time Series

# ## Discrete Stations

#Graphing params
plt.rcParams["figure.figsize"]=(30, 20)

#Param
samples= 3

# +
#Script for generating samples (within each year because coordinates change)

#Restricting to non-continuous data to see how GP fills in data
working=df.loc[~df["Organization"].isin(cont_orgs)].copy()

#Getting stations
by_station=working.drop_duplicates(subset=["Station ID"]).copy()

#Randomly sampling number of samples defined above
stations=by_station.sample(samples, axis=0)
stations=list(stations["Station ID"])
#print(by_station)
agg=pd.DataFrame()

for year in years:
    by_station=df.loc[(df["Station ID"].isin(stations)) & (df["Year"]==year)].drop_duplicates(subset=["Station ID"])
    by_station.drop("Day", axis=1, inplace=True)
    #Adding each day in range for every station samples
    day_list=pd.DataFrame(days, columns=["Day"])
    cross = by_station.merge(day_list, how = "cross")
    agg=pd.concat([agg, cross])

# +
#Plots

fig, ax=plt.subplots(samples, len(years))

for j, year in enumerate(years):
    
#Generating GP time series for randomly selected stations

    #Merging other variables with station and days
    year_data=agg.loc[agg["Year"]==year].copy()
    reindexed=year_data[indep_var].copy()
    
    #Normalizing predictors
    X_pred = reindexed.values
    X_pred=X_pred - xmeans[year]
    X_pred=X_pred / xstds[year]
    #print(X_pred)
    
    #Using reindexed in prediction for given year's model
    y_pred, MSE = models[year].predict(X_pred, return_std=True)
    y_pred+=ymeans[year]
    #print(ymeans[year])
    
    #Adding modeled data back into predictors
    reindexed["Temperature (C)"]=y_pred
    reindexed["Error"]=MSE
    #print(reindexed)
    
    #Adding back in station ID
    reindexed["Station ID"]=year_data["Station ID"]
    #print(reindexed)
    
#Plotting GP time series within each year
    for i, sta in enumerate(pd.unique(reindexed["Station ID"])):
        working=reindexed.loc[reindexed["Station ID"]==sta]
        ax[i, j].plot(working["Day"], working["Temperature (C)"])
        ax[i, j].fill_between(working["Day"], 
                              working["Temperature (C)"] - 1.96 * working["Error"],
                              working["Temperature (C)"] + 1.96 * working["Error"],
                              alpha=.25,
                              label=r" GP 95% confidence interval")
        
#Plotting ctdeep time series for station
    for i, sta in enumerate(pd.unique(reindexed["Station ID"])):
        working = ct_deep_int[year]
        working=working.loc[working["Station ID"]==sta]
        ax[i, j].plot(working["Day"], working["interpolated"])

#Plotting original data with time series for comparison
        data = df.loc[(df["Year"]==year) & (df["Station ID"]==sta)].copy()
        ax[i, j].scatter(data["Day"], data["Temperature (C)"], s=150, color="tab:purple")
        
#Formatting
        ax[i, j].set_title(sta + " " + str(year), fontsize=24)
        ax[i, j].set_ylim([15, 30])
        ax[i, j].set_xlim([min(days), max(days)])
        ax[i, j].tick_params(axis="x", labelsize=22)
        ax[i, j].tick_params(axis="y", labelsize=22)
        fig.legend(["Gaussian Process", "95% Confidence Interval", "Inverse Distance Weighting", "Discrete Datapoints"],  prop={'size': 25}, bbox_to_anchor=[1.15, 1.05])
        
#fig.suptitle("Fig 6a: Synthesizing Continuous Time Series from Discrete Datapoints (Eastern Sound Hyperparameters)", fontsize=32)
fig.supxlabel("Day of the Year", fontsize=24)
fig.supylabel("Temperature (C)", fontsize=24)
plt.tight_layout(pad= 5)

plt.savefig("Figures_for_paper/fig6.png", bbox_inches='tight')
plt.show()
# -
for year in years:
    params=models[year].get_params()


# ## Niantic

#Graphing params
plt.rcParams["figure.figsize"]=(30, 20)

#Identifying control station in STS Continuous data
controls = pd.unique(df.loc[(df["Organization"]=="STS_Tier_II") & (df["Station ID"].str[:3]=="NIR"), "Station ID"])
assert len(controls)==2
controls

# +
#Niantic Specific Samples (location of highest interest)

gt_for_pred={}

#Making a prediction dataframe for each year in case location changes
for year in years:
    #Restricting to non-continuous data to see how GP fills in data
    working=pd.read_csv("Data/Dominion Energy/Millstone_Shoot_Counts_coastal_features.csv", index_col=0)

    #Creating year variable
    working["Date"]=pd.to_datetime(working["Date"])
    working["Year"]=working["Date"].dt.year

    #Limiting to years of interest
    working = working.loc[working["Year"]==year]
    print(pd.unique(working["Year"]))

    #Getting stations
    by_station=working.drop_duplicates(subset=["Station ID"]).copy()

    #Adding each day in range for every station samples
    day_list=pd.DataFrame(days, columns=["Day"])
    gt_for_pred[year] = by_station.merge(day_list, how = "cross")
    #print(gt_for_pred[year])
    
# -

#keys for figures made from niantic-specific samples
keys=dict(zip(range(len(by_station)), "abc"))

# +
#Plots
years=[2021]
k=len(pd.unique(by_station["Station ID"]))
fig, ax=plt.subplots(k)

for i, sta in enumerate(pd.unique(by_station["Station ID"])):
    #Each ground truthed station gets its own plot

    for j, year in enumerate(years):
    
#Generating GP time series for each ground truthing station in Millstone Environmental Lab dataset

        #Merging other variables with station and days
        station_data=gt_for_pred[year].loc[gt_for_pred[year]["Station ID"]==sta]
        reindexed=station_data[indep_var].copy()
        #print(reindexed)
        
        #Normalizing predictors
        X_pred = reindexed.values
        X_pred=X_pred - xmeans[year]
        X_pred=X_pred / xstds[year]
        #print(X_pred)

        #Using reindexed in prediction for given year's model
        y_pred, MSE = models[year].predict(X_pred, return_std=True)
        y_pred+=ymeans[year]
        #print(ymeans[year])

        #Adding modeled data back into predictors
        reindexed["Temperature (C)"]=y_pred
        #print(reindexed)

        #Adding back in station ID
        reindexed["Station ID"]=sta
        #print(reindexed)

#Plotting time series within each year
        ax[i].plot(reindexed["Day"], 
                   reindexed["Temperature (C)"], label=sta + " Gaussian Process")
    
#Plotting 95% confidence interval
        ax[i].fill_between(reindexed["Day"], 
                           reindexed["Temperature (C)"]-1.96*MSE, 
                           reindexed["Temperature (C)"]+1.96*MSE, 
                           alpha=.25, 
                           label=r" GP 95% confidence interval")


        ax[i].set_title("Station " + sta + ", " + str(year) + " w/ Controls", fontsize=32)
        
#Average SE of 95% confidence interval
        print(MSE.mean())

        for k, control in enumerate(controls):
#Plotting niantic_controls for comparison
            data = df.loc[(df["Year"]==year) & (df["Station ID"]==control)].copy()
            ax[i].plot(data["Day"], data["Temperature (C)"], label=control)
    
#Formatting all axes
        for axis in fig.get_axes():
            axis.set_ylim([15, 27])
            axis.set_xlim([min(days), max(days)])
            axis.tick_params(axis="x", labelsize=32)
            axis.tick_params(axis="y", labelsize=32)
    
#fig.suptitle("Fig 12: Synthesizing Continuous Time Series for Millstone Eelgrass Stations (2021)", fontsize=32)
fig.supxlabel("Day of the Year", fontsize=32)
fig.supylabel("Temperature (C)", fontsize=32)
plt.tight_layout(pad= 5)
handles, labels=ax[i].get_legend_handles_labels()
fig.legend(handles, labels, prop={'size': 32}, bbox_to_anchor=[1.25, 1.05])
plt.savefig("Figures_for_paper/fig12" + keys[i] + ".png")
plt.show()
# -

# # Ground Truth at JC, WP, NR


# +
#Dataframe for temp prediction (Code from Above)

gt_for_pred={}

#Making a prediction dataframe for each year in case location changes
for year in years:
    #Restricting to non-continuous data to see how GP fills in data
    working=pd.read_csv("Data/Dominion Energy/Millstone_Shoot_Counts_coastal_features.csv", index_col=0)

    #Creating year variable
    working["Date"]=pd.to_datetime(working["Date"])
    working["Year"]=working["Date"].dt.year

    #Limiting to years of interest
    working = working.loc[working["Year"]==year]
    print(pd.unique(working["Year"]))

    #Getting stations
    by_station=working.drop_duplicates(subset=["Station ID"])[["Station ID", "Year", "Latitude",  "Longitude", "embay_dist"]].copy()

    #Adding each day in range for every station samples
    day_list=pd.DataFrame(days, columns=["Day"])
    gt_for_pred[year] = by_station.merge(day_list, how = "cross")
    print(gt_for_pred[year])
    
# -

# ## Shoot Count (Trailing Mean)

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

# +
#params
gt_path="Data/Dominion Energy/Millstone_Shoot_Counts_coastal_features.csv"
gt_var="ShootCount"

#Days for trailing mean
periods=[90]

# +
gt_data=pd.read_csv(gt_path, index_col=0)
gt_data["Date"]=pd.to_datetime(gt_data["Date"])
gt_data["Day"]=gt_data["Date"].dt.day_of_year
gt_data["Year"]=gt_data["Date"].dt.year
gt_data["Month"]=gt_data["Date"].dt.month
gt_data=gt_data.loc[gt_data["Year"].isin(years)]
gt_data.rename(columns={"Number of Vegetative Shoots":"ShootCount"}, inplace=True)

gt_data
# -

#Demeaning at each sampling time 
grouped=gt_data.groupby(["Year", "Month"]).mean()["ShootCount"].reset_index()
grouped.rename(columns={"ShootCount": "Cross_Sectional_Mean"}, inplace=True)
gt_data=gt_data.merge(grouped, how="left", on=["Year", "Month"])
gt_data["shootcount_demeaned"]=gt_data["ShootCount"]-gt_data["Cross_Sectional_Mean"]
gt_data

#Range of ground truthing data for setting "days" param above
gt_data["Day"].describe()

# +
#Modelling trailing mean temperatures at ground truthing points
agg=pd.DataFrame()
for year in years:

## Getting temperature from GP Model
    working=gt_for_pred[year]
    print(pd.unique(working["Year"]))
    
    #Normalizing predictors
    X_pred = working[indep_var].values
    X_pred=X_pred - xmeans[year]
    X_pred=X_pred / xstds[year]

    #Using reindexed in prediction
    y_pred, MSE = models[year].predict(X_pred, return_std=True)
    y_pred+=ymeans[year]
    #print(ymeans[year])

    #Adding modeled data back into predictors
    working["gp_temp"]=y_pred
    working=interpolate_means(working)
    working.rename(columns={"interpolated":"idw_temp"}, inplace=True)
    
    #Readding station var
    working["Station ID"]=gt_for_pred[year]["Station ID"]
    
    for n in periods:
        working["gp_temp_" + str(n)]=working["gp_temp"].rolling(n, min_periods=1).mean()
        working["idw_temp_" + str(n)]=working["idw_temp"].rolling(n, min_periods=1).mean()

    agg=pd.concat([agg, working])
    
    
##Getting temperature from CTDEEP interpolation
agg.loc[(223<=agg["Day"]) & (agg["Day"]<=225)].head()
# +
#Getting ground truthing for test years and aggregating for shoot count
working=gt_data.loc[gt_data["Year"].isin(years)].copy()
errors=working.groupby(["Station ID", "Year", "Day"]).std().reset_index()
working=working.groupby(["Station ID", "Year", "Day"]).mean().reset_index()

#Merging agg data with gt data
comp=working.merge(agg, how="left", on=["Station ID", "Year", "Day"])
#print(comp)

#Making dataframe for all param results
agg_results=pd.DataFrame()

#Making dataframe for r2 results
r2=pd.DataFrame(index=["R-Squared Value"])

# +
#Plots

#Plotting data and regression line (gp)
fig, ax = plt.subplots()

for i, n in enumerate(periods):
    
    for sta in pd.unique(gt_data["Station ID"]):
        working=comp.loc[comp["Station ID"]==sta]

        bars=errors.loc[errors["Station ID"]==sta, gt_var]
        ax.errorbar(working["gp_temp_" + str(n)], working[gt_var], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)
    
        ax.scatter(working["gp_temp_" + str(n)], working[gt_var], s=100)

        #ax.set_title("Gaussian Process " + str(n) + " Day Trailing Avg Temp. and Ee", fontsize=24)
        ax.set_ylabel("Demeaned Shootcount", fontsize=24)
        ax.set_xlabel("Gaussian Process 90 Day Trailing Temp Avg", fontsize=24)
        ax.tick_params(axis="both", labelsize=18)

    #Reg
    Y = comp[gt_var]
    X = comp[["gp_temp_" + str(n)]]
    X=sm.add_constant(X)
    model=sm.OLS(Y, X)
    results =model.fit()
    p=results.params
    
    #Plotting reg line
    ax.plot(comp["gp_temp_" + str(n)], p.const + p["gp_temp_" + str(n)] * comp["gp_temp_" + str(n)], color="black")
    fig.legend(pd.unique(gt_data["Station ID"]), prop={'size': 25})
    plt.tight_layout(pad=10)

# #Plotting data and regression line (idw)

# for i, n in enumerate(periods):
    
#     for sta in pd.unique(gt_data["Station ID"]):
#         working=comp.loc[comp["Station ID"]==sta]

#         bars=errors.loc[errors["Station ID"]==sta, gt_var]
#         ax[1].errorbar(working["idw_temp_" + str(n)], working[gt_var], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

#         ax[1].scatter(working["idw_temp_" + str(n)], working[gt_var], s=100)

#         ax[1].set_title("CTDEEP IDW " + str(n) + " Day Trailing Avg", fontsize=24)
#         ax[1].set_ylabel("Demeaned Shootcount", fontsize=24)
#         ax[1].set_xlabel(str(n) + " Day Trailing Temp Avg", fontsize=24)
#         ax[1].tick_params(axis="both", labelsize=18)
#     #Reg
#     Y = comp[gt_var]
#     X = comp[["idw_temp_" + str(n)]]
#     X=sm.add_constant(X)
#     model=sm.OLS(Y, X)
#     results =model.fit()
#     p=results.params
    
#     #Plotting reg line
#     ax[1].plot(comp["idw_temp_" + str(n)], p.const + p["idw_temp_" + str(n)] * comp["idw_temp_" + str(n)], color="black")
#     fig.legend(pd.unique(gt_data["Station ID"]), prop={'size': 25})
#     plt.tight_layout(pad=10)

plt.savefig("Figures_for_paper/fig11.png")


# +
#QQ plots
working=comp.loc[comp["Station ID"]=="NR"]

#GP
for i, n in enumerate(periods):
    Y = comp[gt_var]
    X = comp[["gp_temp_" + str(n)]]
    X=sm.add_constant(X)
    model=sm.OLS(Y, X)
    results =model.fit()
    resid=results.resid
    
    qq_plot = sm.qqplot(resid, stats.t, fit=True, line="45")
    plt.show()

#IDW
for i, n in enumerate(periods):
    Y = comp[gt_var]
    X = comp[["idw_temp_" + str(n)]]
    X=sm.add_constant(X)
    model=sm.OLS(Y, X)
    results =model.fit()
    resid=results.resid
    
    qq_plot = sm.qqplot(resid, stats.t, fit=True, line="45")
    plt.show()


# +
#Regression w/ gp temp (Approach II)
variables=["r2", ""]

for i, n in enumerate((periods)):

    reg_name="gp_temp_" + str(n)
    comp["Temp_Rolling_Average"]=comp[reg_name]
    temp_var=reg_name
    control_var=[]
    data=comp

    #Covariance Matrix
    print("Endogenous var correlation:")
    print(np.corrcoef([comp["Temp_Rolling_Average"]] + [comp[var] for var in control_var]))

    #Reg
    Y = comp[gt_var]
    X = comp[["Temp_Rolling_Average"]+control_var]
    X=sm.add_constant(X)
    model=sm.OLS(Y, X)
    results =model.fit()
    p=results.params
    
    #Printing and storing param results for concatentation
    print(reg_name + " Results:")
    print(results.summary())
    
    tabs = results.summary().tables
    res_df = pd.read_html(tabs[1].as_html(), header=0, index_col=0)[0]
    res_df.index=["Constant", "GP Trailing Avg Temperature"]
    
    #Adding to r-squared results
    r2[reg_name]=[results.rsquared]
res_df

# +
#Reformatting results
pd.options.display.float_format = '{:,.3f}'.format

from pandas import IndexSlice as idx
formatted = agg_results.unstack(level=[0, 1]).stack(level=0)
formatted.rename(columns= {col: "GP " + col[-2:] + " Day Trailing Avg" for col in formatted.columns if col[:2]=="gp"}, inplace=True)
formatted.rename(columns= {col: "CTDEEP IDW " + col[-2:] + " Day Trailing Avg" for col in formatted.columns if col[:3]=="idw"}, inplace=True)
formatted.rename(index={"Day":"Day of Year", "Temp_Rolling_Average":"Temperature Variable", "const":"Constant"}, inplace=True)
formatted=formatted.loc[idx[:, ["coef", "t", "P>|t|"]], :]
formatted.sort_index(level=0, inplace=True)

#Adding r2 values
r2.columns=formatted.columns
r2.index=pd.MultiIndex.from_arrays([["R Squared"], [""]])
formatted=pd.concat([formatted, r2])
formatted
# -

r2

#Reformatting
reformatted = formatted.copy()
reformatted.columns=pd.MultiIndex.from_lists([["Approach I: "]])

# ## Repro Counts Trailing Mean

# +
#params
gt_path="Data/Dominion Energy/Millstone_Repro_Counts.csv"
gt_var="ReproCount"

#Days for trailing mean
periods=[30, 60, 90]

#Setting summer as time period for ground truthing of summer mean and days over thresh
gt_days=range(162, 269)

# +
gt_data=pd.read_csv(gt_path)

gt_data.replace({"Jordan Cove": "JC", "Niantic River": "NR", "White Point":"WP"}, inplace=True)
gt_data.rename(columns={"Station": "Station ID"}, inplace=True)

gt_data.head()
# -



# +
#Predicting Temp for all days in gt_for_pred
agg=pd.DataFrame()
for year in years:

## Getting temperature from GP Model
    working=gt_for_pred[year]
    print(pd.unique(working["Year"]))
    
    #Normalizing predictors
    X_pred = working[indep_var].values
    X_pred=X_pred - xmeans[year]
    X_pred=X_pred / xstds[year]

    #Using reindexed in prediction
    y_pred, MSE = models[year].predict(X_pred, return_std=True)
    y_pred+=ymeans[year]
    #print(ymeans[year])

    #Adding modeled data back into predictors
    working["Temperature (C)"]=y_pred
    agg=pd.concat([agg, working])
    
##Getting temperature from CTDEEP interpolation
agg =interpolate_means(agg)
agg.loc[(223<=agg["Day"]) & (agg["Day"]<=225)].head()
# +
#Getting years ground truthing for defined variable above

fig, ax = plt.subplots(2, 2)

##SUMMER MEAN

working=gt_data.loc[gt_data["Year"].isin(years)].copy()

#Aggregating interpolated ground truth data to get summer average and merging it with gt data
temp=agg.groupby(["Station ID", "Year"]).mean().reset_index()
comp=working.merge(temp, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).mean().reset_index()
errors=working.merge(temp, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).std().reset_index()

print("Gaussian Process Summer Avg Regression Results:")
#Regression Line (gp)
Y = comp[gt_var]
X = comp["Temperature (C)"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#Plotting and regression (gp)
for sta in pd.unique(gt_data["Station ID"]):
    working=comp.loc[comp["Station ID"]==sta]
    
    bars=errors.loc[errors["Station ID"]==sta, gt_var]
    ax[0, 0].errorbar(working["Temperature (C)"], working[gt_var], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    ax[0, 0].scatter(working["Temperature (C)"], working[gt_var], s=100)
    
    ax[0, 0].set_title(gt_var +  " vs. Gaussian Process-Predicted Temperature (Summer Avg)", fontsize=18)
    ax[0, 0].set_ylabel(gt_var, fontsize=18)
    ax[0, 0].set_xlabel("Temperature (C)", fontsize=18)

#Plotting gp reg line
ax[0, 0].plot(comp["Temperature (C)"], p.const + p["Temperature (C)"] * comp["Temperature (C)"], color="black")
fig.legend(pd.unique(gt_data["Station ID"]), prop={'size': 25}, bbox_to_anchor=[1.1, 1.05])

print("IDW Summer Avg Regression Results:")
#Regression Line (idw)
Y = comp[gt_var]
X = comp["interpolated"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_idw =model.fit()
print(results_idw.summary())
p=results_idw.params

#plotting (idw)
for sta in pd.unique(gt_data["Station ID"]):
    working=comp.loc[comp["Station ID"]==sta]
    
    bars=errors.loc[errors["Station ID"]==sta, gt_var]
    ax[0, 1].errorbar(working["interpolated"], working[gt_var], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    ax[0, 1].scatter(working["interpolated"], working[gt_var], s=100)

    ax[0, 1].set_title(gt_var + " vs. CTDEEP IDW-Predicted Temperature (Summer Avg)", fontsize=18)
    ax[0, 1].set_ylabel(gt_var, fontsize=18)
    ax[0, 1].set_xlabel("Temperature (C)", fontsize=18)

#Plotting idw reg line
ax[0, 1].plot(comp["interpolated"], p.const + p["interpolated"] * comp["interpolated"], color="black")

## DAYS OVER THRESHOLD
comp=pd.DataFrame()

#Getting days over threshold using the same "agg" from summer means calculation above
working=agg.copy()

#Temperature threshold for gaussian process
working["thresh_gp"]= working["Temperature (C)"]>thresh
working["thresh_gp"]=working["thresh_gp"].astype(int)

#Temperature threshold for CTDEEP idw data
working["thresh_idw"]= working["interpolated"]>thresh
working["thresh_idw"]=working["thresh_idw"].astype(int)

to_merge = pd.DataFrame(working.groupby(["Station ID", "Year"])[["thresh_gp", "thresh_idw"]].sum())
to_merge.reset_index(inplace=True)
#print(to_merge)

working=gt_data.loc[gt_data["Year"].isin(years)].copy()

#Re-merging gt from above with days over threshold
comp=working.merge(to_merge, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).mean().reset_index()
errors = working.merge(to_merge, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).std().reset_index()

print(errors.tail())
print("comp")
print(comp.tail())

print("IDW Days Over Threshold Results:")
#Regression Line (gp)
Y = comp[gt_var]
X = comp["thresh_gp"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#Plotting gp threshold estimates by year to visualize
#Plotting and regression (gp)
for sta in pd.unique(gt_data["Station ID"]):
    #data
    working=comp.loc[comp["Station ID"]==sta]

    #erors
    bars=errors.loc[errors["Station ID"]==sta, gt_var]
    ax[1, 0].errorbar(working["thresh_gp"], working[gt_var], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    #plot data
    ax[1, 0].scatter(working["thresh_gp"], working[gt_var], s=100)

ax[1, 0].plot(comp["thresh_gp"], p.const + p["thresh_gp"] * comp["thresh_gp"], color="black")
ax[1, 0].set_title(gt_var + " vs. GP Predicted Days Over "  + str(thresh) + " Threhsold", fontsize=18)
ax[1, 0].set_ylabel(gt_var, fontsize=18)
ax[1, 0].set_xlabel("Days Above Threshold for the Year", fontsize=18)

print("IDW Days Over Threshold Results:")
#Regression Line (idw)
Y = comp[gt_var]
X = comp["thresh_idw"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#plotting (idw)
for sta in pd.unique(gt_data["Station ID"]):
    #data
    working=comp.loc[comp["Station ID"]==sta]
    
    #errors
    bars=errors.loc[errors["Station ID"]==sta, gt_var]
    ax[1, 1].errorbar(working["thresh_idw"], working[gt_var], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)
    
    #plot data
    ax[1, 1].scatter(working["thresh_idw"], working[gt_var], s=100)
    
ax[1, 1].plot(comp["thresh_idw"], p.const + p["thresh_idw"] * comp["thresh_idw"], color="black")
ax[1, 1].set_title(gt_var +  " vs. IDW Predicted Days Over " + str(thresh) + " Threhsold", fontsize=18)
ax[1, 1].set_ylabel(gt_var, fontsize=18)
ax[1, 1].set_xlabel("Days Above Threshold for the Year", fontsize=18)
fig.tight_layout()

plt.show()
# -

# This is calculated in the same way as shoot count with the same locations only with slightly different data, so all that needs to be changed  from above is the dataframe "gt"

# # Mapping Ground Truth

thresh=25

#Reading in mapping data for actual merging
gt=pd.read_csv("Data/Dominion Energy/Millstone_Eelgrass_Mapping_coastal_features.csv", index_col=0)
gt["Date"]=pd.to_datetime(gt["Date"])
gt["Year"]=gt["Date"].dt.year
gt=gt.loc[gt["Year"].isin(years)]
gt["coords"]=gt.apply(lambda x: (x["Latitude"], x["Longitude"]), axis=1)
gt

# +
#Making Prediction DataFrame for Eelgrass Mapping

gt_for_pred={}

#Making a prediction dataframe for each year in case location changes
for year in years:
    #Restricting to non-continuous data to see how GP fills in data
    working=pd.read_csv("Data/Dominion Energy/Millstone_Eelgrass_Mapping_coastal_features.csv", index_col=0)

    #Creating year variable
    working["Date"]=pd.to_datetime(working["Date"])
    working["Year"]=working["Date"].dt.year

    #Limiting to years of interest
    working = working.loc[working["Year"]==year]
    print(pd.unique(working["Year"]))
    
    #Dropping actual gt variable
    working.drop("Abundance", axis=1, inplace=True)
    
    #Getting sampling locations
    by_station=working.drop_duplicates(subset=["Latitude", "Longitude"]).copy()

    #Adding each day in range for every station samples
    day_list=pd.DataFrame(days, columns=["Day"])
    gt_for_pred[year] = by_station.merge(day_list, how = "cross")
    print(gt_for_pred[year].head())
    print(len(gt_for_pred[year]))

# +
#Modelling summer daily temperatures at ground truthing points
agg=pd.DataFrame()
for year in years:

## Getting temperature from GP Model
    working=gt_for_pred[year]
    print(pd.unique(working["Year"]))
    
    #Normalizing predictors
    X_pred = working[indep_var].values
    X_pred=X_pred - xmeans[year]
    X_pred=X_pred / xstds[year]

    #Using reindexed in prediction
    y_pred, MSE = models[year].predict(X_pred, return_std=True)
    y_pred+=ymeans[year]
    #print(ymeans[year])

    #Adding modeled data back into predictors
    working["Temperature (C)"]=y_pred
    agg=pd.concat([agg, working])
    
##Getting temperature from CTDEEP interpolation
agg =interpolate_means(agg)
agg.loc[(223<=agg["Day"]) & (agg["Day"]<=225)].head()
# -

#Making coords into a tuple so that it's hashable
agg["coords"]=agg["coords"].apply(lambda x: tuple(x))
agg

# +
#Getting years ground truthing in each year


##SUMMER MEAN
for year in years:
    fig, ax = plt.subplots(2, 2)

    working=gt.loc[gt["Year"]==year].copy()

    #Aggregating interpolated ground truth data to get summer average and merging it with gt data
    temp=agg.groupby(["coords", "Year"]).mean().reset_index()
    comp=working.merge(temp, how="left", on=["coords", "Year"]).groupby(["coords", "Year"]).mean().reset_index()
    errors=working.merge(temp, how="left", on=["coords", "Year"]).groupby(["coords", "Year"]).std().reset_index()

    print("Gaussian Process Summer Avg Regression Results:")
    #Regression Line (gp)
    Y = comp["Abundance"]
    X = comp["Temperature (C)"]
    X=sm.add_constant(X)
    model=sm.OLS(Y, X)
    results_gp =model.fit()
    print(results_gp.summary())
    p=results_gp.params

    #Plotting and regression (gp)
    ax[0, 0].errorbar(comp["Temperature (C)"], comp["Abundance"], yerr=errors["Abundance"], fmt="none", zorder=1, color="black", capsize=10)

    ax[0, 0].scatter(comp["Temperature (C)"], comp["Abundance"], s=100)

    ax[0, 0].set_title("Abundance vs. Gaussian Process-Predicted Temperature (Summer Avg)", fontsize=18)
    ax[0, 0].set_ylabel("Abundance (0-5)", fontsize=18)
    ax[0, 0].set_xlabel("Temperature (C)", fontsize=18)

    #Plotting gp reg line
    ax[0, 0].plot(comp["Temperature (C)"], p.const + p["Temperature (C)"] * comp["Temperature (C)"], color="black")

    print("IDW Summer Avg Regression Results:")
    #Regression Line (idw)
    Y = comp["Abundance"]
    X = comp["interpolated"]
    X=sm.add_constant(X)
    model=sm.OLS(Y, X)
    results_idw =model.fit()
    print(results_idw.summary())
    p=results_idw.params

    #plotting (idw)

    ax[0, 1].errorbar(comp["interpolated"], comp["Abundance"], yerr=errors["Abundance"], fmt="none", zorder=1, color="black", capsize=10)

    ax[0, 1].scatter(comp["interpolated"], comp["Abundance"], s=100)

    ax[0, 1].set_title("Abundance vs. CTDEEP IDW-Predicted Temperature (Summer Avg)", fontsize=18)
    ax[0, 1].set_ylabel("Abundance (0-5)", fontsize=18)
    ax[0, 1].set_xlabel("Temperature (C)", fontsize=18)

    #Plotting idw reg line
    ax[0, 1].plot(comp["interpolated"], p.const + p["interpolated"] * comp["interpolated"], color="black")

    ## DAYS OVER THRESHOLD
    comp=pd.DataFrame()

    #Getting days over threshold using the same "agg" from summer means calculation above
    working=agg.copy()

    #Temperature threshold for gaussian process
    working["thresh_gp"]= working["Temperature (C)"]>thresh
    working["thresh_gp"]=working["thresh_gp"].astype(int)

    #Temperature threshold for CTDEEP idw data
    working["thresh_idw"]= working["interpolated"]>thresh
    working["thresh_idw"]=working["thresh_idw"].astype(int)

    to_merge = pd.DataFrame(working.groupby(["coords", "Year"])[["thresh_gp", "thresh_idw"]].sum())
    to_merge.reset_index(inplace=True)
    #print(to_merge)

    working=gt.loc[gt["Year"]==year].copy()

    #Re-merging gt from above with days over threshold
    comp=working.merge(to_merge, how="left", on=["coords", "Year"]).groupby(["coords", "Year"]).mean().reset_index()
    errors = working.merge(to_merge, how="left", on=["coords", "Year"]).groupby(["coords", "Year"]).std().reset_index()

    print(errors.tail())
    print("comp")
    print(comp.tail())

    print("IDW Days Over Threshold Results:")
    #Regression Line (gp)
    Y = comp["Abundance"]
    X = comp["thresh_gp"]
    X=sm.add_constant(X)
    model=sm.OLS(Y, X)
    results_gp =model.fit()
    print(results_gp.summary())
    p=results_gp.params

    #Plotting gp threshold estimates by year to visualize
    #Plotting and regression (gp)

    ax[1, 0].errorbar(comp["thresh_gp"], comp["Abundance"], yerr=errors["Abundance"], fmt="none", zorder=1, color="black", capsize=10)

    #plot data
    ax[1, 0].scatter(comp["thresh_gp"], comp["Abundance"], s=100)

    ax[1, 0].plot(comp["thresh_gp"], p.const + p["thresh_gp"] * comp["thresh_gp"], color="black")
    ax[1, 0].set_title("Abundance vs. GP Predicted Days Over "  + str(thresh) + " Threhsold", fontsize=18)
    ax[1, 0].set_ylabel("Abundance", fontsize=18)
    ax[1, 0].set_xlabel("Days Above Threshold for the Year", fontsize=18)

    print("IDW Days Over Threshold Results:")
    #Regression Line (idw)
    Y = comp["Abundance"]
    X = comp["thresh_idw"]
    X=sm.add_constant(X)
    model=sm.OLS(Y, X)
    results_gp =model.fit()
    print(results_gp.summary())
    p=results_gp.params

    #plotting (idw)

    ax[1, 1].errorbar(comp["thresh_idw"], comp["Abundance"], yerr=errors["Abundance"], fmt="none", zorder=1, color="black", capsize=10)

    #plot data
    ax[1, 1].scatter(comp["thresh_idw"], comp["Abundance"], s=100)

    ax[1, 1].plot(comp["thresh_idw"], p.const + p["thresh_idw"] * comp["thresh_idw"], color="black")
    ax[1, 1].set_title("Abundance vs. IDW Predicted Days Over " + str(thresh) + " Threhsold", fontsize=18)
    ax[1, 1].set_ylabel("Abundance", fontsize=18)
    ax[1, 1].set_xlabel("Days Above Threshold for the Year", fontsize=18)
    fig.tight_layout()

    plt.show()
# -


