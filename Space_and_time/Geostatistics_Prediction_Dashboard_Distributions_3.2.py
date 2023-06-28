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

from osgeo import gdal, gdalconst
import numpy as np
import matplotlib.pyplot as plt

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
lon_min=-72.592354875000
lon_max=-71.811481513000

lat_min=40.970592192000
lat_max=41.545060950000

#Whether or note to restrict to the Eastern Sound window
es=True

#Tags of Organizations for Continous Monitoring
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

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

# ## Reading csv and building model

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
    df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()
    print(len(df))

# # Data summaries

#Summary of df by orgs
working=df.drop_duplicates(["Station ID", "Year"]).copy()
working=working.loc[working["Year"].isin(range(2019, 2022))]
working.replace({"Dominion": "Dominion Power Plant", "EPA_FISM": "FISM"}, inplace=True)
working = working.groupby(["Year", "Organization"])["Station ID"].count()
working=pd.DataFrame(working)
working.rename(columns={"Station ID":"Number of Stations"}, inplace=True)
working.to_csv("Figures_for_Paper/tab1.csv")
working

#Non-continuous data frequencies
working=df.loc[~df["Organization"].isin(cont_orgs)]
working=working.loc[working["Year"].isin(range(2019, 2022))]
working=pd.DataFrame(working.groupby(["Organization", "Station ID", "Year"])["Day"].count())
working.rename(columns={"Day":"Average Number Measurements/Station"}, inplace=True)
working.reset_index(inplace=True)
working=working.groupby(["Year", "Organization"]).mean()
working

df.groupby(["Year", "Organization"]).mean()

pd.unique(df["Organization"])

#Getting list of continuous organizations for alphas (move to YAML)
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]


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

# # Making a version of Data w/ Linear Interpolation for Model Testing

days=range(152, 275)

# +
#Making dataframe with proper index to put interpolation into
df_int=df.copy()

by_station=df_int.drop_duplicates(["Station ID", "Year"])[["Station ID", "Year"]]
print(len(by_station))
ddf=pd.DataFrame(data={"Day":days})

#lengths should match up
cross=by_station.merge(ddf, how="cross")
print(len(cross)/len(days))

reindexed=cross.merge(df_int, how="left", on=["Station ID", "Year", "Day"])
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

# # Parameters and Training Process

#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

# +
#Params
predictors=4
alpha=.25
stride=20
lsb=(1e-5, 1e5)
kernel = 1 * RBF([1]*predictors, length_scale_bounds=lsb) + 1 * RationalQuadratic(length_scale_bounds=lsb)
years = [2019, 2020, 2021]
thresholds = [20, 25]
temp_limits = (19, 27)
day_limits = (0, 50)
#Make sure this matches linear interpolation above
days=range(152, 275)
#continuous monitoring "orgs"
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

#Days for showing model differentiation
display_days={"July": 196, "August": 227}

#Dicts for storage of models
kernels_liss = {2019: 1.06**2 * RBF(length_scale=[0.0334, 0.135, 2.29e+03, 0.185], length_scale_bounds="fixed") + 2.85**2 * RationalQuadratic(alpha=0.266, length_scale=1.85, alpha_bounds="fixed", length_scale_bounds="fixed"),
2020:0.939**2 * RBF(length_scale=[0.0683, 0.106, 7.87, 0.14], length_scale_bounds="fixed") + 5.53**2 * RationalQuadratic(alpha=0.126, length_scale=4.06, alpha_bounds="fixed", length_scale_bounds="fixed"),
2021:1.09**2 * RBF(length_scale=[0.213, 0.159, 4.78, 0.122], length_scale_bounds="fixed") + 1.51**2 * RationalQuadratic(alpha=0.412, length_scale=1.26, alpha_bounds="fixed", length_scale_bounds="fixed")
}
kernels_es = {2019: 0.645**2 * RBF(length_scale=[3.17, 3.04e+03, 6.03, 0.0479]) + 7.4**2 * RationalQuadratic(alpha=0.0217, length_scale=1.29),
2020: 1.3**2 * RBF(length_scale=[4.42, 1e+05, 22.9, 0.054]) + 3.45**2 * RationalQuadratic(alpha=0.0324, length_scale=0.451),
2021: 1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052]) + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538)    
}
#Using 2021 Kernel for All Years
# kernels_es2 = {2019: 1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052]) + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538),
# 2020: 1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052]) + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538),
# 2021: 1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052]) + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538)
# }
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
        models[year], ymeans[year], xmeans[year], xstds[year] = build_model(predictors, year, kernels_liss[year], alpha)
        #print(models[year].kernel_)
else:
#Entire LISS model
    for year in years:
        #Getting models and ymeans outputted
        models[year], ymeans[year] = build_model(predictors, year, kernels_liss[year], alpha)
        print(models[year].kernel_)


assert(os.path.exists('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\'))

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
plt.rcParams["figure.figsize"]=(15, 10)

# +
#Summer Averages histogram

for i, year in enumerate(years):
    
    fig, ax=plt.subplots(3, 1)

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
    ax[0].hist(reindexed["Temperature (C)"], weights=weights, edgecolor="black", color="tab:blue", bins=np.arange(temp_limits[0], temp_limits[1], .2))
    ax[0].set_xlim(temp_limits)
    
    if es:
        ax[0].set_title("ES Summer Average Model Temperature Distribution, " + str(year), fontsize=18)
    else:
        ax[0].set_title("LISS Summer Average Model Temperature Distribution, " + str(year), fontsize=18)

    ax[0].set_ylabel("Frequency")
    ax[0].set_xlabel("Temperature (C)")
    print(np.std(reindexed["Temperature (C)"]))

#Plotting CTDEEP interpolated

    working=ct_deep_int[year]
    working =working.groupby("Station ID").mean()
    
    weights=np.ones_like(working["interpolated"])/len(reindexed)
    ax[1].hist(working["interpolated"], weights=weights, edgecolor="black", color="tab:blue", bins=np.arange(temp_limits[0], temp_limits[1], .2))
    ax[1].set_xlim(temp_limits)
    
    if es:
        ax[1].set_title("ES Summer Average CTDEEP Interpolated Temperature Distribution, " + str(year), fontsize=18)
    else:
        ax[1].set_title("LISS Summer Average CTDEEP Interpolated Temperature Distribution, " + str(year), fontsize=18)

    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Temperature (C)")
    
#Plotting from underlying (linearly interpolated) data
    
    #Averaging year's data by station WITHIN AREA DEFINED ABOVE
    working = df_int.loc[(df_int["Year"]==year)].copy()
    
    #Optional restriction to see if the continuous stations have a distribution that aligns better (Make Sure it Matches Above)
    #working.loc[working["Organization"].isin(cont_orgs)]
    
    #This can be commented out to see histogram of all datapoints (Make Sure it Matches Above)
    working = working.groupby("Station ID").mean()

    #print(working.head())
    weights=np.ones_like(working["Temperature (C)"])/len(working)
    ax[2].hist(working["Temperature (C)"], weights=weights, edgecolor="black", color="tab:orange", bins=np.arange(temp_limits[0], temp_limits[1], .2))
    ax[2].set_xlim(temp_limits)
    print(np.std(working["Temperature (C)"]))
    
    if es:
        ax[2].set_title("ES Summer Average Data Temperature Distribution, " + str(year), fontsize=18)
    else:
        ax[2].set_title("LISS Summer Average Data Temperature Distribution, " + str(year), fontsize=18)
    
    ax[2].set_ylabel("Frequency")
    ax[2].set_xlabel("Temperature (C)")
    if es:
        plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES2_Summer_Average_Distributions_" + str(year) + ".png")
        plt.savefig("../Data Visualization and Analytics Scripts/Graphs/June_Graphs/Eastern_Sound/ES2_Summer_Average_Distributions_" + str(year) + ".png")
    else:
        plt.savefig("Graphs/June_Graphs/LISS_Overall/LISS_Summer_Average_Distributions_" + str(year) + ".png")
        plt.savefig("../Data Visualization and Analytics Scripts/Graphs/June_Graphs/LISS_Overall/LISS_Summer_Average_Distributions_" + str(year) + ".png")
    
    fig.tight_layout()
    plt.show()

# +
#Summer Averages box and whisker

plt.rcParams["figure.figsize"]=(15, 10)

for i, year in enumerate(years):
    lines=[]
    fig, ax = plt.subplots()
    
#Plotting from model
    
    #Getting year's data by station in range defined earlier in notebook
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
    
    #Adding to list to plot
    lines.append(reindexed["Temperature (C)"])
    
#Plotting CTDEEP interpolated

    lines.append(ct_deep_int[year].groupby("Station ID")["interpolated"].mean())

#Plotting from underlying (linearly interpolated) data
        
    #Getting year's data in range defined earlier in notebook
    working = df_int.loc[(df_int["Year"]==year)].copy()
    
    #Optional restriction to see if the continuous stations have a distribution that aligns better (Make Sure it Matches Above)
    #working.loc[working["Organization"].isin(cont_orgs)]
    
    #This can be commented out to see histogram of all datapoints (Make Sure it Matches Above)
    working = working.groupby("Station ID").mean()

    #print(working.head())
    
    #Adding data to list to plot
    lines.append(working["Temperature (C)"])
    
    #Plotting
    plt.boxplot(lines, vert=0)
    plt.xlim(temp_limits)
    if es:
        plt.title("ES Summer Average Model vs. Data Temperature Distribution, " + str(year), fontsize=24)
    else:
        plt.title("LISS Summer Average Model vs. Data Temperature Distribution, " + str(year), fontsize=24)
    ax.set_yticklabels(["GPR Model at Sample Locations", "CT DEEP idw Model", "Data at Sample Locations"])
    ax.set_ylabel("Group")
    ax.set_xlabel("Degrees (C)")
    if es:
        plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES2_Summer_Average_BandW_" + str(year) + ".png")
        plt.savefig("../Data Visualization and Analytics Scripts/Graphs/June_Graphs/Eastern_Sound/ES2_Summer_Average_BandW_" + str(year) + ".png")
    else:
        plt.savefig("Graphs/June_Graphs/LISS_Overall/LISS_Summer_Average_BandW_" + str(year) + ".png")
        plt.savefig("../Data Visualization and Analytics Scripts/Graphs/June_Graphs/LISS_Overall/LISS_Summer_Average_BandW_" + str(year) + ".png")
    plt.show()

# +
#Days over Chosen Threshold(s) box and whisker
plt.rcParams["figure.figsize"]=(30, 30)

#csv for all data
all_data=dict(zip(years, [pd.DataFrame() for i in range(len(years))]))

#Plotting from Model
hfig, hax =plt.subplots(len(years), len(thresholds), squeeze=True)

for i, year in enumerate(years):
    for j, thresh in enumerate(thresholds):
    
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
        
#Plotting from CTDEEP data

        ct_deep_thresh=ct_deep_int[year].copy()
        ct_deep_thresh["Threshold"] = ct_deep_thresh["interpolated"]>thresh
        ct_deep_thresh["Threshold"]=ct_deep_thresh["Threshold"].astype(int)
        ct_deep_thresh = ct_deep_thresh.groupby("Station ID")["Threshold"].sum()*100/ct_deep_thresh.groupby("Station ID")["Threshold"].count()
        
#Plotting from Underlying Lin. Interpolated Data

        #Getting year's data by station in range defined earlier in notebook
        working = df_int.loc[(df_int["Year"]==year)].copy()
        
        #Optional restriction to see if the continuous stations have a distribution that aligns better (make sure it matches below)
        #working=working.loc[working["Organization"].isin(cont_orgs)]
        
        working["Threshold"]= working["Temperature (C)"]>thresh
        working["Threshold"]=working["Threshold"].astype(int)
        working = working.groupby("Station ID")["Threshold"].sum()*100/working.groupby("Station ID")["Threshold"].count()
        #print(working.head())
        
        hax[i, j].boxplot([reindexed, ct_deep_thresh, working], vert=0)
        hax[i, j].set_yticklabels(["GP Model", "CT DEEP IDW Model", "Data"])
        hax[i, j].set_xlabel("Days")
        hax[i, j].set_title("Pct Days over " + str(thresh) + " deg (C) Models vs. Data, " + str(year), fontsize=18)
        hax[i, j].xaxis.set_major_formatter(mtick.PercentFormatter())

#Exporting csv of Results for analysis:
        all_data[year][str(thresh) + " (C)"]=working
        all_data[year]["Year"]=year
        
agg_data=pd.concat(all_data)

fig.suptitle("Eastern Sound", fontsize=18)
if es:
    plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES2_Pct_Days_over_Thresholds_BandW.png")
    plt.savefig("../Data Visualization and Analytics Scripts/Graphs/June_Graphs/Eastern_Sound/ES2_Pct_Days_over_Thresholds_BandW.png")

else:
    plt.savefig("Graphs/June_Graphs/LISS_Overall/LISS_Pct_Days_over_Thresholds_BandW.png")
    plt.savefig("../Data Visualization and Analytics Scripts/Graphs/June_Graphs/Eastern_Sound/LISS_Pct_Days_over_Thresholds_BandW.png")
plt.show()

# -

# # Sample Time Series

# ## Discrete Stations

#Graphing params
plt.rcParams["figure.figsize"]=(30, 20)

#Param
samples= 3

# +
#Script for generating samples

#Restricting to non-continuous data to see how GP fills in data
working=df.loc[~df["Organization"].isin(cont_orgs)].copy()

#Getting stations
by_station=working.drop_duplicates(subset=["Station ID"]).copy()

#Randomly sampling number of samples defined above
by_station=by_station.sample(samples, axis=0)
by_station.drop("Day", axis=1, inplace=True)
#print(by_station)

#Adding each day in range for every station samples
day_list=pd.DataFrame(days, columns=["Day"])
cross = by_station.merge(day_list, how = "cross")
#print(cross)

# +
#Plots

fig, ax=plt.subplots(samples, len(years))

for j, year in enumerate(years):
    
#Generating GP time series for randomly selected stations

    #Merging other variables with station and days
    reindexed=cross[indep_var].copy()

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
    reindexed["Station ID"]=cross["Station ID"]
    #print(reindexed)
    
#Plotting GP time series within each year
    for i, sta in enumerate(pd.unique(reindexed["Station ID"])):
        working=reindexed.loc[reindexed["Station ID"]==sta]
        ax[i, j].plot(working["Day"], working["Temperature (C)"])
        
#Plotting ctdeep time series for station
    for i, sta in enumerate(pd.unique(reindexed["Station ID"])):
        working = ct_deep_int[year]
        working=working.loc[working["Station ID"]==sta]
        ax[i, j].plot(working["Day"], working["interpolated"])

#Plotting original data with time series for comparison
        data = df.loc[(df["Year"]==year) & (df["Station ID"]==sta)].copy()
        ax[i, j].scatter(data["Day"], data["Temperature (C)"], color="tab:orange")
        
#Formatting
        ax[i, j].set_title(sta + " " + str(year), fontsize=24)
        ax[i, j].set_ylim([15, 27])
        ax[i, j].set_xlim([min(days), max(days)])
        ax[i, j].tick_params(axis="x", labelsize=22)
        ax[i, j].tick_params(axis="y", labelsize=22)
        fig.legend(["Gaussian Process", "Inverse Distance Weighting", "Discrete Datapoints"],  prop={'size': 25}, bbox_to_anchor=[1.05, 1.05])
        
fig.suptitle("Fig 3a: Synthesizing Continuous Time Series from Discrete Datapoints", fontsize=32)
fig.supxlabel("Day of the Year", fontsize=24)
fig.supylabel("Temperature (C)", fontsize=24)
plt.tight_layout(pad= 5)

plt.savefig("Figures_for_paper/fig3.png")
plt.show()
# -
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
    #print(cross)
    
# -

#keys for figures made from niantic-specific samples
keys=dict(zip(range(len(by_station)), "abc"))

# +
#Plots

for i, sta in enumerate(pd.unique(by_station["Station ID"])):
    
    #Each ground truthed station gets its own plot
    fig, ax=plt.subplots(len(years))

    for j, year in enumerate(years):
    
#Generating GP time series for each ground truthing station in Millstone Environmental Lab dataset

        #Merging other variables with station and days
        reindexed=gt_for_pred[year][indep_var].copy()
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
        reindexed["Station ID"]=gt_for_pred[year]["Station ID"]
        #print(reindexed)

#Plotting time series within each year
        working=reindexed.loc[reindexed["Station ID"]==sta]
        ax[j].plot(working["Day"], working["Temperature (C)"])
        ax[j].set_title("Millstone Station " + sta + ", " + str(year) + " w/ Controls", fontsize=24)
        
#Plotting ct_deep_idw temp
        working=gt_for_pred[year]
        working["Year"]=year
        working=interpolate_means(working)
        working=working.loc[working["Station ID"]==sta]
        ax[j].plot(working["Day"], working["interpolated"])
    

        for k, control in enumerate(controls):
#Plotting niantic_controls for comparison
            data = df.loc[(df["Year"]==year) & (df["Station ID"]==control)].copy()
            ax[j].plot(data["Day"], data["Temperature (C)"])
        
#Adding legend to jth chart
        ax[j].legend([sta + " GP", sta + " CTDEEP IDW"]+list(controls), fontsize=22)
#Formatting all axes
        for axis in fig.get_axes():
            axis.set_ylim([15, 27])
            axis.set_xlim([min(days), max(days)])
            axis.tick_params(axis="x", labelsize=22)
            axis.tick_params(axis="y", labelsize=22)
    
    fig.suptitle("Fig 4" + keys[i] + ": Synthesizing Continuous Time Series for Millstone Eelgrass Station " + str(sta), fontsize=32)
    fig.supxlabel("Day of the Year", fontsize=24)
    fig.supylabel("Temperature (C)", fontsize=24)
    plt.tight_layout(pad= 5)

    plt.savefig("Figures_for_paper/fig4" + keys[i] + ".png")
    plt.show()
# -

# # Ground Truth


import statsmodels.api as sm

gt=pd.read_csv("Data/Dominion Energy/Millstone_Shoot_Counts_coastal_features.csv", index_col=0)
gt["Date"]=pd.to_datetime(gt["Date"])
gt["Day"]=gt["Date"].dt.day_of_year
gt["Year"]=gt["Date"].dt.year
gt=gt.loc[gt["Year"].isin(years)]
gt

#Ground Truth Data Summary Statistics
working=gt.loc[gt["Year"].isin(range(2019, 2022))]
working=working["Day"].describe()
pd.DataFrame(working)

# # Trailing n day mean

n=60

#Graphing param
plt.rcParams["figure.figsize"]=(24, 8)

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
    working["Temperature (C)"]=y_pred
    working=interpolate_means(working)
    working["Temperature (C)"]=working["Temperature (C)"].rolling(n, min_periods=1).mean()
    working["interpolated"]=working["interpolated"].rolling(n, min_periods=1).mean()

    agg=pd.concat([agg, working])
    
    
##Getting temperature from CTDEEP interpolation
agg.loc[(223<=agg["Day"]) & (agg["Day"]<=225)].head()
# +
#Getting years ground truthing (shoot count)
working=gt.loc[gt["Year"].isin(years)].copy()
errors=working.groupby(["Station ID", "Year", "Day"]).std().reset_index()
working=working.groupby(["Station ID", "Year", "Day"]).mean().reset_index()
working.rename(columns={"Number of Vegetative Shoots":"shoot_count"}, inplace=True)
errors.rename(columns={"Number of Vegetative Shoots":"shoot_count"}, inplace=True)

#Merging agg data with gt data
comp=working.merge(agg, how="left", on=["Station ID", "Year", "Day"])
print(comp)

#Regression Line (gp)
Y = comp["shoot_count"]
X = comp["Temperature (C)"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#Plotting and regression (gp)
for sta in pd.unique(gt["Station ID"]):
    working=comp.loc[comp["Station ID"]==sta]
    
    bars=errors.loc[errors["Station ID"]==sta,"shoot_count"]
    plt.errorbar(working["Temperature (C)"], working["shoot_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    plt.scatter(working["Temperature (C)"], working["shoot_count"], s=100)
    
    plt.title("Shoot Count vs. Gaussian Process-Predicted Temperature, 60 Day Trailing Avg", fontsize=18)
    plt.ylabel("Shoot Count", fontsize=18)
    plt.xlabel("Temperature (C)", fontsize=18)

#Plotting gp reg line
plt.plot(comp["Temperature (C)"], p.const + p["Temperature (C)"] * comp["Temperature (C)"], color="black")
plt.legend(pd.unique(gt["Station ID"]))
plt.show()

#Regression Line (idw)
Y = comp["shoot_count"]
X = comp["interpolated"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_idw =model.fit()
print(results_idw.summary())
p=results_idw.params

#plotting (idw)
for sta in pd.unique(gt["Station ID"]):
    working=comp.loc[comp["Station ID"]==sta]
    
    bars=errors.loc[errors["Station ID"]==sta,"shoot_count"]
    plt.errorbar(working["interpolated"], working["shoot_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    plt.scatter(working["interpolated"], working["shoot_count"], s=100)

    plt.title("Shoot Count vs. CTDEEP IDW-Predicted Temperature, " + str(n) + " Day Trailing Avg", fontsize=18)
    plt.ylabel("Shoot Count", fontsize=18)
    plt.xlabel("Temperature (C)", fontsize=18)

#Plotting idw reg line
plt.plot(comp["interpolated"], p.const + p["interpolated"] * comp["interpolated"], color="black")
plt.legend(pd.unique(gt["Station ID"]))
plt.show()
# -

# ## Shoot Count w/ Mean Temeprature and Days Over Threshold

#Setting threshold
thresh=20

#Graphing param
plt.rcParams["figure.figsize"]=(30, 20)

# +
#Modelling summer average temperatures at ground truthing points
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
#Getting years ground truthing (shoot count)

fig, ax = plt.subplots(2, 2)

##SUMMER MEAN

working=gt.loc[gt["Year"].isin(years)].copy()
working.rename(columns={"Number of Vegetative Shoots":"shoot_count"}, inplace=True)

#Aggregating interpolated ground truth data to get summer average and merging it with gt data
temp=agg.groupby(["Station ID", "Year"]).mean().reset_index()
comp=working.merge(temp, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).mean().reset_index()
errors=working.merge(temp, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).std().reset_index()

print("Gaussian Process Summer Avg Regression Results:")
#Regression Line (gp)
Y = comp["shoot_count"]
X = comp["Temperature (C)"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#Plotting and regression (gp)
for sta in pd.unique(gt["Station ID"]):
    working=comp.loc[comp["Station ID"]==sta]
    
    bars=errors.loc[errors["Station ID"]==sta,"shoot_count"]
    ax[0, 0].errorbar(working["Temperature (C)"], working["shoot_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    ax[0, 0].scatter(working["Temperature (C)"], working["shoot_count"], s=100)
    
    ax[0, 0].set_title("Shoot Count vs. Gaussian Process-Predicted Temperature (Summer Avg)", fontsize=18)
    ax[0, 0].set_ylabel("Shoot Count", fontsize=18)
    ax[0, 0].set_xlabel("Temperature (C)", fontsize=18)

#Plotting gp reg line
ax[0, 0].plot(comp["Temperature (C)"], p.const + p["Temperature (C)"] * comp["Temperature (C)"], color="black")
fig.legend(pd.unique(gt["Station ID"]), prop={'size': 25}, bbox_to_anchor=[1.05, 1.05])

print("IDW Summer Avg Regression Results:")
#Regression Line (idw)
Y = comp["shoot_count"]
X = comp["interpolated"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_idw =model.fit()
print(results_idw.summary())
p=results_idw.params

#plotting (idw)
for sta in pd.unique(gt["Station ID"]):
    working=comp.loc[comp["Station ID"]==sta]
    
    bars=errors.loc[errors["Station ID"]==sta,"shoot_count"]
    ax[0, 1].errorbar(working["interpolated"], working["shoot_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    ax[0, 1].scatter(working["interpolated"], working["shoot_count"], s=100)

    ax[0, 1].set_title("Shoot Count vs. CTDEEP IDW-Predicted Temperature (Summer Avg)", fontsize=18)
    ax[0, 1].set_ylabel("Shoot Count", fontsize=18)
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

working=gt.loc[gt["Year"].isin(years)].copy()
working.rename(columns={"Number of Vegetative Shoots":"shoot_count"}, inplace=True)

#Re-merging gt from above with days over threshold
comp=working.merge(to_merge, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).mean().reset_index()
errors = working.merge(to_merge, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).std().reset_index()

print(errors.tail())
print("comp")
print(comp.tail())

print("IDW Days Over Threshold Results:")
#Regression Line (gp)
Y = comp["shoot_count"]
X = comp["thresh_gp"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#Plotting gp threshold estimates by year to visualize
#Plotting and regression (gp)
for sta in pd.unique(gt["Station ID"]):
    #data
    working=comp.loc[comp["Station ID"]==sta]

    #erors
    bars=errors.loc[errors["Station ID"]==sta,"shoot_count"]
    ax[1, 0].errorbar(working["thresh_gp"], working["shoot_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    #plot data
    ax[1, 0].scatter(working["thresh_gp"], working["shoot_count"], s=100)

ax[1, 0].plot(comp["thresh_gp"], p.const + p["thresh_gp"] * comp["thresh_gp"], color="black")
ax[1, 0].set_title("Shoot Count vs. GP Predicted Days Over "  + str(thresh) + " Threhsold", fontsize=18)
ax[1, 0].set_ylabel("Shoot Count", fontsize=18)
ax[1, 0].set_xlabel("Days Above Threshold for the Year", fontsize=18)

print("IDW Days Over Threshold Results:")
#Regression Line (idw)
Y = comp["shoot_count"]
X = comp["thresh_idw"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#plotting (idw)
for sta in pd.unique(gt["Station ID"]):
    #data
    working=comp.loc[comp["Station ID"]==sta]
    
    #errors
    bars=errors.loc[errors["Station ID"]==sta, "shoot_count"]
    ax[1, 1].errorbar(working["thresh_idw"], working["shoot_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)
    
    #plot data
    ax[1, 1].scatter(working["thresh_idw"], working["shoot_count"], s=100)
    
ax[1, 1].plot(comp["thresh_idw"], p.const + p["thresh_idw"] * comp["thresh_idw"], color="black")
ax[1, 1].set_title("Shoot Count vs. IDW Predicted Days Over " + str(thresh) + " Threhsold", fontsize=18)
ax[1, 1].set_ylabel("Shoot Count", fontsize=18)
ax[1, 1].set_xlabel("Days Above Threshold for the Year", fontsize=18)
fig.tight_layout()

plt.show()
# -

# ## Repro Count w/ Mean Temeprature and Days Over Threshold


# This is calculated in the same way as shoot count with the same locations only with slightly different data, so all that needs to be changed  from above is the dataframe "gt"

# ## Reading in repro counts

gt=pd.read_csv("Data/Dominion Energy/Millstone_Repro_Counts.csv")
gt=gt.loc[gt["Year"].isin(years)]
gt.replace({"Jordan Cove": "JC", "Niantic River": "NR", "White Point":"WP"}, inplace=True)
gt.rename(columns={"Station": "Station ID"}, inplace=True)
gt

# ## Plotting Statistics

#Setting threshold
thresh=20

#Graphing param
plt.rcParams["figure.figsize"]=(30, 20)

# +
#Modelling summer average temperatures at ground truthing points
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
#Getting years ground truthing (shoot count)

fig, ax = plt.subplots(2, 2)

##SUMMER MEAN

working=gt.loc[gt["Year"].isin(years)].copy()
working.rename(columns={"ReproCount":"repro_count"}, inplace=True)

#Aggregating interpolated ground truth data to get summer average and merging it with gt data
temp=agg.groupby(["Station ID", "Year"]).mean().reset_index()
comp=working.merge(temp, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).mean().reset_index()
errors=working.merge(temp, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).std().reset_index()

print("Gaussian Process Summer Avg Regression Results:")
#Regression Line (gp)
Y = comp["repro_count"]
X = comp["Temperature (C)"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#Plotting and regression (gp)
for sta in pd.unique(gt["Station ID"]):
    working=comp.loc[comp["Station ID"]==sta]
    
    bars=errors.loc[errors["Station ID"]==sta,"repro_count"]
    ax[0, 0].errorbar(working["Temperature (C)"], working["repro_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    ax[0, 0].scatter(working["Temperature (C)"], working["repro_count"], s=100)
    
    ax[0, 0].set_title("Reproductive Count vs. Gaussian Process-Predicted Temperature (Summer Avg)", fontsize=18)
    ax[0, 0].set_ylabel("Shoot Count", fontsize=18)
    ax[0, 0].set_xlabel("Temperature (C)", fontsize=18)

#Plotting gp reg line
ax[0, 0].plot(comp["Temperature (C)"], p.const + p["Temperature (C)"] * comp["Temperature (C)"], color="black")
fig.legend(pd.unique(gt["Station ID"]), prop={'size': 25}, bbox_to_anchor=[1.05, 1.05])

print("IDW Summer Avg Regression Results:")
#Regression Line (idw)
Y = comp["repro_count"]
X = comp["interpolated"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_idw =model.fit()
print(results_idw.summary())
p=results_idw.params

#plotting (idw)
for sta in pd.unique(gt["Station ID"]):
    working=comp.loc[comp["Station ID"]==sta]
    
    bars=errors.loc[errors["Station ID"]==sta,"repro_count"]
    ax[0, 1].errorbar(working["interpolated"], working["repro_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    ax[0, 1].scatter(working["interpolated"], working["repro_count"], s=100)

    ax[0, 1].set_title("Reproductive Count vs. CTDEEP IDW-Predicted Temperature (Summer Avg)", fontsize=18)
    ax[0, 1].set_ylabel("Shoot Count", fontsize=18)
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

working=gt.loc[gt["Year"].isin(years)].copy()
working.rename(columns={"ReproCount":"repro_count"}, inplace=True)

#Re-merging gt from above with days over threshold
comp=working.merge(to_merge, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).mean().reset_index()
errors = working.merge(to_merge, how="left", on=["Station ID", "Year"]).groupby(["Station ID", "Year"]).std().reset_index()

print(errors.tail())
print("comp")
print(comp.tail())

print("IDW Days Over Threshold Results:")
#Regression Line (gp)
Y = comp["repro_count"]
X = comp["thresh_gp"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#Plotting gp threshold estimates by year to visualize
#Plotting and regression (gp)
for sta in pd.unique(gt["Station ID"]):
    #data
    working=comp.loc[comp["Station ID"]==sta]

    #erors
    bars=errors.loc[errors["Station ID"]==sta,"repro_count"]
    ax[1, 0].errorbar(working["thresh_gp"], working["repro_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)

    #plot data
    ax[1, 0].scatter(working["thresh_gp"], working["repro_count"], s=100)

ax[1, 0].plot(comp["thresh_gp"], p.const + p["thresh_gp"] * comp["thresh_gp"], color="black")
ax[1, 0].set_title("Reproductive Count vs. GP Predicted Days Over "  + str(thresh) + " Threhsold", fontsize=18)
ax[1, 0].set_ylabel("Shoot Count", fontsize=18)
ax[1, 0].set_xlabel("Days Above Threshold for the Year", fontsize=18)

print("IDW Days Over Threshold Results:")
#Regression Line (idw)
Y = comp["repro_count"]
X = comp["thresh_idw"]
X=sm.add_constant(X)
model=sm.OLS(Y, X)
results_gp =model.fit()
print(results_gp.summary())
p=results_gp.params

#plotting (idw)
for sta in pd.unique(gt["Station ID"]):
    #data
    working=comp.loc[comp["Station ID"]==sta]
    
    #errors
    bars=errors.loc[errors["Station ID"]==sta, "repro_count"]
    ax[1, 1].errorbar(working["thresh_idw"], working["repro_count"], yerr=bars, fmt="none", zorder=1, color="black", capsize=10)
    
    #plot data
    ax[1, 1].scatter(working["thresh_idw"], working["repro_count"], s=100)
    
ax[1, 1].plot(comp["thresh_idw"], p.const + p["thresh_idw"] * comp["thresh_idw"], color="black")
ax[1, 1].set_title("Reproductive Count vs. IDW Predicted Days Over " + str(thresh) + " Threhsold", fontsize=18)
ax[1, 1].set_ylabel("Shoot Count", fontsize=18)
ax[1, 1].set_xlabel("Days Above Threshold for the Year", fontsize=18)
fig.tight_layout()

plt.show()
# -






