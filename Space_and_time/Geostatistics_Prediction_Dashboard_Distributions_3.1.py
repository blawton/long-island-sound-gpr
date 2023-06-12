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

#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

# +
#Global Params:

#Defining Eastern Sound Window (useful for visualizing cropped distance map)
lon_min=-72.592354875000
lon_max=-71.811481513000

lat_min=40.970592192000
lat_max=41.545060950000

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
paths[5] =  "Data/Space_and_time_agg/agg_daily_morning_coastal_features_5_17_2023.csv"

for path in paths.values():
    assert os.path.exists(path), path
# -

# # Mainpulating Coastal Distance to Use in Regression

#Getting open sound as formatted in ArcGIS Pro
open_sound = gdal.Open(paths[1])
band=open_sound.GetRasterBand(1)
array_os=band.ReadAsArray()

#Getting all sound from state lines (solely to remove Island from Mystic and other land patches)
all_sound = gdal.Open(paths[2])
band=all_sound.GetRasterBand(1)
array_all=band.ReadAsArray()

#embayment distance including only MEASURED Vaudrey embayments
embay_dist = gdal.Open(paths[3])

gt=embay_dist.GetGeoTransform()
gt

proj=embay_dist.GetProjection()
proj

band=embay_dist.GetRasterBand(1)
array_embay_dist=band.ReadAsArray()

pixelwidth=gt[1]
pixelheight=-gt[5]
pixelheight

xOrigin = gt[0]
yOrigin = gt[3]

cols = embay_dist.RasterXSize
rows = embay_dist.RasterYSize
cols

array_os

array_embay_dist

#Starting visualization
plt.imshow(array_embay_dist)

#Getting rid of mystic (state boundaries need to be double checked with os to avoid making fisher's island embayments)
array=np.where(array_all!=0, array_embay_dist, np.nan)
array=np.where(np.isnan(array), array_os, array)

#Setting negatives to nan for visualization
array=np.where(array<0, np.nan, array)

#Final Array only for embayments
embay_only=np.where(array_embay_dist>0, array, np.nan)
plt.imshow(embay_only)
plt.axis("off")

# +
# #Outputting Final Embayment Distance to avoid repeating above
# driver = gdal.GetDriverByName("GTiff")
# driver.Register()
# outds = driver.Create(paths[6], xsize=cols, ysize=rows, bands=1, eType=gdal.GDT_Int16)
# outds.SetGeoTransform(gt)
# outds.SetProjection(proj)
# outband = outds.GetRasterBand(1)
# outband.WriteArray(embay_only)
# outband.SetNoDataValue(np.nan)
# outband.FlushCache()
# outband=None
# outds=None
# -

# # Reading in Data and Building in Model

# ## Reading coastal features array

#Using the entire sound (not just eastern sound)
array=embay_only

gt=gdal.Open(paths[6]).GetGeoTransform()
gt

array

#Setting negatives to nan
array=np.where(array<0, np.nan, array)
array

#Setting zeroes to nan
array=np.where(array==0, np.nan, array)
array

array.shape

plt.imshow(array)

# ## Reading csv and building model

#This is the source of the model's predictions
df=pd.read_csv(paths[5], index_col=0)
df.head(50)

#Checking Dominion Stations to Make Sure they are in Vaudrey Embayment (C and NB)
df.loc[df["Organization"]=="Dominion"]

#Dropping nas
print(len(df))
df.dropna(subset=["Station ID", "Longitude", "Latitude", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

#Optional Restriction of TRAINING and later testing params to Eastern Sound
df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()
print(len(df))

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
    X_train=X_train - np.mean(X_train, axis=0)
    X_train=X_train / np.std(X_train, axis=0)
    
    #Constructing Process
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=alpha)


    #Training Process
    gaussian_process.fit(X_train, y_train)

    return(gaussian_process, y_mean)

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
thresholds = [24, 24.5, 25]
temp_limits = (19, 27)
day_limits = (0, 50)
days=range(172, 244)
#continuous monitoring "orgs"
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

#Days for showing model differentiation
display_days={"July": 196, "August": 227}

#Dicts for storage of models
kernels = {}
errors= {}
models = {}
hmaps= {}
ymeans = {}
# -

for year in years:
    #Getting models and ymeans outputted
    models[year], ymeans[year] = build_model(predictors, year, kernel, alpha)

# # Plots of distributions

cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

# +
#Summer Averages histogram

plt.rcParams["figure.figsize"]=(15, 10)

for i, year in enumerate(years):
    
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
    print(reindexed.head)
    
    #Normalizing predictors
    X_pred = reindexed.values
    X_pred=X_pred - np.mean(X_pred, axis=0)
    X_pred=X_pred / np.std(X_pred, axis=0)
    
    #Using reindexed in prediction
    y_pred, MSE = models[year].predict(X_pred, return_std=True)
    y_pred+=ymeans[year]
    print(ymeans[year])
    
    #Adding modeled data back into predictors
    reindexed["Temperature (C)"]=y_pred
    #print(reindexed)
    
    #Adding back in station ID
    reindexed["Station ID"]=cross["Station ID"]
    #print(reindexed)
    
    #This can be commented out to see histogram of all datapoints (Make Sure it Matches Below)
    reindexed=reindexed.groupby("Station ID").mean()
    #print(reindexed)
    
    #Plotting from model
    weights=np.ones_like(reindexed["Temperature (C)"])/len(reindexed)
    plt.hist(reindexed["Temperature (C)"], weights=weights, edgecolor="black", bins=20)
    plt.xlim(temp_limits)
    plt.title("ES Summer Average Model Temperature Distribution, " + str(year), fontsize=24)

    plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES_Summer_Average_Distribution_Model_" + str(year) + ".png")
    plt.show()
    
    #Plotting from underlying data
    
    fig, ax = plt.subplots()
    
    #Averaging year's data by station WITHIN AREA DEFINED ABOVE
    working = df.loc[(df["Year"]==year)].copy()
    
    #Optional restriction to see if the continuous stations have a distribution that aligns better (Make Sure it Matches Above)
    #working.loc[working["Organization"].isin(cont_orgs)]
    
    #This can be commented out to see histogram of all datapoints (Make Sure it Matches Above)
    working = working.groupby("Station ID").mean()

    print(working.head())
    weights=np.ones_like(working["Temperature (C)"])/len(working)
    plt.hist(working["Temperature (C)"], weights=weights, edgecolor="black", bins=20)
    plt.xlim(temp_limits)
    plt.title("ES Summer Average Data Temperature Distribution, " + str(year), fontsize=24)

    plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES_Summer_Average_Distribution_Data_" + str(year) + ".png")
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
    X_pred=X_pred - np.mean(X_pred, axis=0)
    X_pred=X_pred / np.std(X_pred, axis=0)
    
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
    
    #Plotting from underlying data
        
    #Getting year's data in range defined earlier in notebook
    working = df.loc[(df["Year"]==year)].copy()
    
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
    plt.title("ES Summer Average Model vs. Data Temperature Distribution, " + str(year), fontsize=24)
    ax.set_yticklabels(["Model at Sample Locations", "Data at Sample Locations"])
    ax.set_ylabel("Group")
    ax.set_xlabel("Degrees (C)")
    plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES_Summer_Average_BandW_" + str(year) + ".png")
    plt.show()

# +
#Days over Chosen Threshold(s) box and whisker
plt.rcParams["figure.figsize"]=(30, 30)

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
        X_pred=X_pred - np.mean(X_pred, axis=0)
        X_pred=X_pred / np.std(X_pred, axis=0)

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
        reindexed = reindexed.groupby("Station ID")["Threshold"].sum()/reindexed.groupby("Station ID")["Threshold"].count()
        
#Plotting from Underlying Data

        #Getting year's data by station in range defined earlier in notebook
        working = df.loc[(df["Year"]==year)].copy()
        working["Threshold"]= working["Temperature (C)"]>thresh
        working["Threshold"]=working["Threshold"].astype(int)
        working = working.groupby("Station ID")["Threshold"].sum()/working.groupby("Station ID")["Threshold"].count()
        #print(working.head())
        
        
        hax[i, j].boxplot([reindexed, working], vert=0)
        hax[i, j].set_yticklabels(["Model", "Data"])
        hax[i, j].set_xlabel("Days")
        hax[i, j].set_title("Pct Days over " + str(thresh) + " deg (C) Model vs. Data, " + str(year), fontsize=18)
fig.suptitle("Eastern Sound", fontsize=18)        
plt.savefig("Graphs/June_Graphs/Eastern_Sound/ES_Pct_Days_over_Thresholds_BandW.png")
plt.show()

# -

# # Outputting TIFFs

new_gt=(gt[0], gt[1]*stride, gt[2], gt[3], gt[4], gt[5]*stride)
new_gt

years = [2019, 2020, 2021]
for year in years:
    #Downloading geotiff of model
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    name = paths[7] + "Temperature_Model_3_" + str(year) + ".tif"
    outds = driver.Create(name, xsize=models[year].shape[1], ysize=models[year].shape[0], bands=1, eType=gdal.GDT_Int16)
    outds.SetGeoTransform(new_gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(models[year])
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()
    outband=None
    outds=None
    
    #Downloading geotiff of error
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    name = paths[7] + "Temperature_Model_3_Error_" + str(year) + ".tif"
    outds = driver.Create(name, xsize=errors[year].shape[1], ysize=errors[year].shape[0], bands=1, eType=gdal.GDT_Int16)
    outds.SetGeoTransform(new_gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(errors[year])
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()
    outband=None
    outds=None

#Testing saved geotiff
name = paths[7] + "Temperature_Model_3_2021.tif"
test=gdal.Open(name)
band=test.GetRasterBand(1)
array=band.ReadAsArray()
plt.imshow(array)

gt=test.GetGeoTransform()
gt


