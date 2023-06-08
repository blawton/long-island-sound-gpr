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

# +
# #Displaying crucial regions
# plt.figure()
# plt.imshow(np.isnan(array))
# plt.figure()
# plt.imshow(array==0)
# plt.figure()
# plt.imshow(array<0)
# -

#Starting visualization
plt.imshow(array_embay_dist)

# +
# #Setting open sound (non vaudrey embayments) to 0
# array=np.where(array_os>=0, 0, array_embay_dist)
# -

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

# ## Reading array

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

# +
# #Optional Restriction of TRAINING params to Eastern Sound
# df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()
# print(len(df))
# -

# ^The above restriction destroys the model's ability to find meaningful hyperparameters, which shows that training on the LIS as a whole is clearly preferable

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


# # Setting Up Output Heatmap

# +
#Its crucial for the variables in this function to match the order of station_var+indep_var+dep_var 
#(year var added for time func.)
def get_heatmap_inputs(stride, coastal_map, gt, day):
    
    #Reading in geotransform
    pixelwidth=gt[1]
    pixelheight=-gt[5]
    
    xOrigin = gt[0]
    yOrigin = gt[3]
    
    #New array of coastal features
    resampled=coastal_map[0::stride, 0::stride]
    
    #Calculating longitudes of Resampled Array
    
    lons=np.arange(resampled.shape[1])*(pixelwidth)*(stride) + xOrigin
    #Centering
    lons +=pixelwidth/2
    
#     print("Lons")
#     print(min(lons))
#     print(max(lons))
    #Calculating lattitudes of Resampled Array
    
    lats= -np.arange(resampled.shape[0])*(pixelheight)*(stride) + yOrigin
    #Centering
    lats +=pixelheight/2
#     print("Lats")
#     print(min(lats))
#     print(max(lats))
    
    #Making latitude and longitude into a grid
    lonv, latv = np.meshgrid(lons, lats)
    
    #Flattening predictors to run through gaussian_process.predict (adding year)
    X_pred = np.column_stack((lonv.flatten(), latv.flatten(), resampled.flatten(), np.repeat(day, len(resampled.flatten()))))
    
    return(X_pred, resampled, resampled.shape)


# -

# # Model Output

# Initial guesses for model length scales are important because they prevent the model from going into an unstable mode where the middle 2nd length scale parameter blows up. Edit: this still happens by chance even with an initial guess of .1

#Coloring scales
hmin=20
hmax=26
sdmin=0
sdmax=1.2


def model_output(year, stride, process, ymean, day):
    
    #Building Heatmap
    X_pred, resampled, grid_shape = get_heatmap_inputs(stride, array, gt, day)
    
    #Creating y based on X
    X_pred=pd.DataFrame(X_pred)
    y_pred=pd.DataFrame(index=X_pred.index)
    
    #Filtering to prediction region where embay dist is non-nan
    X_pred_filt= X_pred.dropna(axis=0)
    
    #Standardizing X (make sure this matches build_model above)
    period=df.loc[df["Year"]==year, station_var+indep_var+dep_var]
    data=period.values
    X_train = data[:, 1:predictors+1]
    
    #Ensuring proper dtypes
    X_train=np.array(X_train, dtype=np.float64)    
    
    #Using training data to normalize predictors to match trained model
    X_pred_filt = X_pred_filt - np.mean(X_train, axis=0)
    X_pred_filt = X_pred_filt / np.std(X_train, axis=0)
            
    #Running process
    y_pred_filt, MSE = process.predict(X_pred_filt.values, return_std=True)
    y_pred_filt+=ymean
    
    #Adding nas to y dataset
    y_pred["Temp (C)"]=np.nan
    y_pred.loc[X_pred_filt.index, "Temp (C)"]=y_pred_filt
    
    #Creating standard error dataset
    std=pd.DataFrame(index=X_pred.index)
    std["Standard Deviation"]=np.nan
    std.loc[X_pred_filt.index, "Standard Deviation"]=MSE
    
    #Converting Output to Grid
    y_grid = y_pred.values.reshape(grid_shape)
    #y_grid = np.where(resampled!=0, y_grid, np.nan)
    std_grid= std.values.reshape(grid_shape)
    
    return(std_grid, y_grid)


# ## Parameters and Training Process

#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

# +
#Params
predictors=4
alpha=.25
stride=5
lsb=(1e-5, 1e5)
kernel = 1 * RBF([1]*predictors, length_scale_bounds=lsb) + 1 * RationalQuadratic(length_scale_bounds=lsb)
years = [2019, 2020, 2021]

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

# ## Heatmaps

# +
#Graphing selected days above

plt.rcParams["figure.figsize"]=(40, 60)

#Making Figures
hfig, hax =plt.subplots(len(years), len(display_days), gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)
efig, eax =plt.subplots(len(years), len(display_days), gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)

#Plotting
for i, year in enumerate(years):
    for j, day in enumerate(display_days.values()):     
                error, hmap = model_output(year, stride, models[year], ymeans[year], day)

                #hmap plot
                him = hax[i, j].imshow(hmap, vmin=20, vmax=26)
                hax[i, j].set_title("Temperature: " + list(display_days.keys())[j] + " " + str(year), fontsize=24)
                
                #error plot
                eim = eax[i, j].imshow(error, vmin=0, vmax=1.2)
                eax[i, j].set_title("Standard Error: " + list(display_days.keys())[j] + " " + str(year), fontsize=24)
                
#Axes off
[ax.set_axis_off() for ax in hax.ravel()]
[ax.set_axis_off() for ax in eax.ravel()]

#hmap cbar
cax = hfig.add_axes([0.1,0.05,0.8,0.03])
cbar = hfig.colorbar(him, cax=cax, orientation='horizontal')
cbar.set_label('Deg C')

#error cbar
cax = efig.add_axes([0.1,0.05,0.8,0.03])
cbar = efig.colorbar(eim, cax=cax, orientation='horizontal')
cbar.set_label('Standard Deviation in Deg C')

hfig.savefig("Graphs/Time_Dependent_Heatmaps.png")
efig.savefig("Graphs/Time_Dependent_Errors.png")

plt.show()
# -


#Re-setting threshold params
thresholds = [24, 24.5, 25]
stride =20
days=range(172, 244)

# +
#Summer Averages

plt.rcParams["figure.figsize"]=(30, 20)

#Tallying
for i, year in enumerate(years):
    total = np.zeros(1)
    fig, ax = plt.subplots()
    for day in days:
        error, hmap = model_output(year, stride, models[year], ymeans[year], day)
        if len(total)>1:
            total += hmap
        else:
            total = hmap
    total = total/len(range(172, 244))
    avg = plt.imshow(total)
    plt.title("Summer Average Temperature, " + str(year), fontsize=24)
    cbar = fig.colorbar(avg, location="right", shrink=.75)
    cbar.set_label('Deg C', rotation=270, fontsize=24)
    cbar.ax.tick_params(labelsize=24)
    plt.axis("off")
    plt.savefig("Graphs/Time_Dependent_Summer_Average_" + str(year) + ".png")
    plt.show()
    


# +
#Days over Chosen Threshold(s)
plt.rcParams["figure.figsize"]=(60, 60)

#Making Figures
hfig, hax =plt.subplots(len(years), len(thresholds), gridspec_kw={'wspace':0, 'hspace':0}, squeeze=True)

#Plotting
for i, year in enumerate(years):
    for j, thresh in enumerate(thresholds):
        total = np.zeros(1)
        for day in  days:
            error, hmap = model_output(year, stride, models[year], ymeans[year], day)

            #Establishing threshold as the cut
            over = np.where(hmap>thresh, 1, 0)

            #Readding nans for better visualization
            over=np.where(np.isnan(hmap), np.nan, over)

            if len(total)>1:
                total += over
            else:
                total = over
        him = hax[i, j].imshow(total, vmin=0, vmax=10)
        hax[i, j].set_title("Days over Threshold of " + str(thresh) + " degrees (C), " + str(year), fontsize=24)
cax = hfig.add_axes([0.1,0.05,0.8,0.03])
cbar = hfig.colorbar(him, cax=cax, orientation='horizontal')
cbar.set_label('Count of Days', fontsize=24)
cbar.ax.tick_params(labelsize=24)
[ax.set_axis_off() for ax in hax.ravel()]
plt.savefig("Graphs/Days_over_Thresholds.png")
plt.show()
# -


# ## Plots of distributions

temp_limits = (19, 27)
day_limits = (0, 50)
stride =20
days=range(172, 244)

# +
#Summer Averages

plt.rcParams["figure.figsize"]=(15, 10)

for i, year in enumerate(years):
    total = np.zeros(1)
    #Plotting from Model
    fig, ax = plt.subplots()
    for day in days:
        error, hmap = model_output(year, stride, models[year], ymeans[year], day)
        if len(total)>1:
            total += hmap
        else:
            total = hmap
    total = total/len(range(172, 244))
    plt.hist(total.flatten(), bins=20)
    plt.xlim(temp_limits)
    plt.title("Summer Average Model Temperature Distribution, " + str(year), fontsize=24)

    plt.savefig("Graphs/Time_Dependent_Summer_Average_Distribution_Model" + str(year) + ".png")
    plt.show()
    
    #Plotting from underlying data
    fig, ax = plt.subplots()
    
    #Averaging year's data by station WITHIN EASTERN SOUND
    working = df.loc[(df["Year"]==year) & (lat_min<=df["Latitude"]) & (df["Latitude"]<=lat_max) & (lon_min<=df["Longitude"]) & (df["Longitude"]<=lon_max)].copy()
    working = working.groupby("Station ID")["Temperature (C)"].mean()
    print(working.head())
    plt.hist(working, bins=20)
    plt.xlim(temp_limits)
    plt.title("Summer Average Data Temperature Distribution, " + str(year), fontsize=24)

    plt.savefig("Graphs/Time_Dependent_Summer_Average_Distribution_Data" + str(year) + ".png")
    plt.show()

# +
#Days over Chosen Threshold(s)
plt.rcParams["figure.figsize"]=(30, 30)

#Plotting from Model
hfig, hax =plt.subplots(len(years), len(thresholds), squeeze=True)

for i, year in enumerate(years):
    for j, thresh in enumerate(thresholds):
        total = np.zeros(1)
        for day in  days:
            error, hmap = model_output(year, stride, models[year], ymeans[year], day)

            #Establishing threshold as the cut
            over = np.where(hmap>thresh, 1, 0)

            #Readding nans for better visualization
            over=np.where(np.isnan(hmap), np.nan, over)

            if len(total)>1:
                total += over
            else:
                total = over
        hax[i, j].hist(total.flatten(), bins=20)
        hax[i, j].set_title("Days over Threshold of " + str(thresh) + " degrees (C) Model, " + str(year), fontsize=18)
        hax[i, j].set_xlim(day_limits)
plt.savefig("Graphs/Days_over_Thresholds_Distributions_Model.png")
plt.show()

#Plotting from Underlying Data
hfig, hax =plt.subplots(len(years), len(thresholds), squeeze=True)

total = np.zeros(1)
for i, year in enumerate(years):
    for j, thresh in enumerate(thresholds):
        #Getting year's data by station
        working = df.loc[(df["Year"]==year) & (lat_min<=df["Latitude"]) & (df["Latitude"]<=lat_max) & (lon_min<=df["Longitude"]) & (df["Longitude"]<=lon_max)].copy()
        working["Threshold"]= working["Temperature (C)"]>thresh
        working["Threshold"]=working["Threshold"].astype(int)
        working = working.groupby("Station ID")["Threshold"].sum()
        print(working.head())
        hax[i, j].hist(working, bins=20)
        hax[i, j].set_title("Days over Threshold of " + str(thresh) + " degrees (C) Data, " + str(year), fontsize=18)
        hax[i, j].set_xlim(day_limits)
plt.savefig("Graphs/Days_over_Thresholds_Distributions_Data.png")
plt.show()

# -

# ## 2019

year=2019

kernels[year], errors[year], hmaps[year] = model_output(year, stride, kernel=kernel, predictors=4, alpha=.25, day=196)


# ## 2020

year=2020

kernels[year], errors[year], hmaps[year] = model_output(discrete_error, cont_error, year, stride, kernel=kernel,  predictors=4, day=196)

# ## 2021

#Params
year=2021

kernels[year], errors[year], hmaps[year] = model_output(discrete_error, cont_error, year, stride, kernel=kernel,  predictors=4, day=196)

# ## 2019 (August)

year=2019

kernels[year], errors[year], models[year] = model_output(discrete_error, cont_error, year, stride, kernel=kernel, predictors=4, day=227)


# ## 2020 (August)

year=2020

kernels[year], errors[year], models[year] = model_output(discrete_error, cont_error, year, stride, kernel=kernel, predictors=4, day=227)


# ## 2021 (August)

year=2021

kernels[year], errors[year], models[year] = model_output(discrete_error, cont_error, year, stride, kernel=kernel, predictors=4, day=227)


# ## Kernels

#(this time of the preferred modes (where 2019 is similar to 2020))
kernels 

#Standardized Reg
kernels

#Time reg with both kernels (July)
kernels

#Time reg with both kernels (August). Thesse should essentially be the same as above
kernels

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


