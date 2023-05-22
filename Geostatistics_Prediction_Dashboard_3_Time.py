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

#Defining Eastern Sound Window
lon_min=-72.592354875000
lon_max=-71.811481513000

lat_min=40.970592192000
lat_max=41.545060950000

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

#Final embayment distance (output of first paragraph of notebook)
paths[4]  = config["Prediction_Dashboard_path4"]

#Final embayment distance (output of first paragraph of notebook) BUT within measured embays
paths[6] = config["Prediction_Dashboard_path6"]

#Ouput Path for tiffs
paths[7] = config["Prediction_Dashboard_output"]

#CSV

#Training Data
paths[5] = "Data/Space_and_time_agg/agg_daily_morning_coastal_features_5_17_2023.csv"

for path in paths.values():
    assert os.path.exists(path), path
# -

# # Mainpulating Coastal Distance

# Already done in spatial model

# # Reading in Data and Building in Model

# ## Reading array

#Using the entire sound still to test for extrapolation ability
embay_dist = gdal.Open(paths[4])

gt=embay_dist.GetGeoTransform()
gt

proj=embay_dist.GetProjection()
proj

band=embay_dist.GetRasterBand(1)
array=band.ReadAsArray()

pixelwidth=gt[1]
pixelheight=-gt[5]
pixelheight

xOrigin = gt[0]
yOrigin = gt[3]

cols = embay_dist.RasterXSize
rows = embay_dist.RasterYSize
cols

array

#Setting negatives to nan
array=np.where(array<0, np.nan, array)
array

# ## Reading csv and building model

#This is the source of the model's predictions
df=pd.read_csv(paths[5], index_col=0)
df.head()

#Checking Dominion Stations to Make Sure they are in Vaudrey Embayment (C and NB)
df.loc[df["Organization"]=="Dominion"]

#Dropping nas
print(len(df))
df.dropna(subset=["Station ID", "Day", "embay_dist", "Longitude", "Latitude", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

# +
# #Optional Restriction of TRAINING params to Eastern Sound
# df=df.loc[(df["Longitude"]>lon_min) & (df["Longitude"]<lon_max)].copy()
# print(len(df))
# -

# ^The above restriction destroys the model's ability to find meaningful hyperparameters, which shows that training on the LIS as a whole is clearly preferable

pd.unique(df["Organization"])

# +
#Model Params
station_var=["Station ID"]

ind_var=["Longitude", "Latitude", "embay_dist", "Day"]

dep_var = ["Temperature (C)"]

predictors=len(ind_var)


# -

#Building Model (adapted to dashboard from "Geostatistics_Workbook_version_2.1.2.ipynb")
def build_model(predictors, year, kernel, noise, alpha):
    
    #Selecting data from proper year
    period=df.loc[df["Year"]==year, station_var+ind_var+dep_var]
    data=period.values
    
    #Partitioning X and y
    
    X_train, y_train = data[:, 1:predictors+1], data[:, -1]
    print(X_train.shape)
    
    #Ensuring proper dtypes
    X_train=np.array(X_train, dtype=np.float64)
    y_train=np.array(y_train, dtype=np.float64)

    #Demeaning y
    y_mean=np.mean(y_train)
    y_train=y_train-y_mean

    #Normalizing predictors (X) for testing sets
    #X_train=X_train - np.mean(X_train, axis=0)
    #X_train=X_train / np.std(X_train, axis=0)

    #Constructing Process
    if noise:
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=alpha)
    else:
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

    #Training Process
    gaussian_process.fit(X_train, y_train)

    return(gaussian_process, y_mean)


# # Setting Up Output Heatmap

#Defining Resolution of Output Heatmap
stride=5


def get_heatmap_inputs(stride, coastal_map, gt):
    
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
    
    #Calculating lattitudes of Resampled Array
    
    lats= -np.arange(resampled.shape[0])*(pixelheight)*(stride) + yOrigin
    #Centering
    lats +=pixelheight/2
    
    #Making latitude and longitude into a grid
    lonv, latv = np.meshgrid(lons, lats)
    
    #Flattening predictors to run through gaussian_process.predict
    X_pred = np.column_stack((lonv.flatten(), latv.flatten(), resampled.flatten()))
    
    return(X_pred, resampled, resampled.shape)


# # Model Output

# Initial guesses for model length scales are important because they prevent the model from going into an unstable mode where the middle 2nd length scale parameter blows up. Edit: this still happens by chance even with an initial guess of .1

#Coloring scales
hmin=20
hmax=26
sdmin=0
sdmax=1.2


def model_output(day, alpha, year, stride, kernel, predictors):
    
    #Building Process
    kernel = kernel
    
    process, mean =build_model(predictors, year, kernel, True, alpha)
    
    #Building Heatmap
    X_pred, resampled, grid_shape = get_heatmap_inputs(stride, array, gt)
    
    #Narrowing to where embay_dist is defined
    X_pred=pd.DataFrame(X_pred)
    X_pred["Day"]=day
    
    #Xpred columns must match order of indep vars
    print(X_pred.head)
    
    y_pred=pd.DataFrame(index=X_pred.index)

    #Filtering to prediction region
    X_pred_filt= X_pred.dropna(axis=0)
    
    #Running process
    y_pred_filt, MSE = process.predict(X_pred_filt.values, return_std=True)
    y_pred_filt+=mean
    
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
    
    #Graphing and Saving Heatmap
    fig, ax = plt.subplots()
    heatmap = plt.imshow(y_grid, vmin=hmin, vmax=hmax)
    
    title=str(year) + " Embayment Temperature Heatmap"
    plt.title(title, size="x-large")
    plt.xticks(ticks=np.linspace(0, grid_shape[1], 9), labels=np.linspace(lon_min, lon_max, 9))
    plt.xlabel("Longitude", size="x-large")

    plt.yticks(ticks=np.linspace(0, grid_shape[0], 5), labels=np.linspace(lat_min, lat_max, 5))
    plt.ylabel("Latitude", size="x-large")

    cbar = fig.colorbar(heatmap, location="right", shrink=.75)
    cbar.set_label('Deg C', rotation=270, size="large")

    #plt.savefig("Data/Heatmaps/" + title.replace(" ", "_") + "_v3.png")
    plt.show()
    
    #Graphing and Saving STD
    fig, ax = plt.subplots()
    heatmap = plt.imshow(std_grid, vmin=sdmin, vmax=sdmax)
    
    title=str(year) + " Model Error"
    plt.title(title, size="x-large")
    plt.xticks(ticks=np.linspace(0, grid_shape[1], 9), labels=np.linspace(lon_min, lon_max, 9))
    plt.xlabel("Longitude", size="x-large")

    plt.yticks(ticks=np.linspace(0, grid_shape[0], 5), labels=np.linspace(lat_min, lat_max, 5))
    plt.ylabel("Latitude", size="x-large")

    cbar=fig.colorbar(heatmap, location="right", shrink=.75)
    cbar.set_label('Standard Deviation in Deg C', rotation=270, size="large")

    #plt.savefig("Data/Heatmaps/" + title.replace(" ", "_") + "_v3.png")
    plt.show()
    
    return(process.kernel_, std_grid, y_grid)


#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

# +
#Params
years=[2019, 2020, 2021]
predictors=4
alpha=.25
stride=5
lsb=(1e-5, 1e5)

#Day for prediction
day=196

#Dicts for storage of models
kernels = {}
errors= {}
models = {}
x_stds = {}
# -

#Getting x_stds for initial kernel guess
for year in years:
    period=df.loc[df["Year"]==year, station_var+ind_var+dep_var]
    data=period.values

    #Partitioning X and y

    X_train, y_train = data[:, 1:predictors+1], data[:, -1]
    print(X_train.shape)

    #Ensuring proper dtypes
    X_train=np.array(X_train, dtype=np.float64)
    y_train=np.array(y_train, dtype=np.float64)

    #Demeaning y
    y_mean=np.mean(y_train)
    y_train=y_train-y_mean

    #Normalizing predictors (X) for testing sets
    X_train=X_train - np.mean(X_train, axis=0)
    x_stds[year]=np.std(X_train, axis=0)

#Testing output
x_stds[2019]

# ## 2019

year=2019

kernel = 1.52**2 * RBF(length_scale=[0.884, 0.06, 0.0487, 0.00001], length_scale_bounds="fixed") + Matern(length_scale=[1]*4, length_scale_bounds=lsb)
kernels[year], errors[year], models[year] = model_output(day, alpha, year, stride, kernel, predictors) 

kernels[2019]

# ## 2020

year=2020

kernel = 1 * RBF(x_stds[year], length_scale_bounds=lsb)
kernels[year], errors[year], models[year] = model_output(day, alpha, year, stride, kernel, predictors) 

# ## 2021

year=2021

kernel = 1 * RBF(x_stds[year], length_scale_bounds=lsb)
kernels[year], errors[year], models[year] = model_output(day, alpha, year, stride, kernel, predictors) 

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


