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
paths[5] = "Data/Space_agg/agg_summer_means_coastal_features_4_21_2021.csv"

for path in paths.values():
    assert os.path.exists(path), path
# -

# # Mainpulating Coastal Distance (Only Do Once)

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

#Setting open sound (non vaudrey embayments) to 0
array=np.where(array_os>=0, 0, array_embay_dist)

#Getting rid of mystic (state boundaries need to be double checked with os to avoid making fisher's island embayments)
array=np.where(array_all!=0, array, np.nan)
array=np.where(np.isnan(array), array_os, array)

#Setting negatives to nan for visualization
array=np.where(array<0, np.nan, array)

#Final Array only for embayments
embay_only=np.where(array_embay_dist>0, array, np.nan)
plt.imshow(embay_only)
plt.axis("off")

#Outputting Final Embayment Distance to avoid repeating above
driver = gdal.GetDriverByName("GTiff")
driver.Register()
outds = driver.Create(paths[6], xsize=cols, ysize=rows, bands=1, eType=gdal.GDT_Int16)
outds.SetGeoTransform(gt)
outds.SetProjection(proj)
outband = outds.GetRasterBand(1)
outband.WriteArray(embay_only)
outband.SetNoDataValue(np.nan)
outband.FlushCache()
outband=None
outds=None

# # Reading in Data and Building in Model

# ## Reading array

#Using the updated file without points removed
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

#Testing alpha logic
cont_error=.25
discrete_error=1.44
orgs=df.loc[df["Year"]==2019, "Organization"].values
np.where(np.isin(orgs, cont_orgs), cont_error, discrete_error)


#Building Model (adapted to dashboard from "Geostatistics_Workbook_version_2.1.2.ipynb")
def build_model(predictors, year, kernel, noise, discrete_error, cont_error):
    
    #Selecting data from proper year
    period=df.loc[df["Year"]==year, ["Station ID", "Longitude", "Latitude", "embay_dist", "Temperature (C)"]]
    data=period.values

    #Designing an alpha matrix based on which means are from discrete data
    orgs=df.loc[df["Year"]==year, "Organization"].values

    #applying mask
    alpha=np.where(np.isin(orgs, cont_orgs), cont_error, discrete_error)
    
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
    #X_test[i]=X_test[i] - np.mean(X_test[i], axis=0)
    #X_test[i]=X_test[i] / np.std(X_test[i], axis=0)

    #Normalizing predictors (X) for testing sets
    #X_train[i]=X_train[i] - np.mean(X_train[i], axis=0)
    #X_train[i]=X_train[i] / np.std(X_train[i], axis=0)

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
stride=2


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


def model_output(discrete_error, cont_error, year, stride, kernel):
    
    #Building Process
    kernel = kernel
    
    process, mean =build_model(3, year, kernel, True, discrete_error, cont_error)
    
    #Building Heatmap
    X_pred, resampled, grid_shape = get_heatmap_inputs(stride, array, gt)
    
    #Narrowing to where embay_dist is defined
    X_pred=pd.DataFrame(X_pred)
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

    plt.savefig("Data/Heatmaps/" + title.replace(" ", "_") + "_v3.png")
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

    plt.savefig("Data/Heatmaps/" + title.replace(" ", "_") + "_v3.png")
    plt.show()
    
    return(process.kernel_, std_grid, y_grid)


#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

# +
#Params
parameters=3
discrete_error=1.44
cont_error=.25
stride=2
lsb=(1e-5, 1e5)
kernel = 1 * RBF([1]*parameters, length_scale_bounds=lsb)

#Dicts for storage of models
kernels = {}
errors= {}
models = {}
# -

# ## 2019

year=2019

kernels[year], errors[year], models[year] = model_output(discrete_error, cont_error, year, stride, kernel=kernel) 

# ## 2020

year=2020

kernels[year], errors[year], models[year] = model_output(discrete_error, cont_error, year, stride, kernel=kernel) 

# ## 2021

#Params
year=2021

kernels[year], errors[year], models[year] = model_output(discrete_error, cont_error, year, stride, kernel=kernel) 

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


