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
from salem import mercator_grid, Map, open_xr_dataset
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map
from salem import Grid, wgs84
import os
import requests
import xarray as xr
import geopandas

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
lon_min_z = -72.6
lon_max_z = -71.8

lat_min_z= 41.2
lat_max_z=41.4

# +
#Global Params:

#Defining Eastern Sound Window (useful for visualizing cropped distance map)
lon_min=-72.59
lon_max=-71.81
lat_min=40.97
lat_max=41.54

#This should always be set to TRUE for this notebook
es=True

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

# +
#API Key

api_key = config["Gmaps_API_Key"]
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

#For embayment distance but cropped
cropped = config["Prediction_Dashboard_path9"]

#CSV

#Training Data
paths[5] =  "Data/Space_and_time_agg/agg_daily_morning_coastal_features_6_21_2023.csv"

for path in paths.values():
    assert os.path.exists(path), path
# -

# # Google maps background

# +
clat=(lat_max-lat_min)/2
clon=(lon_max-lon_min)/2
zoom = 2

# url = "https://maps.googleapis.com/maps/api/staticmap?"

# r=requests.get(url + "zoom = " + str(clat) + "," + str(clon) + "&zoom = " + 
#                str(zoom) + "&size = 400x400&key =" + api_key + "sensor = False")

fig, ax = plt.subplots()
grid = mercator_grid(center_ll=(10.76, 46.79), extent=(9e5, 4e5))
g = GoogleVisibleMap(x=[lon_min, lon_max], y=[lat_min, lat_max],
                     scale=2, size_x=400, size_y=400, # scale is for more details
                     maptype='satellite')
ggl_img = g.get_vardata()
ax.imshow(ggl_img)
# -

print(ggl_img.shape)

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

#Getting rid of mystic (state boundaries need to be double checked with os to avoid making fisher's island embayments)
array=np.where(array_all!=0, array_embay_dist, np.nan)
array=np.where(np.isnan(array), array_os, array)

#Setting negatives to nan for visualization
array=np.where(array<0, np.nan, array)

#Final Array only for embayments
embay_only=np.where(array_embay_dist>0, array, np.nan)
#plt.imshow(embay_only)
plt.axis("off")

# # Reading in Data and Building in Model

# ## Reading coastal features array

#Using the entire sound (not just eastern sound)
array=embay_only

gt=embay_dist.GetGeoTransform()
gt

print(array)

#Setting negatives to nan
array=np.where(array<0, np.nan, array)
array

#Setting zeroes to nan
array=np.where(array==0, np.nan, array)
array

#Restricting to parts of the sound with lower error
array=np.where(array>.1, np.nan, array)

array.shape

plt.rcParams["figure.figsize"]=(20, 15)
plt.axis("off")
plt.imshow(array)
plt.title("Fig 5: Embayments in Eastern Sound within Range of Model", fontsize=24)
plt.savefig("Figures_for_paper/fig5.png")

# ## Testing google maps api

g =Grid(proj=wgs84, nxny=(5745, 7809), dxdy=(gt[1], gt[5]), x0y0 = (gt[0], gt[3]))
g
stop_lon = gt[0]+gt[1]*7809
stop_lat= gt[3]+gt[5]*5745
print(stop_lon, stop_lat)
print(lon_min, lat_min)

# +
clat=(lat_max-lat_min)/2
clon=(lon_max-lon_min)/2
zoom = 2

# url = "https://maps.googleapis.com/maps/api/staticmap?"

# r=requests.get(url + "zoom = " + str(clat) + "," + str(clon) + "&zoom = " + 
#                str(zoom) + "&size = 400x400&key =" + api_key + "sensor = False")

fig, ax = plt.subplots()
g = GoogleVisibleMap(x=[lon_min_z, lon_max_z], y=[lat_min_z, lat_max_z],
                     scale=2, size_x=400, size_y=400, # scale is for more details
                     maptype='satellite')
ggl_img = g.get_vardata()
#ax.imshow(ggl_img)

sm = Map(g.grid, factor=1, countries=False)
sm.set_rgb(ggl_img)

data = xr.DataArray(array, dims=['lat', 'lon'], coords= {'lon': np.linspace(gt[0], stop_lon, 7809), 'lat': np.linspace(gt[3], stop_lat, 5745)})
sm.set_data(data)


sm.visualize(ax=ax)

plt.show()
#ax.imshow()

# +
#Try 2
plt.rcParams["figure.figsize"]=(30, 30)
fig, ax = plt.subplots()
g = GoogleVisibleMap(x=[lon_min_z, lon_max_z], y=[lat_min_z, lat_max_z],
                     scale=2, size_x=325, size_y=125, # scale is for more details
                     maptype='satellite')
ggl_img = g.get_vardata()
ggl_img[:, :, 3]=.75
#ax.imshow(ggl_img)

sm = Map(g.grid, factor=1, countries=False)
sm.set_rgb(ggl_img)

hmap = Map(g.grid, factor=1, countries=False, cmap="plasma")
hmap.set_data(data)
hmap.visualize()

# data = xr.DataArray(array, dims=['lat', 'lon'], coords= {'lon': np.linspace(gt[0], stop_lon, 7809), 'lat': np.linspace(gt[3], stop_lat, 5745)})
# sm.set_data(data)=''

hmap.set_text(-71.968, 41.3425, "Mystic Harbor  ", fontsize=12, horizontalalignment="right")
hmap.set_points(-71.968, 41.3425)

hmap.set_text(-72.162, 41.309, "Jordan Cove\n", fontsize=12, horizontalalignment="left", verticalalignment="bottom")
hmap.set_points(-72.162, 41.309)

hmap.set_text(-72.145, 41.303, " White Point", fontsize=12)
hmap.set_points(-72.145, 41.303)

hmap.set_text(-72.181, 41.339, "  Niantic River", fontsize=12)
hmap.set_points(-72.181, 41.339)

hmap.set_text(-71.851, 41.326, "  Pawcatuck River", fontsize=12)
hmap.set_points(-71.851, 41.326)

sm.visualize(ax=ax)
hmap.visualize(ax=ax)
plt.show()
# -

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

#Optional Restriction of TRAINING params to Eastern Sound
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
    x_mean = np.mean(X_train, axis=0)
    x_std = np.std(X_train, axis=0)
    X_train=X_train - x_mean
    X_train=X_train / x_std
    
    #Constructing Process
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=alpha, optimizer=None)


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
display_days={"July 15th": 196, "August 15th": 227}

#Dicts for storage of models
kernels_liss = {2019: 1.06**2 * RBF(length_scale=[0.0334, 0.135, 2.29e+03, 0.185], length_scale_bounds="fixed") + 2.85**2 * RationalQuadratic(alpha=0.266, length_scale=1.85, alpha_bounds="fixed", length_scale_bounds="fixed"),
2020:0.939**2 * RBF(length_scale=[0.0683, 0.106, 7.87, 0.14], length_scale_bounds="fixed") + 5.53**2 * RationalQuadratic(alpha=0.126, length_scale=4.06, alpha_bounds="fixed", length_scale_bounds="fixed"),
2021:1.09**2 * RBF(length_scale=[0.213, 0.159, 4.78, 0.122], length_scale_bounds="fixed") + 1.51**2 * RationalQuadratic(alpha=0.412, length_scale=1.26, alpha_bounds="fixed", length_scale_bounds="fixed")
}

kernels_es = {2019: 0.645**2 * RBF(length_scale=[3.17, 3.04e+03, 6.03, 0.0479]) + 7.4**2 * RationalQuadratic(alpha=0.0217, length_scale=1.29),
2020: 1.3**2 * RBF(length_scale=[4.42, 1e+05, 22.9, 0.054]) + 3.45**2 * RationalQuadratic(alpha=0.0324, length_scale=0.451),
2021: 1.46**2 * RBF(length_scale=[5.52, 4.97, 3.84, 0.052]) + 4.03**2 * RationalQuadratic(alpha=0.0231, length_scale=0.538)    
}

#Dicts for storage of models and model properties
kernels = {}
models = {}
ymeans = {}
# -

#Loading Kernels
#Eastern Sound model
if es:
    for year in years:
        #Getting models and ymeans outputted
        models[year], ymeans[year] = build_model(predictors, year, kernels_es[year], alpha)
        #print(models[year].kernel_)
else:
#Entire LISS model
    for year in years:
        #Getting models and ymeans outputted
        models[year], ymeans[year] = build_model(predictors, year, kernels_liss[year], alpha)
        print(models[year].kernel_)


# ## Heatmaps

#Graphing Params
plt.rcParams["figure.figsize"]=(30, 15)

from salem import DataLevels
dl=DataLevels()

# +
#Graphing selected days above

#Making Figures
hfig, hax =plt.subplots(len(years), len(display_days), layout="tight")
efig, eax =plt.subplots(len(years), len(display_days), layout="tight")

#Visualization Params
bottom=.15
top=.75

#Getting same basemap for all figs
fig, ax = plt.subplots()
g = GoogleVisibleMap(x=[lon_min_z, lon_max_z], y=[lat_min_z, lat_max_z],
                     scale=2, size_x=325, size_y=125, # scale is for more details
                     maptype='satellite')
ggl_img = g.get_vardata()
ggl_img[:, :, 3]=.5

#Plotting
for i, year in enumerate(years):
    for j, day in enumerate(display_days.values()):     
                error, means = model_output(year, stride, models[year], ymeans[year], day)
                
                #Predictions
                
                data = xr.DataArray(means, dims=['lat', 'lon'], coords= {'lon': np.linspace(gt[0], stop_lon, 7809)[::stride], 'lat': np.linspace(gt[3], stop_lat, 5745)[::stride]})
                sm.set_data(data)

                sm = Map(g.grid, factor=1, countries=False)
                sm.set_rgb(ggl_img)

                hmap = Map(g.grid, factor=1, countries=False, cmap="plasma", vmin=19, vmax=27, extend="neither")
                hmap.set_data(data)
 
                hmap.set_text(-71.968, 41.3425, "Mystic Harbor  ", fontsize=12, horizontalalignment="right")
                hmap.set_points(-71.968, 41.3425)

                hmap.set_text(-72.162, 41.309, "Jordan Cove\n", fontsize=12, horizontalalignment="left", verticalalignment="bottom")
                hmap.set_points(-72.162, 41.309)

                hmap.set_text(-72.145, 41.303, " White Point", fontsize=12)
                hmap.set_points(-72.145, 41.303)

                hmap.set_text(-72.181, 41.339, "  Niantic River", fontsize=12)
                hmap.set_points(-72.181, 41.339)

                hmap.set_text(-71.851, 41.326, "  Pawcatuck River", fontsize=12)
                hmap.set_points(-71.851, 41.326)
                
                sm.visualize(ax=hax[i, j])
                hmap.visualize(ax=hax[i, j])
                
                letters= ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)", "j)"]
                number=2*i+j
                hax[i, j].set_title(letters[number] + " " + list(display_days.keys())[j] + ", " + str(year), fontsize=24, loc="left")

                #Error
                
                data = xr.DataArray(error, dims=['lat', 'lon'], coords= {'lon': np.linspace(gt[0], stop_lon, 7809)[::stride], 'lat': np.linspace(gt[3], stop_lat, 5745)[::stride]})
                sm.set_data(data)

                sm = Map(g.grid, factor=1, countries=False)
                sm.set_rgb(ggl_img)
        
                hmap = Map(g.grid, factor=1, countries=False, cmap="plasma", vmin=0, vmax=2.5, extend="neither")
                hmap.set_data(data)

                hmap.set_text(-71.968, 41.3425, "Mystic Harbor  ", fontsize=12, horizontalalignment="right")
                hmap.set_points(-71.968, 41.3425)

                hmap.set_text(-72.162, 41.309, "Jordan Cove\n", fontsize=12, horizontalalignment="left", verticalalignment="bottom")
                hmap.set_points(-72.162, 41.309)

                hmap.set_text(-72.145, 41.303, " White Point", fontsize=12)
                hmap.set_points(-72.145, 41.303)

                hmap.set_text(-72.181, 41.339, "  Niantic River", fontsize=12)
                hmap.set_points(-72.181, 41.339)

                hmap.set_text(-71.851, 41.326, "  Pawcatuck River", fontsize=12)
                hmap.set_points(-71.851, 41.326)
                
                sm.visualize(ax=eax[i, j])
                hmap.visualize(ax=eax[i, j])                
                                
                eax[i, j].set_title(list(display_days.keys())[j] + ", " + str(year), fontsize=18)

#Axes off
[ax.set_axis_off() for ax in hax.ravel()]
[ax.set_axis_off() for ax in eax.ravel()]

# hfig.tight_layout(pad=15)
# efig.tight_layout(pad=15)

hfig.savefig("Graphs/June_Heatmaps/ES_Heatmaps.png", bbox_inches='tight')
hfig.savefig("Figures_for_paper/fig7a.png", bbox_inches='tight')

efig.savefig("Graphs/June_Heatmaps/ES_Dependent_Errors.png", bbox_inches='tight')
efig.savefig("Figures_for_paper/fig7b.png")

plt.show()

# +
#Blow-up of one date

plt.rcParams["figure.figsize"]=(30, 30)

#Data params
year = 2021
day = 227

#Visualization Params
bottom=.15
top=.75

error, means = model_output(year, stride, models[year], ymeans[year], day)

fig, ax = plt.subplots()
g = GoogleVisibleMap(x=[lon_min_z, lon_max_z], y=[lat_min_z, lat_max_z],
                     scale=2, size_x=325, size_y=125, # scale is for more details
                     maptype='satellite')
ggl_img = g.get_vardata()
ggl_img[:, :, 3]=.75

#ax.imshow(ggl_img)

data = xr.DataArray(means, dims=['lat', 'lon'], coords= {'lon': np.linspace(gt[0], stop_lon, 7809)[::stride], 'lat': np.linspace(gt[3], stop_lat, 5745)[::stride]})
sm.set_data(data)

sm = Map(g.grid, factor=1, countries=False)
sm.set_rgb(ggl_img)

hmap = Map(g.grid, factor=1, countries=False, cmap="plasma", vmin=19, vmax=27)
hmap.set_data(data)

hmap.set_text(-71.968, 41.3425, "Mystic Harbor  ", fontsize=18, horizontalalignment="right")
hmap.set_points(-71.968, 41.3425)

hmap.set_text(-72.162, 41.309, "Jordan Cove\n", fontsize=18, horizontalalignment="left", verticalalignment="bottom")
hmap.set_points(-72.162, 41.309)

hmap.set_text(-72.145, 41.303, " White Point", fontsize=18)
hmap.set_points(-72.145, 41.303)

hmap.set_text(-72.181, 41.339, "  Niantic River", fontsize=18)
hmap.set_points(-72.181, 41.339)

hmap.set_text(-71.851, 41.326, "  Pawcatuck River", fontsize=18)
hmap.set_points(-71.851, 41.326)

sm.visualize(ax=ax)
hmap.visualize(ax=ax, addcbar=False)

ax.tick_params(labelsize="xx-large")
#ax.set_title(list(display_days.keys())[j] + ", " + str(year), fontsize=18)

cbar=fig.colorbar(plt.cm.ScalarMappable(norm= mpl.colors.Normalize(vmin=19, vmax=27), cmap="plasma"), ax=ax, fraction = .0171)
cbar.ax.tick_params(labelsize="xx-large")

plt.show()
# -

#Params for graphs of thresholds and summer averages
days=range(152, 275)
avg_hmaps= {}
thresh_hmaps={}

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
    total = total/len(days)
    avg_hmaps[year]=total
    avg = plt.imshow(total)
    plt.title("Summer Average Temperature, " + str(year), fontsize=24)
    cbar = fig.colorbar(avg, location="right", shrink=.75)
    cbar.set_label('Deg C', rotation=270)
    cbar.ax.tick_params(labelsize=24)
    plt.axis("off")
    plt.savefig("Graphs/June_Heatmaps/ES_Summer_Average_" + str(year) + ".png")
    #plt.show()
    
# -


# # Updating EHSI

# ## Old Temperature Component

#Make sure this stride matches the above stride
stride=5

print(array.shape)

# +
import numpy as np
import pandas as pd
import os
import json
from json import dumps
import matplotlib.pyplot as plt

#Additional imports for distance calculations
from sklearn.neighbors import DistanceMetric
from math import radians

#defining distance metric of choice
dist = DistanceMetric.get_metric('haversine')

#Reading in CT DEEP averages with fielpath for this program to be run elsewhere
assert(os.path.exists('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\'))

ct_means=pd.read_csv('C:\\Users\\blawton\\Data Visualization and Analytics Challenge\\Data\\ct_deep_temp_2009_2011.csv', index_col=0)
                                 
#Creating col for coords in radians and setting index
ct_means["coords"]=ct_means.apply(lambda x: [radians(x["LATITUDE"]), radians(x["LONGITUDE"])], axis=1)
ct_means

#Defining function of interest
def ct_deep_idw(row, power):
    coord=row.loc["coords"]
    coord=[radians(deg) for deg in coord]    
    distances=ct_means["coords"].apply(lambda x: (dist.pairwise([x, coord]))[0,1])
    #print(distances)
    
    #Accounting for divide by zero by taking mean of first zero value
    if min(distances)>0:
        weights=1/distances
        weights=np.power(weights, power)
        weights=weights/weights.sum()
    else:
        return("Error: Divde by Zero")
    
    weighted=np.dot(weights, ct_means["temperature"])
    return(weighted)



# -

#Make sure that the inverse distance weighter is loaded, this now requires a year variable"Year" and a day variable "Day" in each row
def interpolate_means(data):
    output=data.copy(deep=True)
    output["coords"]=output.apply(lambda x: [x["Latitude"], x["Longitude"]], axis=1)
    output["interpolated"]=output.apply(ct_deep_idw, axis=1, args=[1])
    
    #Unstacking is unneccesary because STS stations have diff coords each year
    #output=output.unstack(level=-1)
    
    #Unnecessary to convert because datatype of year should be numeric for input
    #output.columns=[int(year) for year in output.columns]
    
    return(output)


## Getting IDW CTDEEP TEMP for 2009-2011
X_pred, resampled, grid_shape = get_heatmap_inputs(stride, array, gt, 0)
avg_temp=pd.DataFrame(X_pred, columns = indep_var)
working=avg_temp.loc[~avg_temp["embay_dist"].isna()]
interped = interpolate_means(working)
avg_temp["interpolated"]=np.nan
avg_temp.loc[~avg_temp["embay_dist"].isna()]=interped
print(avg_temp["embay_dist"])

#Aggregating CTDEEP idw temperatures
avg_temp=avg_temp["interpolated"].values.reshape(grid_shape)

bins=[0, 21, 21.4, 21.9, 22.3, 22.8, 23.2, 23.7, 24.1, 24.6, 25, 100]

# +
#Reclassifying
reclass1=np.digitize(avg_temp, bins, right=False)
reclass1=10-(reclass1-1)
reclass1=np.where(~np.isnan(avg_temp), reclass1, np.nan)

# display = plt.imshow(reclass1, vmin=0, vmax=10)
# cbar = plt.colorbar(display, location="right", shrink=.75)
# cbar.set_label('Temperature Score (Open Sound Model)', rotation=270, fontsize=18, labelpad=30)
# cbar.ax.tick_params(labelsize=18)
# plt.axis("off")
# plt.tight_layout()
# -

# ## New Temperature Component

years=[2019, 2020, 2021]

# +
#Averaging summer average heatmaps from 2019 through 2021
avg_temp2= np.zeros(avg_hmaps[years[0]].shape)
for key in avg_hmaps.keys():
    print(key)
    hmap=avg_hmaps[key]
    avg_temp2+=avg_hmaps[key]
avg_temp2=avg_temp2/len(avg_hmaps.keys())

#Mapping (redundant)

# display = plt.imshow(avg_temp2)
# cbar = plt.colorbar(display, location="right", shrink=.75)
# cbar.set_label('Temperature', rotation=270, fontsize=18, labelpad=30)
# cbar.ax.tick_params(labelsize=18)
# plt.title("GP-Predicted Temperature in LIS Embayments (2019 through 2021)", fontsize=24)
# plt.axis("off")
# plt.tight_layout()

# +
#Side by side CTDEEP w2019-2021 avg with GP 2019-2021 avg
plt.rcParams["figure.figsize"]=(30, 25)

#CTDEEP

fig, ax = plt.subplots(2, layout="constrained")
g = GoogleVisibleMap(x=[lon_min_z, lon_max_z], y=[lat_min_z, lat_max_z],
                     scale=2, size_x=325, size_y=125, # scale is for more details
                     maptype='satellite')
ggl_img = g.get_vardata()
ggl_img[:, :, 3]=.5
#dl = DataLevels(vmin=19.4, vmax=20.8)

#ax.imshow(ggl_img)

data = xr.DataArray(avg_temp, dims=['lat', 'lon'], coords= {'lon': np.linspace(gt[0], stop_lon, 7809)[::stride], 'lat': np.linspace(gt[3], stop_lat, 5745)[::stride]})
sm.set_data(data)

sm = Map(g.grid, factor=1, countries=False)
sm.set_rgb(ggl_img)

hmap = Map(g.grid, factor=1, countries=False, cmap="plasma", vmin=19, vmax=27)
hmap.set_data(data)

ax[0].tick_params(labelsize=36)

hmap.set_text(-71.968, 41.3425, "Mystic Harbor  ", fontsize=18, horizontalalignment="right")
hmap.set_points(-71.968, 41.3425)

hmap.set_text(-72.162, 41.309, "Jordan Cove\n", fontsize=18, horizontalalignment="left", verticalalignment="bottom")
hmap.set_points(-72.162, 41.309)

hmap.set_text(-72.145, 41.303, " White Point", fontsize=18)
hmap.set_points(-72.145, 41.303)

hmap.set_text(-72.181, 41.339, "  Niantic River", fontsize=18)
hmap.set_points(-72.181, 41.339)

hmap.set_text(-71.851, 41.326, "  Pawcatuck River", fontsize=18)
hmap.set_points(-71.851, 41.326)

sm.visualize(ax=ax[0])
hmap.visualize(addcbar=False, ax=ax[0])

#GP

g = GoogleVisibleMap(x=[lon_min_z, lon_max_z], y=[lat_min_z, lat_max_z],
                     scale=2, size_x=325, size_y=125, # scale is for more details
                     maptype='satellite')
ggl_img = g.get_vardata()
ggl_img[:, :, 3]=.75

#ax.imshow(ggl_img)

data = xr.DataArray(avg_temp2, dims=['lat', 'lon'], coords= {'lon': np.linspace(gt[0], stop_lon, 7809)[::stride], 'lat': np.linspace(gt[3], stop_lat, 5745)[::stride]})
sm.set_data(data)

sm = Map(g.grid, factor=1, countries=False)
sm.set_rgb(ggl_img)

hmap = Map(g.grid, factor=1, countries=False, cmap="plasma", vmin=19, vmax=27)
hmap.set_data(data)

hmap.set_text(-71.968, 41.3425, "Mystic Harbor  ", fontsize=18, horizontalalignment="right")
hmap.set_points(-71.968, 41.3425)

hmap.set_text(-72.162, 41.309, "Jordan Cove\n", fontsize=18, horizontalalignment="left", verticalalignment="bottom")
hmap.set_points(-72.162, 41.309)

hmap.set_text(-72.145, 41.303, " White Point", fontsize=18)
hmap.set_points(-72.145, 41.303)

hmap.set_text(-72.181, 41.339, "  Niantic River", fontsize=18)
hmap.set_points(-72.181, 41.339)

hmap.set_text(-71.851, 41.326, "  Pawcatuck River", fontsize=18)
hmap.set_points(-71.851, 41.326)

sm.visualize(ax=ax[1])
hmap.visualize(ax=ax[1], addcbar=False)

ax[1].tick_params(labelsize=36)
#ax.set_title(list(display_days.keys())[j] + ", " + str(year), fontsize=18)

cax = fig.add_axes([0.1, -.1, 0.8, 0.05])
cbar=fig.colorbar(plt.cm.ScalarMappable(norm= mpl.colors.Normalize(vmin=19, vmax=27), cmap="plasma"), cax=cax, orientation="horizontal")
cbar.ax.tick_params(labelsize=36)

ax[0].set_title("a) 2009-2011 Avg Temp (IDW)", fontsize=36, loc="left")
ax[1].set_title("b) 2019-2021 Avg Temp (GP)", fontsize=36, loc="left")

plt.show()
# -

bins=[0, 21, 21.4, 21.9, 22.3, 22.8, 23.2, 23.7, 24.1, 24.6, 25, 100]

#Reclassifying new temperature heatmap
reclass2=np.digitize(avg_temp2, bins, right=False)
reclass2=10-(reclass2-1)
reclass2=np.where(~np.isnan(avg_temp2), reclass2, np.nan)
display=plt.imshow(reclass2)
cbar = plt.colorbar(display, location="right", shrink=.75)
cbar.set_label('Temperature Score', rotation=270, fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)
plt.axis("off")
plt.title("Gaussian Processs ")
plt.tight_layout()

# ## EHSI Difference

# ### Raw Difference

plt.rcParams["figure.figsize"]=(30, 20)

height=hmap.shape[0]
hmap=avg_temp-avg_temp2
display=plt.imshow(hmap[int(.15*height):int(.6*height), :])
cbar = plt.colorbar(display, location="bottom", shrink=.75)
cbar.set_label('Increase in Temp with Embayment Data + GP (Degrees (C))', fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)
plt.axis("off")
plt.title("Figure 8: GP Temperature Model and CTDEEP IDW comparison", fontsize=24)
plt.savefig(paths[7] + "EHSI_open_sound_vs_embayment_bias_graph_raw.png", )
#plt.show()

# ### Reclassified Difference

plt.rcParams["figure.figsize"]=(30, 30)

# +
hmap=2*(reclass2-reclass1)

vmin=19.4
vmax=20.8

fig, ax = plt.subplots()
g = GoogleVisibleMap(x=[lon_min_z, lon_max_z], y=[lat_min_z, lat_max_z],
                     scale=2, size_x=325, size_y=125, # scale is for more details
                     maptype='satellite')
ggl_img = g.get_vardata()
ggl_img[:, :, 3]=.5
#dl = DataLevels(vmin=19.4, vmax=20.8)

#ax.imshow(ggl_img)

data = xr.DataArray(hmap, dims=['lat', 'lon'], coords= {'lon': np.linspace(gt[0], stop_lon, 7809)[::stride], 'lat': np.linspace(gt[3], stop_lat, 5745)[::stride]})
sm.set_data(data)

sm = Map(g.grid, factor=1, countries=False)
sm.set_rgb(ggl_img)

hmap = Map(g.grid, factor=1, countries=False, cmap="plasma", vmin=-20, vmax=0)
hmap.set_data(data)

ax.tick_params(labelsize="xx-large")

hmap.set_text(-71.968, 41.3425, "Mystic Harbor  ", fontsize=18, horizontalalignment="right")
hmap.set_points(-71.968, 41.3425)

hmap.set_text(-72.162, 41.309, "Jordan Cove\n", fontsize=18, horizontalalignment="left", verticalalignment="bottom")
hmap.set_points(-72.162, 41.309)

hmap.set_text(-72.145, 41.303, " White Point", fontsize=18)
hmap.set_points(-72.145, 41.303)

hmap.set_text(-72.181, 41.339, "  Niantic River", fontsize=18)
hmap.set_points(-72.181, 41.339)

hmap.set_text(-71.851, 41.326, "  Pawcatuck River", fontsize=18)
hmap.set_points(-71.851, 41.326)

sm.visualize()
hmap.visualize(addcbar=False)


cbar=fig.colorbar(plt.cm.ScalarMappable(norm= mpl.colors.Normalize(vmin=-20, vmax=0), cmap="plasma"), ax=ax, orientation="vertical", fraction=.0171)
cbar.set_label('New Model Increase (+)/Decrease (-) in EHSI Temp. Component', fontsize=18, labelpad=30)
cbar.ax.tick_params(labelsize=18)

#plt.axis("off")
# plt.title("Figure 8: GP Temperature Model and CTDEEP IDW comparison", fontsize=24)
# plt.savefig(paths[7] + "EHSI_open_sound_vs_embayment_bias_graph.png", )
# plt.savefig("Figures_for_paper/fig8.png", bbox_inches='tight')
# plt.show()
# -


