import pandas as pd
import numpy as np
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.interpolate import NearestNDInterpolator
import matplotlib as mpl

from osgeo import gdal, gdalconst
import numpy as np
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# __Readme__

# __v_2__ uses Geostatistics_Prediction_Dashboard_2.1.2 and does it for the entire sound

# +
#Loading paths config.yml
import yaml

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# +
#Setting path variables for coastal features
paths={}

#Embayment Distance for Eastern Sound (cropped and resampled)
paths[1] = config["EHSI_Update_path1"]

#Original EHSI file, uncropped, un-resampled
paths[2] = config["EHSI_Update_path2"]

#EHSI file cropped but un-resampled
paths[3] = config["EHSI_Update_path3"]

#Final EHSI file cropped and resampled
paths[4] = config["EHSI_Update_path4"]

#Input data for version of the model used (currently 2.1.2)
paths[5] = config["EHSI_Update_path5"]

#Output
output_path = config["EHSI_Update_output"]

for path in paths.values():
    assert(os.path.exists(path))
    
assert(os.path.exists(output_path))
# -

#Global Model Params
stride=2
discrete_error=1.44
cont_error=.1

# +
#Defining Eastern Sound Window

lon_min=-72.592354875000
lon_max=-71.811481513000

lat_min=40.970592192000
lat_max=41.545060950000
# -

avg_scales = np.array([0.639, 0.0594, 0.0382]) + np.array([0.835, 0.052, 0.0432])
avg_scales = avg_scales/2
avg_scales

# # Reading in Coastal Features Array

#Using the updated file without points removed
embay_dist = gdal.Open(paths[1])

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
(rows, cols)

array

#Setting negatives to nan
array=np.where(array<0, np.nan, array)
array

# # Preparing EHSI (only needs to be run once)

#Opening prior EHSI
ehsi = gdal.Open(paths[2])
band=ehsi.GetRasterBand(1)
array_ehsi=band.ReadAsArray()

ehsi_gt=ehsi.GetGeoTransform()
ehsi_gt, gt

# Because the uncropped ehsi raster created in arcgis pro does not match the embay_dist file used (notably there is a different tile size as seen in the discrepancy between above two outputs), it needs to be resampled with the gdal package (not just the stride by two filter used for the output of the satellite model

# +
# #Fixing Resolution
# source = paths[3]
# output = paths[4]

# xres=gt[1]
# yres=gt[5]
# resample_alg = 'bilinear'

# ds = gdal.Warp(output, source, xRes=xres, yRes=yres, resampleAlg=resample_alg)
# ds = None

# -

# # Reading the EHSI Resampled

#Opening prior EHSI
ehsi = gdal.Open(paths[4])
band=ehsi.GetRasterBand(1)
array_ehsi=band.ReadAsArray()

gt2=ehsi_gt
gt2==gt

#Slight reshaping
print(array_ehsi.shape)
array_ehsi=array_ehsi[:,:-1]
print(array_ehsi.shape)
array_ehsi.shape==array.shape

# The above reshaping implies there could be a mismatch of the grids on the order of 1 in 7000 of the total lenth of the eastern sound grid. This is an acceptable level of error for the purposes of this exercise

#Fixing noise at the top
array_ehsi[:150, :]=0

fig, ax= plt.subplots()
ehsi=plt.imshow(array_ehsi)
fig.colorbar(ehsi, location="right", shrink=.75)

# +
#Rescaling to actual score (max should be 100)
print(array_ehsi.max())
array_ehsi+=27
print(array_ehsi.max())
array_ehsi=np.where(array_ehsi==27, np.nan, array_ehsi)

#Subtracting 20 for highest temp score (FOR NOW)
array_ehsi-=20
# -

fig, ax= plt.subplots()
ehsi=plt.imshow(array_ehsi)
fig.colorbar(ehsi, location="right", shrink=.75)

#bins for reclassification
bins =[100, 25, 24.6, 24.1, 23.7, 23.2, 22.8, 22.3, 21.9, 21.4, 21.0]
bins.reverse()
bins


#Defining reclassification function
def reclassify(array, bins):
    return(10 - np.digitize(array, bins, right=True))


#Testing above function
test_array=[26, 43, 10, 21, 22, 23]
reclassify(test_array, bins)

# # Reading in Data

#This is the source of the model's predictions
df=pd.read_csv(paths[5], index_col=0)
df.head()

# +
#Deprecated
# Drops URI and USGS data (untested as a result of Mystic Harbor Issues)
# df=df.loc[(df["Organization"]!="URI") & (df["Organization"]!="USGS_Discrete")].copy()
# pd.unique(df["Organization"])

# +
#Getting rid of zeroes in array
#array=np.where(array==0, np.nan, array)

# +
# #Displaying figure analogous to earlier version
# plt.figure()
# plt.imshow(array)
# -

#Dropping nas
print(len(df))
df.dropna(subset=["Station ID", "Longitude", "Latitude", "Temperature (C)", "Organization"], inplace=True)
print(len(df))

# At this point there should be 731 datapoints left

pd.unique(df["Organization"])

#Getting list of continuous organizations for alphas
cont_orgs=["STS_Tier_II", "EPA_FISM", "USGS_Cont"]

#Testing alpha logic
cont_error=.1
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

# Initial guesses for model length scales are important because they prevent the model from going into an unstable mode where the middle 2nd length scale parameter blows up.

#Coloring scales
hmin=20
hmax=26
sdmin=0
sdmax=1.2


def model_output(discrete_error, cont_error, year, stride):
    
    #Building Process
    kernel = 1 * RBF(length_scale=avg_scales, length_scale_bounds=(1e-5, 1e5))
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
    plt.title(title, size="xx-large")
    plt.xticks(ticks=np.linspace(0, grid_shape[1], 9), labels=np.round(np.linspace(lon_min, lon_max, 9), 2))
    plt.xlabel("Longitude", size="xx-large")

    plt.yticks(ticks=np.linspace(0, grid_shape[0], 5), labels=np.round(np.linspace(lat_min, lat_max, 5), 2))
    plt.ylabel("Latitude", size="xx-large")

    cbar = fig.colorbar(heatmap, location="right", shrink=.75)
    cbar.set_label('Deg C')

    plt.savefig("Data/Heatmaps/" + title.replace(" ", "_") + "_v3.png")
    plt.show()
    
    #Graphing and Saving STD
    fig, ax = plt.subplots()
    heatmap = plt.imshow(std_grid, vmin=sdmin, vmax=sdmax)
    
    title=str(year) + " Model Error"
    plt.title(title, size="xx-large")
    plt.xticks(ticks=np.linspace(0, grid_shape[1], 9), labels=np.round(np.linspace(lon_min, lon_max, 9), 2))
    plt.xlabel("Longitude", size="xx-large")

    plt.yticks(ticks=np.linspace(0, grid_shape[0], 5), labels=np.round(np.linspace(lat_min, lat_max, 5), 2))
    plt.ylabel("Latitude", size="xx-large")

    cbar=fig.colorbar(heatmap, location="right", shrink=.75)
    cbar.set_label('Standard Deviation in Deg C')

    plt.savefig("Data/Heatmaps/" + title.replace(" ", "_") + "_v3.png")
    plt.show()
    
    return(process.kernel_, y_grid, std_grid)


#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

#Dicts for Outputs
kernels={}
temps={}
errors={}

# ## 2019

#Param
year=2019

kernels[year], temps[year], errors[year] = model_output(discrete_error, cont_error, year, stride)

# ## 2020

#Params
year=2020

kernels[year], temps[year], errors[year] = model_output(discrete_error, cont_error, year, stride)

# ## 2021

#Params
year=2021

kernels[year], temps[year], errors[year] = model_output(discrete_error, cont_error, year, stride)

# # Putting Data Together

# +
agg_heatmap=sum(temps.values())/len(temps.values())

fig, ax = plt.subplots()
heatmap = plt.imshow(agg_heatmap)
# -

fig, ax = plt.subplots()
heatmap = plt.imshow(reclassify(agg_heatmap, bins))

# +
#Making new EHSI from old (min and max should be close to eachother)

#Resampling EHSI
ehsi_resampled=array_ehsi[0::stride, 0::stride].copy()

#Updating EHSI with new temp data
new_ehsi=ehsi_resampled + 2*reclassify(agg_heatmap, bins)
print(np.nanmax(new_ehsi))
print(np.nanmin(new_ehsi))

#Re-adding subtracted temp score to original EHSI
ehsi_resampled+=20
print(np.nanmax(ehsi_resampled))
print(np.nanmin(ehsi_resampled))

# +
#plt.imshow(np.abs(ehsi_resampled-new_ehsi)>6)
# -

plt.imshow(ehsi_resampled, vmin=40, vmax=100)

plt.imshow(new_ehsi, vmin=40, vmax=100)

# +
# #Finding places where no new temp exists for ehsi
# fig, ax = plt.subplots()
# heatmap = plt.imshow((~np.isnan(ehsi_resampled) & np.isnan(agg_heatmap)))
# -

#Graphing param
plt.rcParams["figure.figsize"]=(30, 25)

from IPython.core.display import display, HTML
display(HTML(
    '<style>'
        '#notebook { padding-top:0px !important; } ' 
        '.container { width:100% !important; } '
        '.end_space { min-height:0px !important; } '
    '</style>'
))

# +
#Side by side plot (surprisingly complicated)

fig, ax = plt.subplots(1, 2)
heatmap = ax[0].imshow(ehsi_resampled, vmin = 40, vmax= 100)

ax[0].set_title("Original EHSI", size="x-large")
ax[0].set_xticks(ticks=np.linspace(0, ehsi_resampled.shape[1], 9), labels=np.round(np.linspace(lon_min, lon_max, 9), 2))
ax[0].set_xlabel("Longitude", size="x-large")

ax[0].set_yticks(ticks=np.linspace(0, ehsi_resampled.shape[0], 5), labels=np.round(np.linspace(lat_min, lat_max, 5), 2))
ax[0].set_ylabel("Latitude", size="x-large")

# cbar = fig.colorbar(heatmap, location="right", shrink=.75)
# cbar.set_label('Deg C', rotation=270, size="large")

new_heatmap = ax[1].imshow(new_ehsi, vmin = 40, vmax= 100)

ax[1].set_title("EHSI with Embayment Temperature Model", size="x-large")
ax[1].set_xticks(ticks=np.linspace(0, ehsi_resampled.shape[1], 9), labels=np.round(np.linspace(lon_min, lon_max, 9), 2))
ax[1].set_xlabel("Longitude", size="x-large")

ax[1].set_yticks(ticks=np.linspace(0, ehsi_resampled.shape[0], 5), labels=np.round(np.linspace(lat_min, lat_max, 5), 2))
ax[1].set_ylabel("Latitude", size="x-large")

# -

ehsi_diff=new_ehsi-ehsi_resampled

# +
fig, ax = plt.subplots(1, 1)
diff=ax.imshow(ehsi_diff, cmap=mpl.colormaps["seismic"], vmin = -10, vmax= 10)

ax.set_title("EHSI Difference with New Embayment Temperature Model", size="xx-large")
ax.set_xticks(ticks=np.linspace(0, ehsi_resampled.shape[1], 9), labels=np.round(np.linspace(lon_min, lon_max, 9), 2))
ax.set_xlabel("Longitude", size="x-large")

ax.set_yticks(ticks=np.linspace(0, ehsi_resampled.shape[0], 5), labels=np.round(np.linspace(lat_min, lat_max, 5), 2))
ax.set_ylabel("Latitude", size="x-large")

cbar = fig.colorbar(diff, location="bottom", ax=ax)
cbar.set_label('Difference in EHSI', size="large")
plt.savefig("Data/Heatmaps/EHSI/EHSI_Difference.png")

# +
#Graphing and Saving Heatmap

fig, ax = plt.subplots()
heatmap = plt.imshow(agg_heatmap)
title="2019-2021 Embayment Temperature Heatmap"
plt.title(title, size="x-large")
plt.xticks(ticks=np.linspace(0, agg_heatmap.shape[1], 9), labels=np.linspace(lon_min, lon_max, 9))
plt.xlabel("Longitude", size="x-large")

plt.yticks(ticks=np.linspace(0, agg_heatmap.shape[0], 5), labels=np.linspace(lat_min, lat_max, 5))
plt.ylabel("Latitude", size="x-large")

cbar = fig.colorbar(heatmap, location="right", shrink=.75)
cbar.set_label('Deg C', rotation=270, size="large")

#plt.savefig("Data/Heatmaps/" + title.replace(" ", "_") + "_v3.png")
plt.show()

# +
fig, ax = plt.subplots()
heatmap = plt.imshow(reclassify(agg_heatmap, bins))
title="2019-2021 Embayment Temperature Reclassified"
plt.title(title, size="x-large")
plt.xticks(ticks=np.linspace(0, agg_heatmap.shape[1], 9), labels=np.linspace(lon_min, lon_max, 9))
plt.xlabel("Longitude", size="x-large")

plt.yticks(ticks=np.linspace(0, agg_heatmap.shape[0], 5), labels=np.linspace(lat_min, lat_max, 5))
plt.ylabel("Latitude", size="x-large")

cbar = fig.colorbar(heatmap, location="right", shrink=.75)
cbar.set_label('Eelgrass Score (1-10)', rotation=270, size="large")

#plt.savefig("Data/Heatmaps/" + title.replace(" ", "_") + "_v3.png")
plt.show()
# -

# # Outputting TIFF

new_gt=(gt[0], gt[1]*10, gt[2], gt[3], gt[4], gt[5]*10)
new_gt

#Downloading difference as geotiff
driver = gdal.GetDriverByName("GTiff")
driver.Register()
outds = driver.Create(output_path, xsize=ehsi_diff.shape[1], ysize=ehsi_diff.shape[0], bands=1, eType=gdal.GDT_Int16)
outds.SetGeoTransform(new_gt)
outds.SetProjection(proj)
outband = outds.GetRasterBand(1)
outband.WriteArray(ehsi_diff)
outband.SetNoDataValue(np.nan)
outband.FlushCache()
outband=None
outds=None

#Testing saved geotiff
test=gdal.Open(output_path)
band=test.GetRasterBand(1)
array=band.ReadAsArray()
plt.imshow(array)


